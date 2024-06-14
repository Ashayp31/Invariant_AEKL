import os
from enum import Enum
from logging import Logger
from math import floor
from typing import Tuple, Union, List, Dict

import pandas as pd
from ignite.engine import _prepare_batch
from ignite.utils import convert_tensor
from monai.data import PersistentDataset, Dataset, DataLoader
from monai.transforms import (
    Compose,
    ToNumpyd,
    ScaleIntensityd,
    RandGaussianNoised,
    RandShiftIntensityd,
    RandAdjustContrastd,
    RandHistogramShiftd,
    CenterSpatialCropd,
    RandAffined,
    ThresholdIntensityd,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    ToTensord,
    Rand3DElasticd,
    RandGaussianSmoothd,
    EnsureChannelFirstd
)
from monai.transforms.io.dictionary import LoadImaged
import torch


class VQVAEModes(Enum):
    TRAINING = "training"
    EXTRACTING = "extracting"
    DECODING = "decoding"
    UNCERTAINTY = "uncertainty"


def get_data_flow(
    config: dict, logger: Logger = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Constructs the data ingestion logic. There are different approaches for full-image and patch-based training due to
    gpu usage efficiency.

    The following fields are needed in config (in order of appearance):

        training_subjects (Union[str,Tuple[str, ...]]): Either one or multiple absolute paths to either a folder, csv or
        tsv. If it is a csv or tsv, it is expected that a "path" column is present and holds absolute paths to
        individual files. Those subjects will be used for the training dataset. For 'decoding' a Tuple of paths it is
        expected matching that the number of elements in num_embeddings each element should point to either folder or
        csv/tsv are pointing towards .npy files otherwise a single element is expected pointing towards .nii.gz files.

        mode (str): For which mode the data flow is being created. It should be one of the following: 'training',
        'extracting',  'decoding'.

        num_embeddings (Tuple[int,...]): How many atomic elements each quantization elements has. This is used to
        determine the number of quantizations to be loaded.

        normalize (bool): Whether or not the training and validation datasets are 0-1 normalised. Defaults to True.

        roi (Tuple[int,int,int]): The region of interest in the image that will be cropped out and forward modified. If
        None then no cropping will happen. Defaults to None.

        patch_size (Tuple[int,int,int]): How big the randomly cropped area will be for training data. If None no random
        crop will happen. Defaults to None.

        persistent_training_dataset (bool): Whether or not a monai.data.dataset.PersistentDataset is going to be used.
        Defaults to False.

        cache_dir (str): Where the monai.data.dataset.PersistentDataset will store the cached versions of the data.

        batch_size (int): The batch size that will be used to train the network. Defaults to 2.

        num_workers (int): How many threads will be spawn to load batches. Defaults to 8.

        prefetch_factor (int): How many batches each thread will try to keep as a buffer. Defaults to 6.

        validation_subjects (Union[str,Tuple[str, ...]]): Either one or multiple absolute paths to either a folder, csv
        or tsv. If it is a csv or tsv, it is expected that a "path" column is present and holds absolute paths to
        individual files. Those subjects will be used for the training dataset. For 'decoding' a Tuple of paths it is
        expected matching that the number of elements in num_embeddings each element should point to either folder or
        csv/tsv are pointing towards .npy files otherwise a single element is expected pointing towards .nii.gz files.

        eval_patch_size (Tuple[int,int,int]): How big the randomly cropped area will be for evaluation data.
        If None no random crop will happen. Defaults to None.

        eval_batch_size (int): The batch size that will be used to evaluate the network. Defaults to 1.

    Args:
        config (dict): Configuration dictionary that holds all the required parameters.

        logger (Logger): Logger that will be used to report DataLoaders parameters.

    Returns:
        DataLoader: Training DataLoader which has data augmentations

        DataLoader: Evaluation DataLoader for the validation data. No data augmentations.

        DataLoader: Evaluation DataLoader for the training data. No data augmentations.
    """

    def get_subjects(
        paths: Union[str, Tuple[str, ...]], mode: str, ct_paths=str,
    ) -> List[Dict[str, str]]:
        if isinstance(paths, str):
            paths = [paths]
        else:
            paths = list(paths)

        files = []
        for path in paths:
            print(path)
            if os.path.isdir(path):
                files.append(
                    [os.path.join(path, os.fsdecode(f)) for f in os.listdir(path)]
                )
            elif os.path.isfile(path):
                if path.endswith(".csv"):
                    files.append(
                        pd.read_csv(filepath_or_buffer=path, sep=",")["path"].to_list()
                    )
                elif path.endswith(".tsv"):
                    files.append(
                        pd.read_csv(filepath_or_buffer=path, sep="\t")["path"].to_list()
                    )
            else:
                raise ValueError(
                    "Path is neither a folder (to source all the files inside) or a csv/tsv with file paths inside."
                )

        if ct_paths:
            if os.path.isfile(ct_paths):
                if ct_paths.endswith(".csv"):
                    ct_paths_file = pd.read_csv(
                        filepath_or_buffer=ct_paths, sep=","
                    )
                elif ct_paths.endswith(".tsv"):
                    ct_paths_file = pd.read_csv(
                        filepath_or_buffer=ct_paths, sep="\t"
                    )
            else:
                raise ValueError("Cropping Path is not a csv/tsv with file paths inside.")
        

        subjects = []

        mia_subjects = 0
        for file in files[0]:
            if file.endswith(".nii.gz"):
                subject = {"MRI": file}
                subject_name = os.path.basename(file)

            else:
                raise ValueError(f"Path given is not a .nii.gz file, but {file} ")


            if ct_paths:
                try:
                    encoding_subject = ct_paths_file.loc[
                        ct_paths_file["subject"] == subject_name, "encoding"
                    ].values[0]

                except IndexError:
                    print("Cannot find Encoding npy file for for ", file)
                    mia_subjects += 1
                    valid_subject = False
                    continue

                subject["encoding"] = encoding_subject

            subjects.append(subject)
        return subjects

    training_subjects = get_subjects(
        paths=config["training_subjects"], mode=config["mode"], ct_paths=config["ct_conditioning_path"],
    )

    if config["mode"] == VQVAEModes.DECODING.value:
        raise ValueError("Invalid Mode for autoencoders")
    else:
        keys = ["MRI"]
        if config["ct_conditioning_path"] is not None:
            keys.append("encoding")
        # Image wide transformations
        training_transform = [LoadImaged(keys=keys), EnsureChannelFirstd(keys=keys)]

        if config.get("normalize", True):
            training_transform += [ScaleIntensityd(keys=keys, minv=0.0, maxv=1.0)]

        if config.get("roi", None):
            training_transform += [
                CenterSpatialCropd(keys=keys, roi_size=config["roi"])
            ]

        if config.get("patch_size", None):
            training_transform += [
                RandSpatialCropd(
                    keys=keys,
                    roi_size=config["patch_size"],
                    random_size=False,
                    random_center=True,
                )
            ]
        else:
            training_transform += [

                Rand3DElasticd(
                        keys=keys,
                        prob=0.7,
                        sigma_range = [1.0, 2.0],
                        magnitude_range = [2.0, 5.0],
                        rotate_range = [0, 0, 0.04],
                        translate_range=[6, 6, 0],
                        scale_range=[0.05, 0.05, 0],
                        padding_mode="zeros"
                    ),
                    ]

        # Patch/Image agnostic transformations

        training_transform += [
            RandAdjustContrastd(keys=keys, prob=0.3, gamma=(0.98, 1.02)),
            RandShiftIntensityd(keys=keys, prob=0.3, offsets=(0.0, 0.025)),
            RandGaussianNoised(keys=keys, prob=0.3, mean=0.0, std=0.01),
            RandGaussianSmoothd(keys=keys, prob=0.3, sigma_x=(0.15, 0.5), sigma_y=(0.15, 0.5),
                                sigma_z=(0.15, 0.5))
        ]

        # Patch based transformations
        if config.get("patch_size", None):
            training_transform += [
                RandFlipd(keys=keys, prob=0.2, spatial_axis=0),
                RandFlipd(keys=keys, prob=0.2, spatial_axis=1),
                RandFlipd(keys=keys, prob=0.2, spatial_axis=2),
                RandRotate90d(keys=keys, prob=0.2, spatial_axes=(0, 1)),
                RandRotate90d(keys=keys, prob=0.2, spatial_axes=(1, 2)),
                RandRotate90d(keys=keys, prob=0.2, spatial_axes=(0, 2)),
            ]

        training_transform += [
            ThresholdIntensityd(keys=keys, threshold=1, above=False, cval=1.0),
            ThresholdIntensityd(keys=keys, threshold=0, above=True, cval=0),
            ToTensord(keys=keys),
        ]

    training_transform = Compose(training_transform)

    training_dataset = (
        PersistentDataset(
            data=training_subjects,
            transform=training_transform,
            cache_dir=config["cache_dir"],
        )
        if config.get("persistent_training_dataset", False)
        else Dataset(data=training_subjects, transform=training_transform)
    )

    training_loader = DataLoader(
        training_dataset,
        batch_size=config.get("batch_size", 2),
        num_workers=config.get("num_workers", 8),
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=config.get("prefetch_factor", 6),
        # Forcefully setting it to false due to this pull request
        # not being in PyTorch 1.7.1
        # https://github.com/pytorch/pytorch/pull/48543
        persistent_workers=False,
    )

    evaluation_subjects = get_subjects(
        paths=config["validation_subjects"], mode=config["mode"], ct_paths=config["ct_conditioning_path"],
    )


    if config["mode"] == VQVAEModes.DECODING.value:
        raise ValueError("Invalid Mode for autoencoders")
    else:
        keys = ["MRI"]
        if config["ct_conditioning_path"] is not None:
            keys.append("encoding")
        evaluations_transform = [LoadImaged(keys=keys), EnsureChannelFirstd(keys=keys)]

        if config.get("normalize", True):
            evaluations_transform += [ScaleIntensityd(keys=keys, minv=0.0, maxv=1.0)]

        if config.get("roi", None):
            evaluations_transform += [
                CenterSpatialCropd(keys=keys, roi_size=config["roi"])
            ]

        if config.get("eval_patch_size", None):
            evaluations_transform += [
                RandSpatialCropd(
                    keys=keys,
                    roi_size=config["eval_patch_size"],
                    random_size=False,
                    random_center=True,
                )
            ]

        evaluations_transform += [ToTensord(keys=keys)]
        evaluations_transform = Compose(evaluations_transform)


    evaluation_dataset = Dataset(
        data=evaluation_subjects, transform=evaluations_transform
    )

    evaluation_loader = DataLoader(
        evaluation_dataset,
        batch_size=config.get("eval_batch_size", 1),
        num_workers=config.get("num_workers", 8),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        prefetch_factor=config.get("prefetch_factor", 6),
        persistent_workers=False,
    )

    training_evaluation_dataset = Dataset(
        data=training_subjects, transform=evaluations_transform
    )

    training_evaluation_loader = DataLoader(
        training_evaluation_dataset,
        batch_size=config.get("eval_batch_size", 1),
        num_workers=config.get("num_workers", 8),
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        prefetch_factor=config.get("prefetch_factor", 6),
        persistent_workers=False,
    )

    if logger:
        logger.info("Dataflow setting:")
        logger.info("\tTraining:")
        if config.get("patch_size", None):
            logger.info(f"\t\tPatch Size: {config['patch_size']}")
        logger.info(f"\t\tLength: {len(training_loader)}")
        logger.info(f"\t\tBatch Size: {training_loader.batch_size}")
        logger.info(f"\t\tPin Memory: {training_loader.pin_memory}")
        logger.info(f"\t\tNumber of Workers: {training_loader.num_workers}")
        logger.info(f"\t\tPrefetch Factor: {training_loader.prefetch_factor}")
        logger.info("\tValidation:")
        if config.get("eval_patch_size", None):
            logger.info(f"\t\tPatch Size: {config['eval_patch_size']}")
        logger.info(f"\t\tLength: {len(evaluation_loader)}")
        logger.info(f"\t\tBatch Size: {evaluation_loader.batch_size}")
        logger.info(f"\t\tPin Memory: {evaluation_loader.pin_memory}")
        logger.info(f"\t\tNumber of Workers: {evaluation_loader.num_workers}")
        logger.info(f"\t\tPrefetch Factor: {evaluation_loader.prefetch_factor}")

    config["epoch_length"] = len(training_loader)

    return training_loader, evaluation_loader, training_evaluation_loader


def get_ms_ssim_window(config: dict, logger: Logger = None) -> int:
    """
    Calculates the window size of the gaussian kernel for the MS-SSIM if the smallest dimension of the image is
    lower than 160 (requirement of the default parameters of MS-SSIM).

    It expects to find a 'eval_patch_size' field that can be either None or a Tuple[int,...].

    Args:
        config (dict): Dictionary that holds the whole configuration
        logger (Logger): Logger for printing the logic

    Returns:
        int: Half of the maximum kernel size allowed or next odd int
    """
    if config["eval_patch_size"]:
        min_ps = min(config["eval_patch_size"])

        if min_ps > 160:
            win_size = 11
        else:
            win_size = floor(((min_ps / 2 ** 4) + 1) / 2)

            if win_size <= 1:
                raise ValueError(
                    "Window size for MS-SSIM can't be calculated. Please increase patch_size's smallest dimension."
                )

            # Window size must be odd
            if win_size % 2 == 0:
                win_size += 1
    else:
        win_size = 11

    if logger:
        logger.info("MS-SSIM window calculation:")
        if config["eval_patch_size"]:
            logger.info(f"\tMinimum spatial dimension: {min_ps}")
        logger.info(f"\tWindow size {win_size}")

    return win_size


def prepare_batch(batch, add_ct_channel=False, device=None, non_blocking=False):
    """
    Prepare batch function that allows us to train an unsupervised mode by using the SupervisedTrainer
    """

    x_input = x_target = batch["MRI"]
    if add_ct_channel:
        ct_input = batch["encoding"]
        x_input = torch.concat((x_input, ct_input), dim=1)
    return _prepare_batch((x_input, x_target), device, non_blocking)


def prepare_decoding_batch(
    batch, num_quantization_levels, device=None, non_blocking=False
):
    """
    Prepare batch function that allows us to train an unsupervised mode by using the SupervisedTrainer
    """
    x_input = x_target = [
        convert_tensor(
            batch[f"quantization_{i}"].long(), device=device, non_blocking=non_blocking
        )
        for i in range(num_quantization_levels)
    ]
    return x_input, x_target
