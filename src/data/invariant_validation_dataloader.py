import pandas as pd
import torch.distributed as dist
from monai import transforms
from monai.data import CacheDataset, Dataset, ThreadDataLoader, partition_dataset, DataLoader
from monai.utils import first
import random
import os
import numpy as np

from monai.transforms import MapTransform, Resize
from monai.config import KeysCollection



class FixedResizeImg(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            allow_missing_keys: bool = False,

    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        img = d["image"]
        d["dimension"] = [img.shape[1], img.shape[2], img.shape[3]]
        for resolution_change in [1.2,1.6,2,2.4,2.8,3.2,3.6,4]:
            x_len = img.shape[1]
            y_len = img.shape[2]
            z_len = img.shape[3]
            new_x_len = int(8 * ((x_len / resolution_change) // 8))
            new_y_len = int(8 * ((y_len / resolution_change) // 8))
            new_z_len = int(8 * ((z_len / resolution_change) // 8))

            resize_transform = Resize(spatial_size=[new_x_len, new_y_len, new_z_len], mode="area")
            resolution_change_actual = x_len / new_x_len
            img_resized = resize_transform(img)
            d["image_res_" + str(resolution_change)] = img_resized
            d["dimension_res_" + str(resolution_change)] = [img_resized.shape[1], img_resized.shape[2], img_resized.shape[3]]
            d["resolution_res_" + str(resolution_change)] = [resolution_change_actual, resolution_change_actual, resolution_change_actual]


        d["image"] = img
        d["dimension"] = [img.shape[1], img.shape[2], img.shape[3]]
        d["resolution"] = [1.0, 1.0, 1.0]


        return d


class GetResolutionArrays(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            allow_missing_keys: bool = False,

    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for resolution_change in [1.2,1.6,2,2.4,2.8,3.2,3.6,4]:
            res = d["resolution_res_" + str(resolution_change)]
            res_array = np.array(res, dtype=np.float32)
            d["resolution_array_res_" + str(resolution_change)] = res_array

        res = d["resolution"]
        res_array = np.array(res, dtype=np.float32)
        d["resolution_array"] = res_array

        return d



def get_data_dicts(ids_path: str, shuffle: bool = False, first_n=False):

    """Get data dicts for data loaders."""
    if ids_path.endswith(".csv"):
        df = pd.read_csv(ids_path, sep=",")
        if shuffle:
            df = df.sample(frac=1, random_state=1)
        df = list(df)
        data_dicts = []
        for row in df:
            data_dicts.append({"image": (row)})
    elif os.path.isdir(ids_path):
        data_dicts = []
        for sample_file in os.listdir(ids_path):
            data_dicts.append({"image":ids_path + "/" + sample_file})
        if shuffle:
            random.seed(a=1)
            random.shuffle(data_dicts)
    else:
        raise ValueError(f"Subject path is neither a csv or Directory of files")


    if first_n is not False:
        data_dicts = data_dicts[:first_n]

    print(f"Found {len(data_dicts)} subjects.")
    if dist.is_initialized():
        print(dist.get_rank())
        print(dist.get_world_size())
        return partition_dataset(
            data=data_dicts,
            num_partitions=dist.get_world_size(),
            shuffle=True,
            seed=0,
            drop_last=False,
            even_divisible=True,
        )[dist.get_rank()]
    else:
        return data_dicts


def get_invariant_validation_loader(
    batch_size: int,
    training_ids: str,
    validation_ids: str,
    only_val: bool = False,
    augmentation: bool = True,
    drop_last: bool = False,
    num_workers: int = 8,
    num_val_workers: int = 3,
    cache_data=True,
    first_n=None,

):
    image_names = ["image"]
    resolution_names = ["resolution"]
    resolution_array_names = ["resolution_array"]
    dimension_names = ["dimension"]
    for resolution_change in [1.2, 1.6, 2, 2.4, 2.8, 3.2, 3.6, 4]:
        image_names.append("image_res_" + str(resolution_change))
        resolution_names.append("resolution_res_" + str(resolution_change))
        resolution_array_names.append("resolution_array_res_" + str(resolution_change))
        dimension_names.append("dimension_res_" + str(resolution_change))

    # NEED TO FIX THE VAL RESIZE TRANSFORM HERE TO CHANGE THE RESOLUTION INPUT
    val_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            FixedResizeImg(keys="image"),
            transforms.SignalFillEmptyd(keys=image_names),
            transforms.ThresholdIntensityd(keys=image_names, threshold=1, above=False, cval=1.0),
            transforms.ThresholdIntensityd(keys=image_names, threshold=0, above=True, cval=0),
            GetResolutionArrays(keys=resolution_names),
            transforms.ToTensord(keys=image_names+resolution_array_names),
        ]
    )


    val_dicts = get_data_dicts(validation_ids, shuffle=False, first_n=first_n)
    if first_n:
        val_dicts = val_dicts[:first_n]

    if cache_data:
        val_ds = CacheDataset(
            data=val_dicts,
            transform=val_transforms,
        )
    else:
        val_ds = Dataset(
            data=val_dicts,
            transform=val_transforms,
        )

    val_loader = ThreadDataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_val_workers,
        drop_last=drop_last,
        pin_memory=False,
    )

    return val_loader

