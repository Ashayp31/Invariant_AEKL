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


class RandResizeImg(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            allow_missing_keys: bool = False,

    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        rand_val = random.uniform(0, 1)
        img = d["image"]
        rand_resolution_change = 1
        if rand_val < 0.6:
            if min(img.shape[1:]) < 96:
                rand_resolution_change = 1 + random.uniform(0, 1)
            else:
                rand_resolution_change = 1 + random.uniform(0, 2)
            x_len = img.shape[1]
            y_len = img.shape[2]
            z_len = img.shape[3]
            new_x_len = int(16 * ((x_len / rand_resolution_change) // 16))
            new_y_len = int(16 * ((y_len / rand_resolution_change) // 16))
            new_z_len = int(16 * ((z_len / rand_resolution_change) // 16))

            resize_transform = Resize(spatial_size=[new_x_len, new_y_len, new_z_len], mode="area")

            img = resize_transform(img)

        d["image"] = img
        d["resolution"] = [1*rand_resolution_change, 1*rand_resolution_change, 1*rand_resolution_change]
        d["dimension"] = [img.shape[1], img.shape[2], img.shape[3]]

        return d



class CropWithoutPadding(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        use_crop_input: bool = False,
        allow_missing_keys: bool = False,

    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.use_crop = use_crop_input

    def __call__(self, data):
        d = dict(data)

        rand_val_x = random.uniform(0, 1)
        rand_val_y = random.uniform(0, 1)
        rand_val_z = random.uniform(0, 1)
        image_data = d["image"]
        x_len = image_data.shape[1]
        y_len = image_data.shape[2]
        z_len = image_data.shape[3]

        if rand_val_x > 0.4:

            first_crop_x = random.randint(0, x_len - 64) if x_len > 64 else 0
            second_crop_x = random.randint(first_crop_x, x_len)
            image_size_x = second_crop_x - first_crop_x
            new_image_size_x = max(image_size_x, 64)
            second_crop_x = min(x_len,first_crop_x + new_image_size_x)
            new_image_size_x = min(144,((second_crop_x - first_crop_x)//16) * 16)
            second_crop_x = first_crop_x + new_image_size_x
            image_data = image_data[:,first_crop_x:second_crop_x,:,:]
        else:
            if x_len > 144:
                first_crop_x = random.randint(0, x_len - 144)
                second_crop_x = first_crop_x + 144
                image_data = image_data[:,first_crop_x:second_crop_x,:,:]
            else:
                first_crop_x = 0
                second_crop_x = x_len

        if rand_val_y > 0.4:


            first_crop_y = random.randint(0, y_len - 64) if y_len > 64 else 0
            second_crop_y = random.randint(first_crop_y, y_len)
            image_size_y = second_crop_y - first_crop_y
            new_image_size_y = max(image_size_y, 64)
            second_crop_y = min(y_len,first_crop_y + new_image_size_y)
            new_image_size_y = min(144,((second_crop_y - first_crop_y)//16) * 16)
            second_crop_y = first_crop_y + new_image_size_y
            image_data = image_data[:,:,first_crop_y:second_crop_y,:]
        else:
            if y_len > 144:
                first_crop_y = random.randint(0, y_len - 144)
                second_crop_y = first_crop_y + 144
                image_data = image_data[:,:,first_crop_y:second_crop_y,:]
            else:
                first_crop_y = 0
                second_crop_y = y_len

        if rand_val_z > 0.4:

            first_crop_z = random.randint(0, z_len - 64) if z_len > 64 else 0
            second_crop_z = random.randint(first_crop_z, z_len)
            image_size_z = second_crop_z - first_crop_z
            new_image_size_z = max(image_size_z, 64)
            second_crop_z = min(z_len,first_crop_z + new_image_size_z)
            new_image_size_z = min(144,((second_crop_z - first_crop_z)//16) * 16)
            second_crop_z = first_crop_z + new_image_size_z
            image_data = image_data[:,:,:,first_crop_z:second_crop_z]

        else:
            if z_len > 144:
                first_crop_z = random.randint(0, z_len - 144)
                second_crop_z = first_crop_z + 144
                image_data = image_data[:,:,:,first_crop_z:second_crop_z]
            else:
                first_crop_z = 0
                second_crop_z = z_len


        d["image"] = image_data
        d["crop"] = [first_crop_x/x_len, second_crop_x/x_len,first_crop_y/y_len, second_crop_y/y_len,first_crop_z/z_len, second_crop_z/z_len]
        d["dimension"] = [image_data.shape[1], image_data.shape[2], image_data.shape[3]]

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
            data_dicts.append({"image": (row), "resolution": [1.0,1.0,1.0], "dimension": [100,100,100]})
    elif os.path.isdir(ids_path):
        data_dicts = []
        for sample_file in os.listdir(ids_path):
            data_dicts.append({"image":ids_path + "/" + sample_file, "resolution": [1.0,1.0,1.0], "dimension": [100,100,100]})
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


def get_training_data_loader(
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
    is_grayscale=False,
    add_vflip=False,
    add_hflip=False,
    image_size=None,
    image_roi=None,
    spatial_dimension=2,
    has_coordconv=True,
    apply_geometric_aug=False,
):
    # Define transformations
    # resize_transform = (
    #     transforms.ResizeD(keys=["image"], spatial_size=(image_size,) * spatial_dimension)
    #     if image_size
    #     else lambda x: x
    # )
    resize_transform = (
        transforms.Resized(keys=["image"], spatial_size=(136,120,120))
    )

    val_resize_transform = (
        transforms.Resized(keys=["image"], spatial_size=(160,144,144))
    )

    # resize_transform = (
    #     transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=(192,192,192))
    # )

    central_crop_transform = (
        transforms.CenterSpatialCropD(keys=["image"], roi_size=[256,256,256])
        if image_roi
        else lambda x: x
    )


    train_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            # transforms.AddChanneld(keys=["image"]) if not has_coordconv else transforms.EnsureChannelFirstd(keys=["image"], channel_dim=-1),
            transforms.EnsureChannelFirstd(keys=["image"]),
            CropWithoutPadding(keys=["image", "resolution", "dimension"]) if apply_geometric_aug else lambda x: x,
            RandResizeImg(keys=["image", "dimension"]) if apply_geometric_aug else lambda x: x,
            # transforms.Lambdad(keys="image", func=lambda x: x[0, None, ...])
            # if is_grayscale
            # else lambda x: x,  # needed for BRATs data with 4 modalities in 1
            # central_crop_transform,
            resize_transform if not apply_geometric_aug else lambda x: x,
            # transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            # transforms.Rand3DElasticd(
            #     keys=["image"],
            #     prob=1.0,
            #     sigma_range=[1.0, 2.0],
            #     magnitude_range=[2.0, 5.0],
            #     rotate_range=None,
            #     translate_range=[6, 6, 0],
            #     scale_range=[0.05, 0.05, 0],
            #     padding_mode='zeros'
            # ),
            transforms.SignalFillEmptyd(keys=["image"]),
            transforms.ThresholdIntensityd(keys=["image"], threshold=1, above=False, cval=1.0),
            transforms.ThresholdIntensityd(keys=["image"], threshold=0, above=True, cval=0),
            transforms.RandFlipD(keys=["image"], spatial_axis=0, prob=1.0)
            if add_vflip
            else lambda x: x,
            transforms.RandFlipD(keys=["image"], spatial_axis=1, prob=1.0)
            if add_hflip
            else lambda x: x,
            transforms.ToTensord(keys=["image"]),
        ]
    )


    val_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            val_resize_transform,
            transforms.SignalFillEmptyd(keys=["image"]),
            transforms.ThresholdIntensityd(keys=["image"], threshold=1, above=False, cval=1.0),
            transforms.ThresholdIntensityd(keys=["image"], threshold=0, above=True, cval=0),
            transforms.RandFlipD(keys=["image"], spatial_axis=0, prob=1.0)
            if add_vflip
            else lambda x: x,
            transforms.RandFlipD(keys=["image"], spatial_axis=1, prob=1.0)
            if add_hflip
            else lambda x: x,
            transforms.ToTensord(keys=["image"]),
        ]
    )


    # no augmentation for now
    if augmentation:
        train_transforms = train_transforms
    else:
        train_transforms = train_transforms

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

    if only_val:
        return val_loader

    train_dicts = get_data_dicts(training_ids, shuffle=False, first_n=first_n)

    if cache_data:
        train_ds = CacheDataset(
            data=train_dicts,
            transform=train_transforms,
        )
    else:
        train_ds = Dataset(
            data=train_dicts,
            transform=train_transforms,
        )

    train_loader = ThreadDataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=False)

    return train_loader, val_loader
