# SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and contributors.
# SPDX-License-Identifier: Apache-2.0

import math
import random
from collections.abc import Hashable
from typing import Dict, List, Tuple, Optional

import monai.transforms as mt
from monai.transforms.spatial.functional import rotate
import torch

from autopet3.datacentric.utils import get_file_dict_nn, read_split
import os

import numpy as np


class CustomSampleCropd(mt.Transform):
    def __init__(
        self,
        keys: List[str],
        label_key: str,
        roi_size: Tuple,
        border_pad: Tuple[int, int, int] = (64, 80, 56),
        return_pad: bool = False,
        prob: float = 0.5,
    ):
        """Custom sampling class for sampling foreground and background samples.
        Args:
            keys (List[str]): List of keys.
            label_key (str): Key for labels.
            roi_size (Tuple[int, int, int]): Size of the ROI.
            border_pad (Tuple[int, int, int], optional): Padding for borders. Defaults to (64, 80, 56).
            return_pad (bool, optional): Flag to return padding. Defaults to False.
            prob (float, optional): Probability. Defaults to 0.5.

        """
        self.keys = keys
        self.label_key = label_key
        self.roi_size = roi_size
        self.prob = prob
        self.return_pad = return_pad
        self.crop_content = mt.CropForegroundd(keys=self.keys, source_key="ct", allow_smaller=True)
        self.pad = mt.BorderPadd(keys=self.keys, spatial_border=border_pad)
        self.rand_crop = mt.RandSpatialCropd(keys=self.keys, roi_size=self.roi_size)
        self.foreground_crop = mt.RandCropByPosNegLabeld(
            keys=self.keys, label_key=self.label_key, spatial_size=self.roi_size, pos=1.0, neg=0.0, num_samples=1
        )

    def __call__(self, data: Dict[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        label = data[self.label_key]
        data = self.crop_content(data)
        data = self.pad(data)

        if self.return_pad:
            return data
        if torch.any(label) and random.random() > self.prob:
            return self.foreground_crop(data)[0]
        return self.rand_crop(data)


class Unpackd(mt.Transform):
    def __init__(self, keys: List[str]):
        """Unpackd is a transform that extracts specified keys from a dictionary and returns them as a tuple.
        Args:
           keys (List[str]): The keys to extract from the dictionary.
        Returns:
           tuple: A tuple containing the values corresponding to the specified keys.

        """
        self.keys = keys

    def __call__(self, data: Dict[Hashable, torch.Tensor]) -> tuple:
        return tuple([data[k] for k in self.keys])

def crop_tensor(input_tensor, center, crop_size, pad_mode='constant', pad_kwargs=None):
    """
    Crops and pads an input tensor based on the specified center and crop size. Padding can be customized.

    Parameters:
    - input_tensor (torch.Tensor): The input tensor with shape (c, x, y) or (c, x, y, z).
    - center (tuple): The center coordinates of the crop (x, y) or (x, y, z).
    - crop_size (tuple): The size of the crop (width, height) or (width, height, depth).
    - pad_mode (str): The mode to use for padding (see torch.nn.functional.pad documentation).
    - pad_kwargs (dict, optional): Additional keyword arguments for padding.

    Returns:
    - torch.Tensor: The cropped and possibly padded tensor.
    Reference: taken from batchgenerators v2 https://github.com/MIC-DKFZ/batchgenerators
    """
    if pad_kwargs is None:
        pad_kwargs = {'value': 0}

    # Calculate dimensions
    dim = len(center)  # Spatial dimensions
    assert len(crop_size) == dim, "Crop size and center must have the same number of dimensions"
    assert input_tensor.ndim - 1 == dim, "Crop size and input_tensor must have the same number of spatial dimensions"

    spatial_shape = input_tensor.shape[-dim:]
    start = [max(0, cen - cs // 2) for cen, cs in zip(center, crop_size)]
    end = [min(sh, st + cs) for sh, st, cs in zip(spatial_shape, start, crop_size)]

    # Calculate padding
    padding_needed = [(cs - (e - s)) for cs, s, e in zip(crop_size, start, end)]
    pad_before = [max(0, - (cen - cs // 2)) for cen, cs in zip(center, crop_size)]
    pad_after = [pn - pb for pn, pb in zip(padding_needed, pad_before)]

    # Adjust start and end for the case where the crop is entirely outside the input tensor
    start = [min(max(0, s), sh) for s, sh in zip(start, spatial_shape)]
    end = [max(min(e, sh), 0) for e, sh in zip(end, spatial_shape)]

    # Perform crop
    slices = [slice(None)] + [slice(s, e) for s, e in zip(start, end)]
    cropped = input_tensor[tuple(slices)]

    # Pad
    pad_width = sum([[b, a] for b, a in zip(pad_before[::-1], pad_after[::-1])], [])
    if any(pad_width):
        cropped = torch.nn.functional.pad(cropped, pad_width, mode=pad_mode, **pad_kwargs)

    return cropped

# A custom monai transform called Misalign that randomly shifts the selected channel(s) of the input data
# by a random number of pixels in the range [-max_shift, max_shift], different for all directions. This transform is useful for data augmentation
# in the context of PET/CT image registration, where the PET and CT images may be misaligned.
class Misalign(mt.Transform):
    def __init__(self, keys2misalign: List[str] = ["ct"],
                 
                 max_rotation_sag_cor_ax: Tuple[float, ...] = (5, 5, 5),
                 rad_or_deg: str = "deg",
                 prob_rot: float = 0.1,

                 max_shiftXYZ: Tuple[int, ...] = (2, 2, 1),
                 prob_shift: float = 0.1):
        """Misalign is a transform that randomly shifts the selected channel(s) of the input data by a random number of pixels in the range [-max_shift, max_shift].
        Args:
            keys (List[str]): List of keys to apply the transform.
            max_shift (int): Maximum shifts in pixels. Defaults to (0, 2, 2).
            prob (float): Probability of applying the transform. Defaults to 0.1.
        """
        self.keys = keys2misalign

        if rad_or_deg == "rad":
            if any(rot > np.pi for rot in max_rotation_sag_cor_ax):
                raise ValueError("The rotation is probably in deg or bigger than 180°")
            self.max_rotation_sag_cor_ax = max_rotation_sag_cor_ax
        elif rad_or_deg == "deg":
            self.max_rotation_sag_cor_ax = [rot/360*(2*np.pi) for rot in max_rotation_sag_cor_ax]
        else:
            raise RuntimeError('Please define the rad_or_deg: "rad"/"deg"')
        self.prob_rot = prob_rot

        self.max_shift = max_shiftXYZ
        self.prob_shift = prob_shift

    def __call__(self, data):
        shape = data['ct'].shape[1:]
        # print(shape)
        # print(self.max_shift)

        do_rot = random.random() < self.prob_rot
        do_shift = random.random() < self.prob_shift
        
        if do_rot:
            print("Rotating")
            rot_angles = [random.uniform(-a, a) for a in self.max_rotation_sag_cor_ax]
            print("Rotating by", rot_angles)
            for ch in self.keys:
                data[ch] = rotate(data[ch],
                                  angle=rot_angles,
                                  output_shape=shape,
                                  mode='bilinear',
                                  padding_mode='zeros',
                                  align_corners=False,
                                  dtype=None,
                                  lazy=False,
                                  transform_info=None)
                print(data[ch].shape)

        if do_shift:
            center_location_in_pixels = []
            for d in range(3):
                center_location_in_pixels.append(shape[d] / 2 + random.randint(self.max_shift[d], self.max_shift[d]))
            # print("Old image center", [i / 2 for i in shape])
            # print("New image center", center_location_in_pixels)

            for ch in self.keys:
                # print(ch)
                # print(data[ch].shape)
                # print("Output shape from function crop_tensor:", crop_tensor(data[ch], [math.floor(i) for i in center_location_in_pixels],
                                #   shape, pad_mode='constant', pad_kwargs={'value': 0}).shape)
                data[ch] = crop_tensor(data[ch], [math.floor(i) for i in center_location_in_pixels],
                                       shape, pad_mode='constant', pad_kwargs={'value': 0})
        return data
# ------------------------------------------------------------------------------------------------


def get_transforms(
    stage: str,
    target_shape: Tuple,
    resample: bool = False,
    load: bool = True,
    spacing: Tuple[float, float, float] = (2.0364201068878174, 2.0364201068878174, 3.0),
    ct_percentiles: Tuple[float, float] = (-832.062744140625, 1127.758544921875),
    pet_percentiles: Tuple[float, float] = (1.0433332920074463, 51.211158752441406),
    pet_norm: torch.Tensor = torch.Tensor((7.063827929027176, 7.960414805306728)),
    ct_norm: torch.Tensor = torch.Tensor((107.73438968591431, 286.34403119451997)),
    do_misalign: bool = False,
    max_rotation_sag_cor_ax=(5, 5, 5), rad_or_deg="deg", prob_rot=0.1,
    max_shiftXYZ=(2, 2, 1), prob_shift=0.1,
    do_random_other_transforms: bool = True,
    transforms_name: Optional[str] = None,
) -> mt.Compose:
    """The get_transforms function generates a series of transforms based on the stage and target shape.
    It performs loading, resampling, shifting intensity ranges, and applying various augmentation techniques such
    as affine transforms, noise, blur, intensity adjustments, flips, and more. It also performs normalization and
    concatenates the transformed data into a single tensor.
    Args:
        stage (str): The stage of the transformation (e.g., "train", "val", "val2", "val_sampled").
        target_shape (Tuple[int, int, int]): The target shape of the transformation.
        resample (bool): Flag indicating whether resampling should be performed.
        load (bool): Flag indicating whether loading should be performed.
        spacing (Tuple[float, float, float]): The spacing for resampling.
        ct_percentiles (Tuple[float, float]): The percentiles for CT normalization.
        pet_percentiles (Tuple[float, float]): The percentiles for PET normalization.
        pet_norm (torch.Tensor): The normalization factor for PET.
        ct_norm (torch.Tensor): The normalization factor for CT.
    Returns:
        mt.Compose: A composed set of transforms.

    """
    input_keys = ["ct", "pet"]
    if stage in ["train", "val", "val2", "val_sampled"]:
        keys = ["ct", "pet", "label"]
        out = ["image", "label"]
        mode = ("bilinear", "bilinear", "nearest")
    else:
        keys = ["ct", "pet"]
        out = ["image"]
        mode = ("bilinear", "bilinear")

    # Define a list to store all the transforms
    all_transforms = []

    # loading
    if load:
        load_transforms = [
            mt.LoadImaged(keys=keys),
            mt.EnsureChannelFirstd(keys=keys),
            mt.EnsureTyped(keys=keys),
        ]
        all_transforms.extend(load_transforms)

    if resample:
        # resampling
        sample_transforms = [
            mt.Orientationd(keys=keys, axcodes="LAS"),  # RAS
            mt.Spacingd(
                keys=keys,
                pixdim=spacing,
                mode=mode,
            ),
        ]
        all_transforms.extend(sample_transforms)

    # shift ct and pet in value range 0 to 1 for transforms to work properly eg zero padding
    shift_transforms = [
        mt.ScaleIntensityRanged(
            keys=["ct"], a_min=ct_percentiles[0], a_max=ct_percentiles[1], b_min=0, b_max=1, clip=True
        ),
        mt.ScaleIntensityRanged(
            keys=["pet"], a_min=pet_percentiles[0], a_max=pet_percentiles[1], b_min=0, b_max=1, clip=True
        ),
    ]
    all_transforms.extend(shift_transforms)

    if stage == "train":

        if do_misalign:
            misalign_transforms = [
                # Misalign first the modalities:
                Misalign(keys2misalign=["ct"],
                         max_rotation_sag_cor_ax=max_rotation_sag_cor_ax, rad_or_deg=rad_or_deg, prob_rot=prob_rot,
                         max_shiftXYZ=max_shiftXYZ, prob_shift=prob_shift),
            ]
            all_transforms.extend(misalign_transforms)
            
        other_transforms = [
            # pad to target shape times 1.2 times sqrt(2) (because of affine transforms) and sample 50% a class
            # foreground part, after that apply affine transform and crop
            mt.SpatialPadd(keys=keys, spatial_size=tuple(int(math.sqrt(2) * x * 1.2) for x in target_shape)),
            CustomSampleCropd(
                keys=keys, label_key="label", roi_size=tuple(int(math.sqrt(2) * x * 1.2) for x in target_shape)
            ),
            mt.RandAffined(
                keys=keys,
                mode=mode,
                prob=0.2,
                spatial_size=target_shape,
                translate_range=(20, 20, 20),
                rotate_range=(0.52, 0.52, 0.52),
                scale_range=((-0.3, 0.4), (-0.3, 0.4), (-0.3, 0.4)),
                padding_mode="zeros",
            ),
            # noise, blur, intensity and flips
            mt.RandGaussianNoised(keys=input_keys, std=0.1, prob=0.15),
            mt.RandGaussianSmoothd(
                keys=input_keys,
                sigma_x=(0.5, 1),
                sigma_y=(0.5, 1),
                sigma_z=(0.5, 1),
                prob=0.2,
            ),
            mt.RandScaleIntensityd(keys=input_keys, factors=0.25, prob=0.15),
            mt.RandSimulateLowResolutiond(keys=input_keys, zoom_range=(0.5, 1), prob=0.25),
            mt.RandAdjustContrastd(keys=input_keys, gamma=(0.7, 1.5), invert_image=True, retain_stats=True, prob=0.1),
            mt.RandAdjustContrastd(keys=input_keys, gamma=(0.7, 1.5), invert_image=False, retain_stats=True, prob=0.3),
            mt.RandFlipd(keys=keys, spatial_axis=[0], prob=0.5),
            mt.RandFlipd(keys=keys, spatial_axis=[1], prob=0.5),
            mt.RandFlipd(keys=keys, spatial_axis=[2], prob=0.5),
        ]
        if transforms_name is None:
            if do_random_other_transforms:
                all_transforms.extend(other_transforms)
            # else do not apply other transforms
        elif isinstance(transforms_name, str):
            # overwrite default transforms with custom ones
            if transforms_name == "custom_v1":
                other_transforms = [
                    # pad to target shape times 1.2 times sqrt(2) (because of affine transforms) and sample 50% a class
                    # foreground part, after that apply affine transform and crop
                    mt.SpatialPadd(keys=keys, spatial_size=tuple(int(math.sqrt(2) * x * 1.2) for x in target_shape)),
                    CustomSampleCropd(
                        keys=keys, label_key="label", roi_size=tuple(int(math.sqrt(2) * x * 1.2) for x in target_shape)
                    ),
                    mt.RandAffined(
                        keys=keys,
                        mode=mode,
                        prob=0.2,
                        spatial_size=target_shape,
                        translate_range=(10, 10, 10),
                        rotate_range=(0.3, 0.3, 0.3),
                        scale_range=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2)),
                        padding_mode="zeros",
                    ), # adjust hyperparameters for less aggressive augmentations
                    # noise, blur, intensity and flips
                    mt.RandGaussianNoised(keys=input_keys, std=0.05, prob=0.2), # adjust std for less noise
                    # (remove random gaussian noise to avoid blurring small lesions)
                    mt.RandScaleIntensityd(keys=input_keys, factors=0.25, prob=0.15),
                    mt.RandSimulateLowResolutiond(keys=input_keys, zoom_range=(0.5, 1), prob=0.25),
                    # (remove random adjust contrast with invert_image=True to avoid overexposure)
                    mt.RandAdjustContrastd(keys=input_keys, gamma=(0.7, 1.5), invert_image=False, retain_stats=True, prob=0.3),
                    mt.RandFlipd(keys=keys, spatial_axis=[0], prob=0.5),
                    mt.RandFlipd(keys=keys, spatial_axis=[1], prob=0.5),
                    mt.RandFlipd(keys=keys, spatial_axis=[2], prob=0.5),
                ]
                all_transforms.extend(other_transforms)
            elif transforms_name == "default":
                all_transforms.extend(other_transforms)
            else:
                raise ValueError("Unknown transforms_name")       

    elif stage == "val_sampled":
        other_transforms = [
            mt.SpatialPadd(keys=keys, spatial_size=target_shape),
            CustomSampleCropd(keys=keys, label_key="label", roi_size=target_shape),
        ]
        all_transforms.extend(other_transforms)

    elif stage == "val":
        other_transforms = [
            mt.SpatialPadd(keys=keys, spatial_size=target_shape),
        ]
        all_transforms.extend(other_transforms)

    normalization_transforms = [
        # shift ct and pet range back to original and clip (again)
        mt.ScaleIntensityRanged(
            keys=["ct"], a_min=0, a_max=1, b_min=ct_percentiles[0], b_max=ct_percentiles[1], clip=True
        ),
        mt.ScaleIntensityRanged(
            keys=["pet"], a_min=0, a_max=1, b_min=pet_percentiles[0], b_max=pet_percentiles[1], clip=True
        ),
        mt.NormalizeIntensityd(keys=["ct"], subtrahend=ct_norm[0], divisor=ct_norm[1]),
        mt.NormalizeIntensityd(keys=["pet"], subtrahend=pet_norm[0], divisor=pet_norm[1]),
    ]
    all_transforms.extend(normalization_transforms)

    final_transforms = [
        mt.ConcatItemsd(keys=input_keys, name="image", dim=0),
        mt.EnsureTyped(keys=keys),
        mt.ToTensord(keys=keys),
        Unpackd(keys=out),  # unpack dict, pickable version
    ]
    all_transforms.extend(final_transforms)
    return mt.Compose(all_transforms)


if __name__ == "__main__":
    # Test the transforms
    from autopet3.fixed.utils import plot_ct_pet_label

    root_dir = "/path/to/miccai2024_autopet3_datacentric"

    split = read_split(os.path.join(root_dir, "test/data/splits_final.json"), 0)
    test_sample = get_file_dict_nn(os.path.join(root_dir,"test/data/"), split["train"], suffix=".nii.gz")[0]
    print("test sample:", test_sample)

    transform = get_transforms(stage="train", resample=True, target_shape=(112, 160, 128) ,do_misalign=False, do_random_other_transforms=False)
    transform_misalign = get_transforms(stage="train", resample=True, target_shape=(112, 160, 128), do_misalign=True, do_random_other_transforms=False)

    res = transform(test_sample)
    res_misalign = transform_misalign(test_sample)
    plot_ct_pet_label(ct=res[0][0], pet=res[0][1], label=res[1])
    plot_ct_pet_label(ct=res_misalign[0][0], pet=res_misalign[0][1], label=res_misalign[1])

    transform = get_transforms(stage="val_sampled", resample=False, target_shape=(112, 160, 128))
    res = transform(test_sample)
    plot_ct_pet_label(ct=res[0][0], pet=res[0][1], label=res[1])
