# ruff: noqa: F401
import sys

# import pytorch_lightning as pl
import numpy as np
from glob import glob
import os
import torch
from monai.apps.detection.transforms.dictionary import ConvertBoxToStandardModed, AffineBoxToImageCoordinated, \
    ClipBoxToImaged, RandCropBoxByPosNegLabeld, SpatialCropBox, MaskToBoxd, BoxToMaskd
from monai.data.utils import no_collation
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    DeleteItemsd,
    DivisiblePadd,
    CropForegroundd,
    ResizeWithPadOrCropd,
    RandSpatialCropd, ScaleIntensityRanged,
)
from monai.data import DataLoader, Dataset,CacheDataset, load_decathlon_datalist


class BifDet2024DataModule():
    def __init__(self, train_parent_path, batch_size, bbox_path=None, test_batch_size=1,
                 train_val_test_ratio=[0.8, 0.1, 0.1], augment=False, params=None, compute_dtype=None, **kwargs):
        super().__init__()
        self.train_data_path = os.path.join(train_parent_path, 'imagesTr/')
        self.train_label_path = os.path.join(train_parent_path, 'lungsTr/')  # Corrected variable name
        self.val_label_path = os.path.join(train_parent_path, 'labelsTr/')  # Corrected variable name
        self.annotation_path = train_parent_path / bbox_path
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.train_val_test_ratio = train_val_test_ratio
        self.augment = augment
        self.train_transform = None
        self.val_transform = None
        self.train_set = None
        self.test_set = None
        self.val_set = None
        self.params = params
        self.max_cardinality = 22
        self.compute_dtype = compute_dtype

    def prepare_data_monai(self) -> None:
        self.raw_train_set = load_decathlon_datalist(
            self.annotation_path,
            is_segmentation=False,
            data_list_key="training",
            base_dir=self.train_data_path,
        )
        self.max_cardinality = len(self.raw_train_set)
        print(f"Number of Patients in training set {self.max_cardinality}")
        for i in range(self.max_cardinality):
            self.raw_train_set[i]['awlabel'] = self.raw_train_set[i]['awlabel'].replace("imagesTr", "labelsTr")
            self.raw_train_set[i]['lung'] = self.raw_train_set[i]['lung'].replace("imagesTr", "lungsTr")

    def get_preprocess_monai_transforms(self):
        train_transforms = Compose(
            [
                LoadImaged(keys=['image', 'lung']),
                EnsureChannelFirstd(keys=['image', 'lung'],
                                    strict_check=True),
                EnsureTyped(keys=['image', 'lung', 'boxes'], dtype=self.compute_dtype),
                EnsureTyped(keys=['label'], dtype=torch.long),
                ScaleIntensityRanged(keys='image',
                                     a_min=self.params['PIXEL_VALUE_MIN'],
                                     a_max=self.params['PIXEL_VALUE_MAX'],
                                     b_min=self.params['PIXEL_NORM_MIN'],
                                     b_max=self.params['PIXEL_NORM_MAX'], clip=True),
                ConvertBoxToStandardModed(box_keys=['boxes'], mode="cccwhd"),
                CropForegroundd(keys=['image', 'lung'], source_key='lung', allow_smaller=False),
                DivisiblePadd(keys=['image'], k=self.params["PATCH_SIZE"]),
                AffineBoxToImageCoordinated(
                    box_keys=['boxes'],
                    box_ref_image_keys='image',
                    image_meta_key_postfix="meta_dict",
                    affine_lps_to_ras=True,
                ),
                RandCropBoxByPosNegLabeld(
                    image_keys=['image'],
                    box_keys='boxes',
                    label_keys='label',
                    spatial_size=self.params["PATCH_SIZE"],
                    whole_box=True,
                    num_samples=1,
                    pos=1,
                    neg=0,
                    allow_smaller=False
                ),
                EnsureTyped(keys=['image', 'lung', 'boxes'], dtype=self.compute_dtype),
                EnsureTyped(keys=['label'], dtype=torch.long),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=['image', 'lung']),
                EnsureChannelFirstd(keys=['image', 'lung'],
                                    strict_check=True),
                EnsureTyped(keys=['image', 'lung', 'boxes'], dtype=self.compute_dtype),
                EnsureTyped(keys=['label'], dtype=torch.long),
                ScaleIntensityRanged(keys='image',
                                     a_min=self.params['PIXEL_VALUE_MIN'],
                                     a_max=self.params['PIXEL_VALUE_MAX'],
                                     b_min=self.params['PIXEL_NORM_MIN'],
                                     b_max=self.params['PIXEL_NORM_MAX'], clip=True),
                ConvertBoxToStandardModed(box_keys=['boxes'], mode="cccwhd"),
                CropForegroundd(keys=['image', 'lung'], source_key='lung', allow_smaller=False),
                # DivisiblePadd(keys=['image'], k=self.params["PATCH_SIZE"]),
                ResizeWithPadOrCropd(keys=['image'], spatial_size=self.params['VAL_PATCH_SIZE']),
                # RandCropBoxByPosNegLabeld(
                #     image_keys=['image'],
                #     box_keys='boxes',
                #     label_keys='label',
                #     spatial_size=self.params["PATCH_SIZE"],
                #     whole_box=True,
                #     num_samples=1,
                #     pos=1,
                #     neg=0,
                #     allow_smaller=False
                # ),
                AffineBoxToImageCoordinated(
                    box_keys=['boxes'],
                    box_ref_image_keys='image',
                    image_meta_key_postfix="meta_dict",
                    affine_lps_to_ras=True,
                ),
                EnsureTyped(keys=['image', 'lung', 'boxes'], dtype=self.compute_dtype),
                EnsureTyped(keys=['label'], dtype=torch.long),
            ]
        )
        return train_transforms, val_transforms

    def setup(self, stage=None, dataset_library='monai') -> None:
        self.train_preprocess, self.val_preprocess = self.get_preprocess_monai_transforms()
        self.train_transform = self.train_preprocess if not self.augment else None  # TODO:augmentation should b e implemented here
        self.val_transform = self.val_preprocess if not self.augment else None  # TODO:augmentation should b e implemented here

        rng = np.random.default_rng(seed=0)
        perm = rng.permutation(self.max_cardinality)

        if dataset_library == 'monai':
            if self.params["CACHE_DS"]:
                self.train_set = CacheDataset(data=list(np.array(self.raw_train_set)[perm][:18]), transform=self.train_transform)
                # self.val_set = CacheDataset(data=self.train_set[-2:], transform=self.val_transform)
            else:
                self.train_set = Dataset(data=list(np.array(self.raw_train_set)[perm][:18]), transform=self.train_transform)
                # self.val_set = Dataset(data=self.train_set[-2:], transform=self.val_transform)
            self.val_set = Dataset(data=list(np.array(self.raw_train_set)[perm][18:]), transform=self.val_transform)
            # self.test_set = Dataset(data=self.test_files, transform=self.val_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_set,
                          batch_size=self.batch_size,
                          num_workers=self.params['NUM_WORKERS'],
                          shuffle=True,
                          pin_memory=torch.cuda.is_available(),
                          collate_fn=no_collation,
                          persistent_workers=False)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.val_set,
                          batch_size=self.batch_size,
                          num_workers=self.params['NUM_WORKERS'],
                          shuffle=False,
                          pin_memory=torch.cuda.is_available(),
                          collate_fn=no_collation,
                          persistent_workers=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.test_set,
                          batch_size=1,
                          num_workers=self.params['NUM_WORKERS'])
