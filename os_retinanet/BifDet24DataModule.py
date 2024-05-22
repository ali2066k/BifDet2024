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
        self.annotation_path = train_parent_path + bbox_path
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
        self.max_cardinality = 10
        self.compute_dtype = compute_dtype

    def prepare_data_monai(self) -> None:
        self.train_set = load_decathlon_datalist(
            self.annotation_path,
            is_segmentation=True,
            data_list_key="training",
            base_dir=self.train_data_path,
        )
        print(f"Number of Patients in training set {len(self.train_set)}")
        for i in range(len(self.train_set)):
            self.train_set[i]['awlabel'] = self.train_set[i]['awlabel'].replace("imagesTr", "labelsTr")
            self.train_set[i]['lung'] = self.train_set[i]['lung'].replace("imagesTr", "lungsTr")

    def get_preprocess_monai_transforms(self):
        train_transforms = Compose(
            [
                LoadImaged(keys=['image', 'lung']),
                EnsureChannelFirstd(keys=['image', 'lung']),
                EnsureTyped(keys=['image', 'lung', 'boxes'], dtype=self.compute_dtype),
                EnsureTyped(keys=['label'], dtype=torch.long),
                ScaleIntensityRanged(keys='image',
                                     a_min=self.params['PIXEL_VALUE_MIN'],
                                     a_max=self.params['PIXEL_VALUE_MAX'],
                                     b_min=self.params['PIXEL_NORM_MIN'],
                                     b_max=self.params['PIXEL_NORM_MAX'], clip=True),
                ConvertBoxToStandardModed(box_keys=['boxes'], mode="cccwhd"),
                AffineBoxToImageCoordinated(
                    box_keys=['boxes'],
                    box_ref_image_keys='image',
                    image_meta_key_postfix="meta_dict",
                    affine_lps_to_ras=True,
                ),
                BoxToMaskd(
                    box_keys=['boxes'],
                    label_keys=['label'],
                    box_mask_keys=["box_mask"],
                    box_ref_image_keys='image',
                    min_fg_label=0,
                    ellipse_mask=False,
                ),
                CropForegroundd(keys=['image', 'lung', 'box_mask'], source_key='lung'),
                RandCropBoxByPosNegLabeld(
                    image_keys=['image'],
                    box_keys='boxes',
                    label_keys='label',
                    spatial_size=self.params["PATCH_SIZE"],
                    whole_box=True,
                    num_samples=1,
                    pos=1,
                    neg=0,
                    allow_smaller=True
                ),
                ResizeWithPadOrCropd(keys=['image', 'lung', 'box_mask'], spatial_size=256),
                MaskToBoxd(
                    box_mask_keys="box_mask", box_keys="boxes",
                    label_keys="label", min_fg_label=0
                ),
                DeleteItemsd(keys=["box_mask"]),
                ClipBoxToImaged(
                    box_keys='boxes',
                    label_keys=['label'],
                    box_ref_image_keys='image',
                    remove_empty=True,
                ),
                EnsureTyped(keys=['image', 'boxes'], dtype=self.compute_dtype),
                EnsureTyped(keys=['label'], dtype=torch.long),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=['image', 'lung']),
                EnsureChannelFirstd(keys=['image', 'lung']),
                EnsureTyped(keys=['image', 'lung', 'boxes'], dtype=torch.float32),
                EnsureTyped(keys=['label'], dtype=torch.long),
                ScaleIntensityRanged(keys='image',
                                     a_min=self.params['PIXEL_VALUE_MIN'],
                                     a_max=self.params['PIXEL_VALUE_MAX'],
                                     b_min=self.params['PIXEL_NORM_MIN'],
                                     b_max=self.params['PIXEL_NORM_MAX'], clip=True),
                ConvertBoxToStandardModed(box_keys=['boxes'], mode="cccwhd"),
                AffineBoxToImageCoordinated(
                    box_keys=['boxes'],
                    box_ref_image_keys='image',
                    image_meta_key_postfix="meta_dict",
                    affine_lps_to_ras=True,
                ),
                BoxToMaskd(
                    box_keys=['boxes'],
                    label_keys=['label'],
                    box_mask_keys=["box_mask"],
                    box_ref_image_keys='image',
                    min_fg_label=0,
                    ellipse_mask=False,
                ),
                CropForegroundd(keys=['image', 'lung', 'box_mask'], source_key='lung'),
                RandCropBoxByPosNegLabeld(
                    image_keys=['image'],
                    box_keys='boxes',
                    label_keys='label',
                    spatial_size=self.params["VAL_PATCH_SIZE"],
                    whole_box=True,
                    num_samples=1,
                    pos=1,
                    neg=0,
                ),
                ResizeWithPadOrCropd(keys=['image', 'lung', 'box_mask'], spatial_size=256),
                MaskToBoxd(
                    box_mask_keys="box_mask", box_keys="boxes",
                    label_keys="label", min_fg_label=0
                ),
                DeleteItemsd(keys=["box_mask"]),
                ClipBoxToImaged(
                    box_keys='boxes',
                    label_keys=['label'],
                    box_ref_image_keys='image',
                    remove_empty=True,
                ),
                EnsureTyped(keys=['image', 'boxes'], dtype=self.compute_dtype),
                EnsureTyped(keys=['label'], dtype=torch.long),
            ]
        )
        return train_transforms, val_transforms

    def setup(self, stage=None, dataset_library='monai') -> None:
        self.train_preprocess, self.val_preprocess = self.get_preprocess_monai_transforms()
        self.train_transform = self.train_preprocess if not self.augment else None  # TODO:augmentation should b e implemented here
        self.val_transform = self.val_preprocess if not self.augment else None  # TODO:augmentation should b e implemented here

        if dataset_library == 'monai':
            if self.params["CACHE_DS"]:
                self.train_set = CacheDataset(data=self.train_set[:-2], transform=self.train_preprocess)
                self.val_set = CacheDataset(data=self.train_set[-2:], transform=self.val_transform)
            else:
                self.train_set = Dataset(data=self.train_set[:-2], transform=self.train_preprocess)
                self.val_set = Dataset(data=self.train_set[-2:], transform=self.val_transform)

            # self.test_set = Dataset(data=self.test_files, transform=self.val_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.train_set,
                          batch_size=self.batch_size,
                          num_workers=self.params['NUM_WORKERS'],
                          shuffle=True,
                          pin_memory=torch.cuda.is_available(),
                          collate_fn=no_collation,
                          persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.val_set,
                          batch_size=self.params['BATCH_SIZE'],
                          num_workers=self.params['VAL_NUM_WORKERS'])

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.test_set,
                          batch_size=1,
                          num_workers=self.params['NUM_WORKERS'])
