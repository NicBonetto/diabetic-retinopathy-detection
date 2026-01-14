import os
from pathlib import Path
from typing import Optional, Callable, Tuple

import numpy as np
import pandas as pd 
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A 
from albumentations.pytorch import ToTensorV2

class DRDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        labels_file: str,
        transform: Optional[Callable]=None,
        image_size: Tuple[int, int]=(256, 256)
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size

        self.labels_df = pd.read_csv(labels_file)
        self.image_ids = self.labels_df['image_id'].values
        self.labels = self.labels_df['diagnosis'].values

        print(f'Loaded {len(self.image_ids)} images from {labels_file}')

        self.transform = transform if transform is not None else self.get_default_tranform()

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_id = self.image_ids[idx]

        image_path = self.data_dir / f'{image_id}.png'
        image = np.array(Image.open(image_path).convert('RGB'))

        image = self.preprocess_image(image)
        label = int(self.labels[idx])

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
 
        return image, label

    def preprocess_image(self, image: np.ndarray, threshold: int=7) -> np.ndarray:
        """
        Preprocess retinal fundus image.
        Crop black borders.
        Assumes all images are colored (ndim = 3).
        """
        gray = np.mean(image, axis=2)
        mask = gray > threshold

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if rows.any() and cols.any():
           y_min, y_max = np.where(rows)[0][[0, -1]]
           x_min, x_max = np.where(cols)[0][[0, -1]]
           image = image[y_min:y_max+1, x_min:x_max+1]

        return image

    def get_default_tranform(self) -> A.Compose:
        """
        Default transform pipeline for training.
        Uses standard normalization ImageNet values.
        """
        return A.Compose([
            A.Resize(height=self.image_size[0], width=self.image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

    @staticmethod
    def get_train_transforms(image_size: Tuple[int, int]=(512, 512)) -> A.Compose:
        """Augmentation pipeline for training data."""
        return A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.5
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

    @staticmethod
    def get_val_transforms(image_size: Tuple[int, int]=(512, 512)) -> A.Compose:
        """Transform pipeline for test data."""
        return A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

    def get_class_weights(self) -> np.ndarray:
        """
        Calculate class weights for handling imbalanced data.
        Uses inverse frequency weighting.
        """
        from sklearn.utils.class_weight import compute_class_weight

        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.labels),
            y=self.labels
        )
        return class_weights
