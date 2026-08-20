from pathlib import Path
from typing import Callable, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

# Defaults for the fundus preprocessing 
DEFAULT_TARGET_RADIUS = 300
DEFAULT_BEN_GRAHAM_SIGMA = 10
DEFAULT_CIRCLE_SHRINK = 0.9
DEFAULT_BLACK_THRESHOLD = 7

# Guard against a degenerate radius estimate blowing up memory on resize.
MAX_RESCALE_FACTOR = 10.0


def crop_black_borders(
    image: np.ndarray,
    threshold: int = DEFAULT_BLACK_THRESHOLD
) -> np.ndarray:
    """
    Crop the black letterboxing around a fundus photo.

    Returns the image unchanged if it is entirely below threshold.
    """
    gray = np.mean(image, axis=2)
    mask = gray > threshold

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if rows.any() and cols.any():
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        image = image[y_min:y_max + 1, x_min:x_max + 1]

    return image


def estimate_retina_radius(
    image: np.ndarray,
    threshold: int = DEFAULT_BLACK_THRESHOLD
) -> float:
    """
    Estimate the radius of the retinal disc in pixels.

    Measures the lit width of the centre row, which is the disc diameter for
    a roughly centred fundus photo. Falls back to half the largest dimension
    when the centre row is dark (badly cropped or near-black images).
    """
    gray = np.mean(image, axis=2)
    centre_row = gray[gray.shape[0] // 2, :] > threshold
    radius = float(centre_row.sum()) / 2.0

    if radius < 1.0:
        radius = float(max(image.shape[:2])) / 2.0

    return radius


def scale_to_radius(
    image: np.ndarray,
    target_radius: int = DEFAULT_TARGET_RADIUS,
    threshold: int = DEFAULT_BLACK_THRESHOLD
) -> np.ndarray:
    """Rescale so the retinal disc has a constant radius across the dataset."""
    radius = estimate_retina_radius(image, threshold)
    scale = min(target_radius / radius, MAX_RESCALE_FACTOR)

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=interpolation)


def ben_graham(
    image: np.ndarray,
    sigma: int = DEFAULT_BEN_GRAHAM_SIGMA
) -> np.ndarray:
    """
    Subtract the local average colour and re-centre on mid-grey.

    Normalizes the illumination and camera-colour variation that dominates
    raw fundus photos, which makes haemorrhages and exudates far more
    visible.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 4, blurred, -4, 128)


def circular_crop(
    image: np.ndarray,
    shrink: float = DEFAULT_CIRCLE_SHRINK,
    fill: int = 0
) -> np.ndarray:
    """
    Mask everything outside the retinal disc.

    Removes the corner artifacts a rectangular crop leaves behind. `fill`
    should be 128 after ben_graham so the masked region matches that
    transform's mid-grey background instead of introducing a hard edge.
    """
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    centre = (width // 2, height // 2)
    radius = int(min(height, width) / 2 * shrink)
    cv2.circle(mask, centre, radius, 1, thickness=-1)

    mask = mask[..., None]
    return (image * mask + fill * (1 - mask)).astype(image.dtype)


def preprocess_fundus(
    image: np.ndarray,
    target_radius: int = DEFAULT_TARGET_RADIUS,
    apply_ben_graham: bool = True,
    ben_graham_sigma: int = DEFAULT_BEN_GRAHAM_SIGMA,
    circle_shrink: float = DEFAULT_CIRCLE_SHRINK,
    threshold: int = DEFAULT_BLACK_THRESHOLD
) -> np.ndarray:
    """
    Full fundus preprocessing pipeline: crop, scale, normalize, mask.

    This is expensive to run per-epoch on full-resolution images. Prefer
    baking it in once with `dr-preprocess --preprocess-images`, then set
    `data.preprocess.enabled: false` in the training config.
    """
    image = crop_black_borders(image, threshold)
    image = scale_to_radius(image, target_radius, threshold)

    if apply_ben_graham:
        image = ben_graham(image, ben_graham_sigma)
        fill = 128
    else:
        fill = 0

    return circular_crop(image, circle_shrink, fill)


class DRDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        labels_file: str,
        transform: Optional[Callable]=None,
        image_size: Tuple[int, int]=(256, 256),
        preprocess: bool=True,
        target_radius: int=DEFAULT_TARGET_RADIUS,
        apply_ben_graham: bool=True,
        ben_graham_sigma: int=DEFAULT_BEN_GRAHAM_SIGMA
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size

        # Set preprocess=False when the images on disk have already been
        # through preprocess_fundus (see dr-preprocess --preprocess-images).
        self.preprocess = preprocess
        self.target_radius = target_radius
        self.apply_ben_graham = apply_ben_graham
        self.ben_graham_sigma = ben_graham_sigma

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

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the fundus preprocessing pipeline to one image.

        A no-op when preprocess=False, which is the right setting if the
        pipeline was already baked into the files on disk.
        Assumes all images are colored (ndim = 3).
        """
        if not self.preprocess:
            return image

        return preprocess_fundus(
            image,
            target_radius=self.target_radius,
            apply_ben_graham=self.apply_ben_graham,
            ben_graham_sigma=self.ben_graham_sigma
        )

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
    def get_train_transforms(
        image_size: Tuple[int, int]=(512, 512),
        ben_graham: bool=True
    ) -> A.Compose:
        """
        Augmentation pipeline for training data.

        When ben_graham is True, HueSaturationValue is dropped: that
        preprocessing already normalises away camera-colour variation, so
        jittering hue and saturation just re-injects the noise it removed.
        """
        colour_jitter = [] if ben_graham else [
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.5
            )
        ]

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
            *colour_jitter,
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
