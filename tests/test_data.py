
import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from dr_detection.core.data import (
    DRDataset,
    ben_graham,
    circular_crop,
    crop_black_borders,
    estimate_retina_radius,
    preprocess_fundus,
    scale_to_radius,
)


class TestDRDataset:
    """Test suite for DRDataset class."""

    @pytest.fixture
    def mock_dataset(self, tmp_path):
        """Create mock datasets"""
        data_dir = tmp_path / 'images'
        data_dir.mkdir()

        n_samples = 10
        image_ids = []
        labels = []

        for i in range(n_samples):
            image_id = f'abc{i:04d}xyz'
            image_ids.append(image_id)
            labels.append(i % 5)

            img = Image.fromarray(
                np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            )
            img.save(data_dir / f'{image_id}.png')

        labels_file = tmp_path / 'labels.csv'

        df = pd.DataFrame({
            'image_id': image_ids,
            'diagnosis': labels
        })
        df.to_csv(labels_file, index=False)

        return str(data_dir), str(labels_file)
 
    def test_dataset_length(self, mock_dataset):
        """Test dataset returns correct length"""
        data_dir, labels_file = mock_dataset
        dataset = DRDataset(data_dir, labels_file)
        assert len(dataset) == 10

    def test_dataset_getitem(self, mock_dataset):
        """Test __getitem__ returns correct format"""
        data_dir, labels_file = mock_dataset
        dataset = DRDataset(data_dir, labels_file, image_size=(224, 224))

        image, label = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert isinstance(label, int)
        assert image.shape == (3, 224, 224)
        assert 0 <= label < 5

    def test_transforms_applied(self, mock_dataset):
        """Test that transforms are applied correctly."""
        data_dir, labels_file = mock_dataset

        train_transform = DRDataset.get_train_transforms((512, 512))
        val_transform = DRDataset.get_val_transforms((512, 512))

        dataset_train = DRDataset(data_dir, labels_file, transform=train_transform)
        dataset_val = DRDataset(data_dir, labels_file, transform=val_transform)

        image_train, _ = dataset_train[0]
        image_val, _ = dataset_val[0]

        assert image_train.shape == (3, 512, 512)
        assert image_val.shape == (3, 512, 512)

    def test_class_weights_calculation(self, mock_dataset):
        """Test class weights are calculated correctly."""
        data_dir, labels_file = mock_dataset
        dataset = DRDataset(data_dir, labels_file)

        class_weights = dataset.get_class_weights()

        assert len(class_weights) == 5
        assert all(weight > 0 for weight in class_weights)

    def test_preprocess_flag_disables_pipeline(self, mock_dataset):
        """preprocess=False must leave the array untouched (baked images)."""
        data_dir, labels_file = mock_dataset
        dataset = DRDataset(data_dir, labels_file, preprocess=False)

        image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        assert np.array_equal(dataset.preprocess_image(image), image)

    def test_preprocess_flag_enables_pipeline(self, mock_dataset):
        """preprocess=True must actually transform the array."""
        data_dir, labels_file = mock_dataset
        dataset = DRDataset(data_dir, labels_file, preprocess=True)

        image = _synthetic_fundus(600, radius=200)
        processed = dataset.preprocess_image(image)

        assert not np.array_equal(processed, image)
        assert estimate_retina_radius(processed) == pytest.approx(
            dataset.target_radius, abs=2
        )


def _synthetic_fundus(size: int, radius: int, value: int = 140) -> np.ndarray:
    """A lit disc on a black background, standing in for a fundus photo."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(image, (size // 2, size // 2), radius,
               (value, value // 2, value // 3), thickness=-1)
    return image


class TestFundusPreprocessing:
    """Test suite for the fundus preprocessing pipeline."""

    def test_crop_black_borders(self):
        """Letterboxing is removed down to the lit bounding box."""
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        image[50:250, 50:250] = 128

        assert crop_black_borders(image).shape[:2] == (200, 200)

    def test_crop_black_borders_no_op(self):
        """An image with no borders comes back unchanged."""
        image = np.full((300, 300, 3), 128, dtype=np.uint8)
        assert crop_black_borders(image).shape == image.shape

    def test_crop_black_borders_all_black(self):
        """An entirely black image is returned rather than cropped to nothing."""
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        assert crop_black_borders(image).shape == image.shape

    def test_estimate_retina_radius(self):
        """The centre-row measurement recovers the disc radius."""
        assert estimate_retina_radius(_synthetic_fundus(600, 150)) == pytest.approx(
            150, abs=2
        )

    def test_estimate_retina_radius_fallback(self):
        """A dark centre row falls back to half the largest dimension."""
        assert estimate_retina_radius(np.zeros((400, 200, 3), np.uint8)) == 200.0

    def test_scale_to_radius_is_zoom_invariant(self):
        """This is the point of the step: equal lesion scale across images.

        Three source images at very different zoom levels must all come out
        with the same retina radius.
        """
        for source_size, source_radius in ((600, 120), (900, 400), (1200, 300)):
            scaled = scale_to_radius(
                _synthetic_fundus(source_size, source_radius), target_radius=300
            )
            assert estimate_retina_radius(scaled) == pytest.approx(300, abs=3)

    def test_ben_graham_normalises_background(self):
        """Local-average subtraction re-centres flat regions on mid-grey."""
        result = ben_graham(_synthetic_fundus(600, 240))

        assert result.shape == (600, 600, 3)
        assert result.dtype == np.uint8
        assert result[5, 5] == pytest.approx(128, abs=2)

    def test_circular_crop_masks_corners(self):
        """Corners take the fill value; the centre is left alone."""
        result = circular_crop(
            np.full((200, 200, 3), 200, dtype=np.uint8), shrink=0.9, fill=128
        )

        assert list(result[0, 0]) == [128, 128, 128]
        assert list(result[100, 100]) == [200, 200, 200]

    def test_pipeline_output_is_consistent_across_zoom(self):
        """End to end, differing input zoom yields a constant output radius."""
        shapes = set()

        for source_size, source_radius in ((600, 120), (900, 400), (1200, 300)):
            result = preprocess_fundus(_synthetic_fundus(source_size, source_radius))
            shapes.add(result.shape)
            assert estimate_retina_radius(result) == pytest.approx(300, abs=3)

        assert len(shapes) == 1, f'inconsistent output shapes: {shapes}'

    def test_pipeline_without_ben_graham_uses_black_fill(self):
        """With Ben Graham off the masked ring is black, not mid-grey."""
        result = preprocess_fundus(
            _synthetic_fundus(600, 240), apply_ben_graham=False
        )
        assert list(result[0, 0]) == [0, 0, 0]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
