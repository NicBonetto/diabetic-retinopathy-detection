import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import pandas as pd 
from PIL import Image

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.core.data import DRDataset

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
            'id_code': image_ids,
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

    def test_black_border_removal(self, tmp_path):
        """Test that black borders are correctly removed."""
        data_dir = tmp_path / 'images'
        data_dir.mkdir()

        img_array = np.zeros((300, 300, 3), dtype=np.uint8)
        img_array[50:250, 50:250] = 128

        img = Image.fromarray(img_array)
        img.save(data_dir / 'test_image.png')

        labels_file = tmp_path / 'labels.csv'
        df = pd.DataFrame({'id_code': ['test_image'], 'diagnosis': [0]})
        df.to_csv(labels_file, index=False)

        dataset = DRDataset(str(data_dir), str(labels_file))

        image_pil = Image.open(data_dir / 'test_image.png').convert('RGB')
        image_np = np.array(image_pil)
        processed = dataset.preprocess_image(image_np)

        assert processed.shape[0] < 300
        assert processed.shape[1] < 300
        assert 190 <= processed.shape[0] <= 210
        assert 190 <= processed.shape[1] <= 210

    def test_preprocessing_no_black_borders(self):
        """Test preprocessing when image has no black borders."""
        dataset = DRDataset.__new__(DRDataset)

        image = np.ones((300, 300, 3), dtype=np.uint8) * 128
        processed = dataset.preprocess_image(image)

        assert processed.shape == image.shape

    def test_preprocessing_all_black_image(self):
        """Test preprocessing when image is entirely black."""
        dataset = DRDataset.__new__(DRDataset)

        image = np.zeros((300, 300, 3), dtype=np.uint8)
        processed = dataset.preprocess_image(image)

        assert processed is not None

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
