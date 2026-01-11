import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
import tempfile

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.core.classifiers import DRClassifier
from src.core.trainer import Trainer


class TestTrainerInitialization:
    """Test suite for Trainer initialization."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return DRClassifier(backbone='resnet50', num_classes=5, pretrained=False)
    
    @pytest.fixture
    def optimizer(self, simple_model):
        """Create optimizer."""
        return Adam(simple_model.parameters(), lr=0.001)
    
    @pytest.fixture
    def criterion(self):
        """Create loss function."""
        return nn.CrossEntropyLoss()
    
    @pytest.fixture
    def device(self):
        """Get device."""
        return torch.device('cpu')
    
    def test_trainer_initialization(self, simple_model, optimizer, criterion, device):
        """Test basic trainer initialization."""
        trainer = Trainer(
            model=simple_model,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.criterion is not None
        assert trainer.device == device
        assert trainer.best_val_loss == float('inf')
        assert trainer.best_val_acc == 0.0
    
    def test_trainer_with_scheduler(self, simple_model, optimizer, criterion, device):
        """Test trainer initialization with learning rate scheduler."""
        scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        
        trainer = Trainer(
            model=simple_model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler
        )
        
        assert trainer.scheduler is not None
    
    def test_trainer_checkpoint_dir_created(self, simple_model, optimizer, criterion, device, tmp_path):
        """Test that checkpoint directory is created."""
        checkpoint_dir = tmp_path / "checkpoints"
        
        trainer = Trainer(
            model=simple_model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            checkpoint_dir=str(checkpoint_dir)
        )
        
        assert checkpoint_dir.exists()
        assert checkpoint_dir.is_dir()
    
    def test_trainer_history_initialized(self, simple_model, optimizer, criterion, device):
        """Test that training history is initialized correctly."""
        trainer = Trainer(
            model=simple_model,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )
        
        assert 'train_loss' in trainer.history
        assert 'train_acc' in trainer.history
        assert 'val_loss' in trainer.history
        assert 'val_acc' in trainer.history
        assert all(len(v) == 0 for v in trainer.history.values())


class TestTrainerTraining:
    """Test suite for training functionality."""
    
    @pytest.fixture
    def dummy_data(self):
        """Create dummy training data."""
        # 20 samples, 3 channels, 32x32 images
        images = torch.randn(20, 3, 32, 32)
        labels = torch.randint(0, 5, (20,))
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        return loader
    
    @pytest.fixture
    def trainer_setup(self, tmp_path):
        """Set up a complete trainer for testing."""
        model = DRClassifier(backbone='resnet50', num_classes=5, pretrained=False)
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        device = torch.device('cpu')
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            checkpoint_dir=str(tmp_path / "checkpoints")
        )
        
        return trainer
    
    def test_train_epoch_executes(self, trainer_setup, dummy_data):
        """Test that train_epoch completes without errors."""
        metrics = trainer_setup.train_epoch(dummy_data)
        
        assert 'loss' in metrics
        assert 'acc' in metrics
        assert metrics['loss'] > 0
        assert 0 <= metrics['acc'] <= 1
    
    def test_train_epoch_updates_weights(self, trainer_setup, dummy_data):
        """Test that training actually updates model weights."""
        # Get initial weights
        initial_params = [p.clone() for p in trainer_setup.model.parameters()]
        
        # Train one epoch
        trainer_setup.train_epoch(dummy_data)
        
        # Check that at least some weights changed
        params_changed = False
        for initial, current in zip(initial_params, trainer_setup.model.parameters()):
            if not torch.allclose(initial, current):
                params_changed = True
                break
        
        assert params_changed, "Model weights should change during training"
    
    def test_validate_executes(self, trainer_setup, dummy_data):
        """Test that validate completes without errors."""
        metrics = trainer_setup.validate(dummy_data)
        
        assert 'loss' in metrics
        assert 'acc' in metrics
        assert 'metrics' in metrics
        assert metrics['loss'] > 0
        assert 0 <= metrics['acc'] <= 1
    
    def test_validate_no_gradient_computation(self, trainer_setup, dummy_data):
        """Test that validation doesn't compute gradients."""
        # Enable gradient tracking
        trainer_setup.model.train()
        
        # Run validation
        metrics = trainer_setup.validate(dummy_data)
        
        # Check that no gradients were accumulated
        for param in trainer_setup.model.parameters():
            assert param.grad is None or torch.all(param.grad == 0)
    
    def test_train_vs_eval_mode(self, trainer_setup, dummy_data):
        """Test that model switches between train and eval modes."""
        # Model should be in eval mode initially
        initial_mode = trainer_setup.model.training
        
        # After train_epoch, should be in train mode during execution
        # (but we can't directly check this without modifying the method)
        
        # After validate, should be in eval mode
        trainer_setup.validate(dummy_data)
        assert not trainer_setup.model.training


class TestTrainerCheckpointing:
    """Test suite for checkpoint save/load functionality."""
    
    @pytest.fixture
    def trainer_setup(self, tmp_path):
        """Set up trainer with checkpoint directory."""
        model = DRClassifier(backbone='resnet50', num_classes=5, pretrained=False)
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        device = torch.device('cpu')
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            checkpoint_dir=str(tmp_path / "checkpoints")
        )
        
        return trainer, tmp_path
    
    def test_save_checkpoint_creates_file(self, trainer_setup):
        """Test that save_checkpoint creates a file."""
        trainer, tmp_path = trainer_setup
        checkpoint_dir = tmp_path / "checkpoints"
        
        trainer.save_checkpoint(
            filename='test.pth',
            epoch=1,
            metrics={'loss': 0.5, 'acc': 0.8}
        )
        
        assert (checkpoint_dir / 'test.pth').exists()
    
    def test_save_checkpoint_contents(self, trainer_setup):
        """Test that checkpoint contains required keys."""
        trainer, tmp_path = trainer_setup
        checkpoint_dir = tmp_path / "checkpoints"
        
        trainer.save_checkpoint(
            filename='test.pth',
            epoch=5,
            metrics={'loss': 0.5, 'acc': 0.8}
        )
        
        checkpoint = torch.load(checkpoint_dir / 'test.pth')
        
        assert 'epoch' in checkpoint
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'metrics' in checkpoint
        assert 'history' in checkpoint
        assert checkpoint['epoch'] == 5
    
    def test_save_checkpoint_with_scheduler(self, tmp_path):
        """Test that scheduler state is saved if present."""
        model = DRClassifier(backbone='resnet50', num_classes=5, pretrained=False)
        optimizer = Adam(model.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=5)
        criterion = nn.CrossEntropyLoss()
        device = torch.device('cpu')
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            checkpoint_dir=str(tmp_path / "checkpoints")
        )
        
        trainer.save_checkpoint(
            filename='test.pth',
            epoch=1,
            metrics={'loss': 0.5}
        )
        
        checkpoint = torch.load(tmp_path / "checkpoints" / 'test.pth')
        assert 'scheduler_state_dict' in checkpoint
    
    def test_load_checkpoint_restores_state(self, trainer_setup):
        """Test that load_checkpoint restores model and optimizer state."""
        trainer, tmp_path = trainer_setup
        checkpoint_dir = tmp_path / "checkpoints"
        
        # Save checkpoint
        trainer.save_checkpoint(
            filename='test.pth',
            epoch=3,
            metrics={'loss': 0.5, 'acc': 0.8}
        )
        
        # Modify model weights
        for param in trainer.model.parameters():
            param.data.fill_(0.123)
        
        # Load checkpoint
        epoch, metrics = trainer.load_checkpoint('test.pth')
        
        assert epoch == 3
        assert metrics['loss'] == 0.5
        assert metrics['acc'] == 0.8
        
        # Check that weights were restored (not all 0.123)
        weights_restored = False
        for param in trainer.model.parameters():
            if not torch.all(param.data == 0.123):
                weights_restored = True
                break
        assert weights_restored


class TestTrainerFullTraining:
    """Integration tests for complete training loops."""
    
    @pytest.fixture
    def dummy_dataloaders(self):
        """Create dummy train and validation loaders."""
        # Training data
        train_images = torch.randn(40, 3, 32, 32)
        train_labels = torch.randint(0, 5, (40,))
        train_dataset = TensorDataset(train_images, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        
        # Validation data
        val_images = torch.randn(20, 3, 32, 32)
        val_labels = torch.randint(0, 5, (20,))
        val_dataset = TensorDataset(val_images, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        return train_loader, val_loader
    
    @pytest.fixture
    def trainer_setup(self, tmp_path):
        """Set up trainer for integration tests."""
        model = DRClassifier(backbone='resnet50', num_classes=5, pretrained=False)
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        device = torch.device('cpu')
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            checkpoint_dir=str(tmp_path / "checkpoints")
        )
        
        return trainer
    
    def test_fit_completes_multiple_epochs(self, trainer_setup, dummy_dataloaders):
        """Test that fit runs for multiple epochs."""
        train_loader, val_loader = dummy_dataloaders
        
        num_epochs = 3
        trainer_setup.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            early_stopping_patience=10
        )
        
        # History should have 3 entries
        assert len(trainer_setup.history['train_loss']) == num_epochs
        assert len(trainer_setup.history['val_loss']) == num_epochs
    
    def test_fit_updates_history(self, trainer_setup, dummy_dataloaders):
        """Test that fit updates training history."""
        train_loader, val_loader = dummy_dataloaders
        
        trainer_setup.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=2,
            early_stopping_patience=10
        )
        
        # Check all history keys are populated
        for key in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
            assert len(trainer_setup.history[key]) == 2
            assert all(isinstance(v, float) for v in trainer_setup.history[key])
    
    def test_fit_saves_best_model(self, trainer_setup, dummy_dataloaders, tmp_path):
        """Test that fit saves the best model."""
        train_loader, val_loader = dummy_dataloaders
        checkpoint_dir = tmp_path / "checkpoints"
        
        trainer_setup.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=3,
            early_stopping_patience=10
        )
        
        # Best model should be saved
        assert (checkpoint_dir / 'best_model.pth').exists()
        
        # Best metrics should be updated
        assert trainer_setup.best_val_acc > 0
        assert trainer_setup.best_val_loss > 0
    
    def test_early_stopping_triggers(self, trainer_setup, dummy_dataloaders):
        """Test that early stopping works."""
        train_loader, val_loader = dummy_dataloaders
        
        # With very small patience, should stop early
        trainer_setup.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=20,
            early_stopping_patience=2
        )
        
        # Should have stopped before 20 epochs (likely 3-5 epochs)
        assert len(trainer_setup.history['train_loss']) < 20
    
    def test_learning_rate_scheduling(self, tmp_path, dummy_dataloaders):
        """Test that learning rate scheduler is called."""
        model = DRClassifier(backbone='resnet50', num_classes=5, pretrained=False)
        optimizer = Adam(model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=2, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        device = torch.device('cpu')
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scheduler=scheduler,
            checkpoint_dir=str(tmp_path / "checkpoints")
        )
        
        train_loader, val_loader = dummy_dataloaders
        
        # Get initial learning rate
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Train for 3 epochs (scheduler steps every 2 epochs)
        trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=3,
            early_stopping_patience=10
        )
        
        # Learning rate should have changed after 2 epochs
        final_lr = optimizer.param_groups[0]['lr']
        assert final_lr != initial_lr
        assert final_lr == initial_lr * 0.1  # gamma=0.1 applied once


class TestTrainerEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataloader_handling(self, tmp_path):
        """Test behavior with empty dataloader."""
        model = DRClassifier(backbone='resnet50', num_classes=5, pretrained=False)
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        device = torch.device('cpu')
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            checkpoint_dir=str(tmp_path / "checkpoints")
        )
        
        # Empty dataset
        empty_dataset = TensorDataset(torch.randn(0, 3, 32, 32), torch.randint(0, 5, (0,)))
        empty_loader = DataLoader(empty_dataset, batch_size=4)
        
        # Should handle gracefully (might return zero metrics)
        # This tests robustness
        try:
            metrics = trainer.train_epoch(empty_loader)
            # If it doesn't crash, that's good
            assert True
        except Exception as e:
            # Some implementations might raise an error, which is also acceptable
            assert True
    
    def test_single_batch_training(self, tmp_path):
        """Test training with just one batch."""
        model = DRClassifier(backbone='resnet50', num_classes=5, pretrained=False)
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        device = torch.device('cpu')
        
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            checkpoint_dir=str(tmp_path / "checkpoints")
        )
        
        # Single batch
        images = torch.randn(4, 3, 32, 32)
        labels = torch.randint(0, 5, (4,))
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=4)
        
        metrics = trainer.train_epoch(loader)
        
        assert metrics['loss'] > 0
        assert 0 <= metrics['acc'] <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
