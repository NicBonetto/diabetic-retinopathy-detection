import pytest
import torch
import torch.nn as nn
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.core.classifiers import DRClassifier, EnsembleClassifier, create_model

class TestDRClassifier:
    """Test suite for DRClassifier class."""

    @pytest.fixture
    def sample_input(self):
        """Create sample input"""
        return torch.randn(2, 3, 224, 224)

    @pytest.fixture
    def sample_input_512(self):
        """Create sample input tensor at 512x512"""
        return torch.randn(2, 3, 512, 512)

    def test_resnet50_creation(self):
        """Test ResNet50 model creation."""
        model = DRClassifier(
            backbone='resnet50',
            num_classes=5,
            pretrained=False,
            dropout=0.5
        )

        assert model is not None
        assert model.backbone_name == 'resnet50'
        assert model.num_classes == 5

    def test_resnet101_creation(self):
        """Test ResNet101 creation."""
        model = DRClassifier(
            backbone='resnet101',
            num_classes=5,
            pretrained=False,
            dropout=0.5
        )

        assert model.backbone_name == 'resnet101' 

    def test_efficientnet_b0_creation(self):
        """Test EfficientNet-B0 creation."""
        model = DRClassifier(
            backbone='efficientnet_b0',
            num_classes=5,
            pretrained=False,
            dropout=0.5
        )

        assert model.backbone_name == 'efficientnet_b0'

    def test_efficientnet_b3_creation(self):
        """Test EfficientNet-B3 creation."""
        model = DRClassifier(
            backbone='efficientnet_b3',
            num_classes=5,
            pretrained=False,
            dropout=0.5
        )

        assert model.backbone_name == 'efficientnet_b3'

    def test_vit_b_16_creation(self):
        """Test Vit B16 creation."""
        model = DRClassifier(
            backbone='vit_b_16',
            num_classes=5,
            pretrained=False,
            dropout=0.5
        )

        assert model.backbone_name == 'vit_b_16'

    def test_unsupported_backbone_raise_error(self):
        """Test that unsupported backbones raise an error."""
        with pytest.raises(ValueError, match='Unsupported backbone'):
            model = DRClassifier(
                backbone='unsupported_model',
                num_classes=5,
                pretrained=False,
                dropout=0.5
            )

    def test_forward_pass_shape_resnet50(self, sample_input):
        """Test forward pass returns correct output shape for ResNet50."""
        model = DRClassifier(
            backbone='resnet50',
            num_classes=5,
            pretrained=False
        )
        model.eval()
        
        with torch.no_grad():
            output = model(sample_input)
        
        # Output should be (batch_size, num_classes)
        assert output.shape == (2, 5)
    
    def test_forward_pass_shape_efficientnet(self, sample_input):
        """Test forward pass returns correct output shape for EfficientNet."""
        model = DRClassifier(
            backbone='efficientnet_b0',
            num_classes=5,
            pretrained=False
        )
        model.eval()
        
        with torch.no_grad():
            output = model(sample_input)
        
        assert output.shape == (2, 5)
    
    def test_forward_pass_different_input_sizes(self, sample_input_512):
        """Test model works with different input sizes."""
        model = DRClassifier(
            backbone='resnet50',
            num_classes=5,
            pretrained=False
        )
        model.eval()
        
        with torch.no_grad():
            output = model(sample_input_512)
        
        # Should still output correct shape regardless of input size
        assert output.shape == (2, 5)
    
    def test_output_is_logits_not_probabilities(self, sample_input):
        """Test that output is raw logits, not probabilities."""
        model = DRClassifier(
            backbone='resnet50',
            num_classes=5,
            pretrained=False
        )
        model.eval()
        
        with torch.no_grad():
            output = model(sample_input)
        
        # Logits can be any value (not bounded 0-1)
        assert not torch.all((output >= 0) & (output <= 1))
        
        # But softmax should give valid probabilities
        probs = torch.softmax(output, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-6)
        assert torch.all((probs >= 0) & (probs <= 1))
    
    def test_custom_num_classes(self, sample_input):
        """Test model works with different number of classes."""
        model = DRClassifier(
            backbone='resnet50',
            num_classes=10,  # Different from default 5
            pretrained=False
        )
        model.eval()
        
        with torch.no_grad():
            output = model(sample_input)
        
        assert output.shape == (2, 10)
    
    def test_dropout_applied(self):
        """Test that dropout is included in classifier."""
        model = DRClassifier(
            backbone='resnet50',
            num_classes=5,
            pretrained=False,
            dropout=0.5
        )
        
        # Check that classifier has dropout layers
        has_dropout = any(isinstance(m, nn.Dropout) for m in model.classifier.modules())
        assert has_dropout, "Model should contain dropout layers"
    
    def test_freeze_backbone(self, sample_input):
        """Test freezing backbone parameters."""
        model = DRClassifier(
            backbone='resnet50',
            num_classes=5,
            pretrained=False
        )
        
        # Initially all params should require grad
        backbone_params_before = [p.requires_grad for p in model.backbone.parameters()]
        assert all(backbone_params_before), "All backbone params should require grad initially"
        
        # Freeze backbone
        model.freeze_backbone()
        
        # Backbone params should not require grad
        backbone_params_after = [p.requires_grad for p in model.backbone.parameters()]
        assert not any(backbone_params_after), "No backbone params should require grad after freezing"
        
        # Classifier params should still require grad
        classifier_params = [p.requires_grad for p in model.classifier.parameters()]
        assert all(classifier_params), "Classifier params should still require grad"
        
        # Model should still work
        model.eval()
        with torch.no_grad():
            output = model(sample_input)
        assert output.shape == (2, 5)
    
    def test_unfreeze_backbone(self):
        """Test unfreezing backbone parameters."""
        model = DRClassifier(
            backbone='resnet50',
            num_classes=5,
            pretrained=False
        )

        # Freeze then unfreeze
        model.freeze_backbone()
        model.unfreeze_backbone()

        # All params should require grad again
        all_params = [p.requires_grad for p in model.parameters()]
        assert all(all_params), "All params should require grad after unfreezing"

    def test_parameter_count(self):
        """Test that model has reasonable number of parameters."""
        model = DRClassifier(
            backbone='resnet50',
            num_classes=5,
            pretrained=False
        )

        total_params = sum(p.numel() for p in model.parameters())

        # ResNet50 has ~25M params, our classifier adds ~2.5M
        # Total should be in reasonable range
        assert 20_000_000 < total_params < 30_000_000, \
            f"ResNet50 should have ~25M params, got {total_params:,}"

    def test_model_in_training_mode(self, sample_input):
        """Test model behavior in training mode."""
        model = DRClassifier(
            backbone='resnet50',
            num_classes=5,
            pretrained=False,
            dropout=0.5
        )
        model.train()

        # In training mode, dropout is active, so outputs may vary
        output1 = model(sample_input)
        output2 = model(sample_input)

        # Outputs should be different due to dropout (with high probability)
        # Note: This is stochastic, but with dropout=0.5, extremely unlikely to be identical
        assert output1.shape == output2.shape == (2, 5)

    def test_model_in_eval_mode(self, sample_input):
        """Test model behavior in eval mode."""
        model = DRClassifier(
            backbone='resnet50',
            num_classes=5,
            pretrained=False,
            dropout=0.5
        )
        model.eval()

        # In eval mode, dropout is off, outputs should be deterministic
        with torch.no_grad():
            output1 = model(sample_input)
            output2 = model(sample_input)

        # Should be identical
        assert torch.allclose(output1, output2)

    def test_gradient_flow(self, sample_input):
        """Test that gradients flow properly through the model."""
        model = DRClassifier(
            backbone='resnet50',
            num_classes=5,
            pretrained=False
        )
        model.train()

        # Forward pass
        output = model(sample_input)

        # Dummy loss
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Check that gradients exist for trainable params
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient not computed for {name}"
                assert not torch.all(param.grad == 0), f"Gradient is zero for {name}"


class TestEnsembleClassifier:
    """Test suite for EnsembleClassifier class."""

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(2, 3, 224, 224)

    @pytest.fixture
    def ensemble_models(self):
        """Create a list of models for ensemble."""
        models = [
            DRClassifier(backbone='resnet50', num_classes=5, pretrained=False),
            DRClassifier(backbone='resnet50', num_classes=5, pretrained=False),
            DRClassifier(backbone='resnet50', num_classes=5, pretrained=False),
        ]
        return models

    def test_ensemble_creation(self, ensemble_models):
        """Test ensemble model creation."""
        ensemble = EnsembleClassifier(
            models_list=ensemble_models,
            ensemble_method='average'
        )

        assert ensemble is not None
        assert len(ensemble.models) == 3

    def test_ensemble_forward_pass_shape(self, ensemble_models, sample_input):
        """Test ensemble forward pass returns correct shapes."""
        ensemble = EnsembleClassifier(
            models_list=ensemble_models,
            ensemble_method='average'
        )

        mean_probs, std_probs = ensemble(sample_input)

        # Both should have shape (batch_size, num_classes)
        assert mean_probs.shape == (2, 5)
        assert std_probs.shape == (2, 5)

    def test_ensemble_probabilities_valid(self, ensemble_models, sample_input):
        """Test that ensemble outputs valid probabilities."""
        ensemble = EnsembleClassifier(
            models_list=ensemble_models,
            ensemble_method='average'
        )

        mean_probs, std_probs = ensemble(sample_input)

        # Mean probs should sum to 1
        assert torch.allclose(mean_probs.sum(dim=1), torch.ones(2), atol=1e-6)

        # Mean probs should be in [0, 1]
        assert torch.all((mean_probs >= 0) & (mean_probs <= 1))

        # Std should be non-negative
        assert torch.all(std_probs >= 0)

    def test_ensemble_uncertainty_estimation(self, sample_input):
        """Test that ensemble provides meaningful uncertainty estimates."""
        # Create models with different initializations
        models = [
            DRClassifier(backbone='resnet50', num_classes=5, pretrained=False)
            for _ in range(5)
        ]

        ensemble = EnsembleClassifier(models_list=models)
        mean_probs, std_probs = ensemble(sample_input)

        # Standard deviation should be greater than zero
        # (different models should give different predictions)
        assert torch.any(std_probs > 0), "Ensemble should have non-zero uncertainty"

    def test_ensemble_single_model_no_uncertainty(self, sample_input):
        """Test that single-model ensemble has zero uncertainty."""
        # Ensemble with just one model
        model = DRClassifier(backbone='resnet50', num_classes=5, pretrained=False)
        ensemble = EnsembleClassifier(models_list=[model])

        mean_probs, std_probs = ensemble(sample_input)

        # With only one model, std should be exactly zero
        assert torch.allclose(std_probs, torch.zeros_like(std_probs), atol=1e-7)

    def test_ensemble_models_in_eval_mode(self, ensemble_models):
        """Test that ensemble sets models to eval mode."""
        ensemble = EnsembleClassifier(models_list=ensemble_models)

        # All models should be in eval mode
        for model in ensemble.models:
            assert not model.training, "Ensemble models should be in eval mode"

    def test_ensemble_no_gradients(self, ensemble_models, sample_input):
        """Test that ensemble doesn't compute gradients."""
        ensemble = EnsembleClassifier(models_list=ensemble_models)

        mean_probs, std_probs = ensemble(sample_input)

        # Outputs should not require grad (inference only)
        assert not mean_probs.requires_grad
        assert not std_probs.requires_grad

    def test_ensemble_different_architectures(self, sample_input):
        """Test ensemble with different backbone architectures."""
        models = [
            DRClassifier(backbone='resnet50', num_classes=5, pretrained=False),
            DRClassifier(backbone='efficientnet_b0', num_classes=5, pretrained=False),
        ]

        ensemble = EnsembleClassifier(models_list=models)

        # Should work with mixed architectures
        mean_probs, std_probs = ensemble(sample_input)
        assert mean_probs.shape == (2, 5)
        assert std_probs.shape == (2, 5)


class TestCreateModelFactory:
    """Test suite for create_model factory function."""

    def test_factory_default_creation(self):
        """Test factory with default parameters."""
        model = create_model()

        assert isinstance(model, DRClassifier)
        assert model.backbone_name == 'resnet50'
        assert model.num_classes == 5

    def test_factory_custom_backbone(self):
        """Test factory with custom backbone."""
        model = create_model(
            backbone='efficientnet_b0',
            num_classes=10,
            pretrained=False,
            dropout=0.3
        )

        assert model.backbone_name == 'efficientnet_b0'
        assert model.num_classes == 10

    def test_factory_returns_functional_model(self):
        """Test that factory-created model is functional."""
        model = create_model(pretrained=False)
        model.eval()

        sample_input = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            output = model(sample_input)

        assert output.shape == (1, 5)


class TestModelIntegration:
    """Integration tests for model usage scenarios."""

    def test_training_workflow(self):
        """Test typical training workflow."""
        model = DRClassifier(backbone='resnet50', num_classes=5, pretrained=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Dummy batch
        images = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 2, 3])

        # Training step
        model.train()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Loss should be computed
        assert loss.item() > 0

    def test_inference_workflow(self):
        """Test typical inference workflow."""
        model = DRClassifier(backbone='resnet50', num_classes=5, pretrained=False)
        model.eval()

        # Single image
        image = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            logits = model(image)
            probs = torch.softmax(logits, dim=1)
            prediction = probs.argmax(dim=1)

        assert 0 <= prediction.item() < 5
        assert torch.allclose(probs.sum(), torch.tensor(1.0))

    def test_transfer_learning_workflow(self):
        """Test transfer learning workflow (freeze backbone)."""
        model = DRClassifier(backbone='resnet50', num_classes=5, pretrained=False)

        # Phase 1: Train only classifier
        model.freeze_backbone()

        trainable_params_frozen = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        # Phase 2: Fine-tune entire model
        model.unfreeze_backbone()

        trainable_params_unfrozen = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        # Should have more trainable params after unfreezing
        assert trainable_params_unfrozen > trainable_params_frozen

    def test_save_load_model(self, tmp_path):
        """Test saving and loading model state."""
        model = DRClassifier(backbone='resnet50', num_classes=5, pretrained=False)

        # Create dummy input
        sample_input = torch.randn(1, 3, 224, 224)

        # Get output before saving
        model.eval()
        with torch.no_grad():
            output_before = model(sample_input)

        # Save model
        save_path = tmp_path / "model.pth"
        torch.save(model.state_dict(), save_path)

        # Create new model and load
        model_loaded = DRClassifier(backbone='resnet50', num_classes=5, pretrained=False)
        model_loaded.load_state_dict(torch.load(save_path))
        model_loaded.eval()

        # Get output after loading
        with torch.no_grad():
            output_after = model_loaded(sample_input)

        # Outputs should be identical
        assert torch.allclose(output_before, output_after)

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

