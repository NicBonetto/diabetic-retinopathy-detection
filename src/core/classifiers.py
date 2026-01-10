import torch
import torch.nn as nn
from torchvision import models
from typing import Optional

class DRClassifier(nn.Module):
    def __init__(
        self,
        backbone: str='resnet50',
        num_classes: int=5,
        pretrained: bool=True,
        dropout: float=0.3
    ):
        super().__init__()

        self.backbone_name = backbone
        self.num_classes = num_classes

        self.backbone, feature_dim = self._load_backbone(backbone, pretrained)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def _load_backbone(self, backbone: str, pretrained: bool) -> tuple:
        """Load pre-trained backbone and return feature dimensions."""
        if backbone == 'resnet50':
            model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
            feature_dim = model.fc.in_features
            model = nn.Sequential(*list(model.children())[:-2])
        elif backbone == 'resnet101':
            model = models.resnet101(weights='IMAGENET1K_V2' if pretrained else None)
            feature_dim = model.fc.in_features
            model = nn.Sequential(*list(model.children())[:-2])
        elif backbone == 'efficientnet_b0':
            model = models.efficientnet_b0(weights='IMAGENET1K_V2' if pretrained else None)
            feature_dim = model.classifier[1].in_features
            model = model.features
        elif backbone == 'efficientnet_b3':
            model = models.efficientnet_b3(weights='IMAGENET1K_V2' if pretrained else None)
            feature_dim = model.classifier[1].in_features
            model = model.features
        elif backbone == 'vit_b_16':
            model = models.vit_b_16(weights='IMAGENET1K_V2' if pretrained else None)
            feature_dim = model.heads.head.in_features
            model.heads = nn.Identity()
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')

        return model, feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagation."""
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

class EnsembleClassifier(nn.Module):
    def __init__(self, models_list: list, ensemble_method: str='average'):
        super().__init__()

        self.models = nn.ModuleList(models_list)
        self.ensemble_method = ensemble_method

        for model in self.models:
            model.eval()

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward propagation through ensemble."""
        with torch.no_grad():
            outputs = []
            for model in self.models:
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                outputs.append(probs)

            outputs = torch.stack(outputs)

            mean_probs = outputs.mean(dim=0)
            if len(self.models) == 1:
                std_probs = torch.zeros_like(mean_probs)
            else:
                std_probs = outputs.std(dim=0)

            return mean_probs, std_probs

def create_model(
    backbone: str = 'resnet50',
    num_classes: int = 5,
    pretrained: bool = True,
    dropout: float = 0.3
) -> DRClassifier:
    """Factory function to create DR Classifier."""
    return DRClassifier(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )
