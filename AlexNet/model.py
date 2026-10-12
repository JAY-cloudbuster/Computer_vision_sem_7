"""
AlexNet Architecture Module for License Plate Classification.

Architecture Highlights:
- Introduced by Krizhevsky, Sutskever, and Hinton (NeurIPS 2012).
- 5 Convolutional Layers + 3 Fully-Connected Layers.
- Key innovations: ReLU activation, Dropout regularization (0.5), Overlapping Max Pooling.
- Total parameters: ~57.01 Million (217.5 MB weights).
- Binary Head: Replaced 1000-class ImageNet FC(4096 -> 1000) with FC(4096 -> 2).
"""

import torch
import torch.nn as nn
from torchvision import models


class AlexNetPlateClassifier(nn.Module):
    """
    AlexNet for Plate vs. Non-Plate binary classification.
    Supports transfer learning with frozen convolutional backbone and fine-tuned classifier head.
    """
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=True):
        super().__init__()
        weights = models.AlexNet_Weights.DEFAULT if pretrained else None
        base_alexnet = models.alexnet(weights=weights)

        # Convolutional feature extractor
        self.features = base_alexnet.features
        self.avgpool = base_alexnet.avgpool

        # Freeze feature extraction layers for fast transfer learning
        if freeze_backbone and pretrained:
            for param in self.features.parameters():
                param.requires_grad = False

        # Classifier head: replace the final linear layer (index 6: 4096 -> num_classes)
        classifier_layers = list(base_alexnet.classifier.children())
        in_features = classifier_layers[-1].in_features
        classifier_layers[-1] = nn.Linear(in_features, num_classes)

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def count_parameters(self):
        """Returns total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def get_alexnet_model(num_classes=2, pretrained=True, freeze_backbone=True):
    """Factory function returning the AlexNet plate classifier."""
    return AlexNetPlateClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )
