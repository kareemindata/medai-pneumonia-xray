import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
