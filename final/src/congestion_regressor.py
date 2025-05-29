# src/congestion_regressor.py

import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class CongestionRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        # Use the new weights enum instead of pretrained=True
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.backbone = models.resnet18(weights=weights)

        # freeze all layers except the final classifier
        for param in self.backbone.parameters():
            param.requires_grad = False

        # replace the final fc with a small regression head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)