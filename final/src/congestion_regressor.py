import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, EfficientNet_B0_Weights

class CongestionRegressor(nn.Module):
    def __init__(self,
                 backbone_name: str = "resnet18",
                 fine_tune: bool = False,
                 dropout_p: float = 0.5):
        """
        backbone_name: one of ["resnet18", "resnet34", "efficientnet_b0"]
        fine_tune: if True, unfreeze layer4 (or equivalent) + head; if False, freeze all except head
        dropout_p: dropout probability in the regression head
        """
        super().__init__()

        if backbone_name == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1
            base_model = models.resnet18(weights=weights)
        elif backbone_name == "resnet34":
            weights = ResNet34_Weights.IMAGENET1K_V1
            base_model = models.resnet34(weights=weights)
        elif backbone_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            base_model = models.efficientnet_b0(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.backbone_name = backbone_name
        self.backbone = base_model

        # Freeze or unfreeze as appropriate
        if not fine_tune:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            # For ResNets: unfreeze layer4 + fc; for EfficientNet: unfreeze features[-1] + classifier
            if backbone_name.startswith("resnet"):
                for name, param in self.backbone.named_parameters():
                    if name.startswith("layer4") or name.startswith("fc"):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            elif backbone_name.startswith("efficientnet"):
                # EfficientNet's last block is features[7] or so; keep classifier
                for name, param in self.backbone.named_parameters():
                    if name.startswith("features.7") or name.startswith("classifier"):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

        # Replace classification head with a regression head
        if backbone_name.startswith("resnet"):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Linear(in_features, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_p),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        elif backbone_name.startswith("efficientnet"):
            in_features = self.backbone.classifier[1].in_features  # classifier is [Dropout, Linear]
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(dropout_p),
                nn.Linear(in_features, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_p),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        if self.backbone_name.startswith("resnet"):
            return self.backbone(x).squeeze(1)
        elif self.backbone_name.startswith("efficientnet"):
            return self.backbone(x).squeeze(1)

    def set_fine_tune(self, fine_tune: bool = True):
        """
        Switch between freezing all backbone params and unfreezing layer4 (or equivalent) + head.
        """
        if not fine_tune:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Then ensure head is trainable
            if self.backbone_name.startswith("resnet"):
                for param in self.backbone.fc.parameters():
                    param.requires_grad = True
            else:  # efficientnet
                for param in self.backbone.classifier.parameters():
                    param.requires_grad = True

        else:
            if self.backbone_name.startswith("resnet"):
                for name, param in self.backbone.named_parameters():
                    # Unfreeze layer4 + fc
                    if name.startswith("layer4") or name.startswith("fc"):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            else:  # efficientnet
                for name, param in self.backbone.named_parameters():
                    # Unfreeze last feature block (features.7) + classifier
                    if name.startswith("features.7") or name.startswith("classifier"):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False