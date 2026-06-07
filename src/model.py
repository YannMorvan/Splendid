import torch.nn as nn
from torchvision import models


def build_model(num_classes=2, pretrained=True):
    """MobileNetV3-Small with a replaced classification head."""
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    # Replace the final linear layer (index 3 in the classifier Sequential)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    return model


def freeze_backbone(model):
    """Freeze everything except the classification head."""
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True


def unfreeze_last_block(model):
    """Unfreeze the last inverted-residual block for fine-tuning."""
    for p in model.features[-1].parameters():
        p.requires_grad = True
