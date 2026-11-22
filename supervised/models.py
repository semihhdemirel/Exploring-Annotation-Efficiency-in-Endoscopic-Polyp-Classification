"""Model factory for the fully supervised baselines."""
import torch.nn as nn
from torchvision import models

ARCHITECTURES = ["resnet50", "efficientb0", "ViT"]

_EFFNET = {"efficientb0": "efficientnet-b0"}


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build an ImageNet-initialised classifier with a fresh `num_classes` head.

    resnet50     : torchvision ResNet-50 (IMAGENET1K_V1 weights)
    efficientb*  : `efficientnet_pytorch` (lukemelas) EfficientNet-B0/B1/B2/B7
    ViT          : torchvision ViT-B/32 (IMAGENET1K_V1 weights)
    """
    if arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if arch in _EFFNET:
        from efficientnet_pytorch import EfficientNet  # lazy: only if requested
        name = _EFFNET[arch]
        if pretrained:
            return EfficientNet.from_pretrained(name, num_classes=num_classes)
        return EfficientNet.from_name(name, num_classes=num_classes)
    if arch == "ViT":
        m = models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
        return m
    raise ValueError(f"Unknown architecture '{arch}'. Choose from {ARCHITECTURES}")
