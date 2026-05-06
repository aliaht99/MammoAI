"""
EfficientNet-B4 fine-tuned for binary mammogram classification.

Two-phase training:
  Phase 1 (warmup)  — backbone frozen, only classifier head trains
  Phase 2 (finetune)— all layers unfrozen at a lower learning rate
"""

import torch
import torch.nn as nn
from torchvision import models

import config


class MammoNet(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if config.PRETRAINED else None
        backbone = models.efficientnet_b4(weights=weights)

        # replace classifier head: 1792 → 512 → 1
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=config.DROP_RATE),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.Dropout(p=config.DROP_RATE / 2),
            nn.Linear(512, config.NUM_CLASSES),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(1)   # (B,)

    def freeze_backbone(self):
        """Phase 1: freeze everything except the classifier head."""
        for name, param in self.backbone.named_parameters():
            param.requires_grad = "classifier" in name
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Frozen backbone — trainable params: {trainable:,}")

    def unfreeze_all(self):
        """Phase 2: unfreeze all layers for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Unfrozen all   — trainable params: {trainable:,}")


def load_checkpoint(path: str, device: torch.device) -> tuple["MammoNet", dict]:
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model = MammoNet().to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt
