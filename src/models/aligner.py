"""Full EEG <-> image alignment model (encoder + projection heads + optional adversary)."""

from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn as nn

from src.models.eeg_encoder import build_eeg_encoder
from src.models.projection_head import EEGProjectionHead
from src.models.subject_adversary import SubjectClassifier
from src.vision.image_projection import ImageProjectionHead

class NeuroVisionModel(nn.Module):
    def __init__(
        self, eeg_encoder: nn.Module, eeg_head: EEGProjectionHead,
        image_head: ImageProjectionHead, subject_classifier: Optional[SubjectClassifier] = None,
    ) -> None:
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.eeg_head = eeg_head
        self.image_head = image_head
        self.subject_classifier = subject_classifier
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.eeg_head(self.eeg_encoder(eeg))

    def encode_image(self, dinov2: torch.Tensor, clip: torch.Tensor) -> torch.Tensor:
        return self.image_head(dinov2, clip)

    def forward(
        self, eeg: torch.Tensor, dinov2: torch.Tensor, clip: torch.Tensor
    ) -> Dict[str, Optional[torch.Tensor]]:
        z_eeg = self.encode_eeg(eeg)
        z_img = self.encode_image(dinov2, clip)
        logits = self.subject_classifier(z_eeg) if self.subject_classifier is not None else None
        
        return {"z_eeg": z_eeg, "z_img": z_img, "logit_scale": self.logit_scale.exp().clamp(max=100), "subject_logits": logits}

def build_model(cfg: Any, num_subjects: int = 1) -> NeuroVisionModel:
    eeg_cfg, vis_cfg = cfg["eeg_model"], cfg["vision"]
    
    encoder = build_eeg_encoder(eeg_cfg)
    eeg_head = EEGProjectionHead(encoder.rep_dim, eeg_cfg["eeg_embed_dim"])
    image_head = ImageProjectionHead(
        dinov2_dim=vis_cfg.get("dinov2_dim", 384),
        clip_dim=vis_cfg.get("clip_dim", 512),
        embed_dim=vis_cfg.get("image_embed_dim", 512),
        hidden_dim=1024
    )

    adversary = None
    adv_cfg = cfg.get("subject_adversarial", {})
    if adv_cfg and adv_cfg.get("enabled", False) and num_subjects >= 2:
        adversary = SubjectClassifier(eeg_cfg["eeg_embed_dim"], num_subjects)
        
    return NeuroVisionModel(encoder, eeg_head, image_head, adversary)