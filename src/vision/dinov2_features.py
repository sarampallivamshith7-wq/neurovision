"""
DINOv2 Visual Feature Extractor for NeuroVision.

Uses pretrained, frozen DINOv2 (dinov2_vits14) to extract self-supervised
visual representations (384-dimensional feature vector per image).
"""

from __future__ import annotations

from typing import List, Union

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


class DINOv2Extractor:
    """Extracts 384-dim visual representations using pretrained frozen DINOv2."""

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            verbose=False,
        )
        self.model.to(self.device)
        self.model.eval()

        # Standard DINOv2 evaluation preprocessing
        self.transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract_from_pil_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Extract features from a list of PIL Images.

        Parameters
        ----------
        images : list of PIL.Image.Image

        Returns
        -------
        features : torch.Tensor of shape (batch_size, 384)
        """
        tensors = [self.transform(img.convert("RGB")) for img in images]
        batch = torch.stack(tensors, dim=0).to(self.device)
        features = self.model(batch)  # shape: (B, 384)
        return features.cpu()

    @torch.no_grad()
    def extract_from_pil(self, image: Image.Image) -> torch.Tensor:
        """Extract feature from a single PIL Image. Returns shape (384,)."""
        return self.extract_from_pil_batch([image])[0]
