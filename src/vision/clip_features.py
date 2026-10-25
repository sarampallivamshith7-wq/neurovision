"""
CLIP Visual Feature Extractor for NeuroVision.

Uses pretrained, frozen CLIP (ViT-B/32) to extract semantically-aligned
visual representations (512-dimensional feature vector per image).
"""

from __future__ import annotations

from typing import List, Union

import open_clip
import torch
from PIL import Image


class CLIPExtractor:
    """Extracts 512-dim visual representations using pretrained frozen CLIP."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.device = torch.device(device)
        # open_clip names use dashes ("ViT-B-32"); accept the "ViT-B/32" spelling too.
        model_name = model_name.replace("/", "-")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.model.eval()

    @torch.no_grad()
    def extract_from_pil_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Extract CLIP features from a list of PIL Images.

        Parameters
        ----------
        images : list of PIL.Image.Image

        Returns
        -------
        features : torch.Tensor of shape (batch_size, 512)
        """
        tensors = [self.preprocess(img.convert("RGB")) for img in images]
        batch = torch.stack(tensors, dim=0).to(self.device)
        features = self.model.encode_image(batch)  # shape: (B, 512)
        return features.cpu()

    @torch.no_grad()
    def extract_from_pil(self, image: Image.Image) -> torch.Tensor:
        """Extract CLIP feature from a single PIL Image. Returns shape (512,)."""
        return self.extract_from_pil_batch([image])[0]
