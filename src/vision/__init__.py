"""Vision feature extraction and projection package for NeuroVision.

Heavy dependencies (open_clip, torch.hub DINOv2) are imported lazily so that
lightweight components such as ``ImageProjectionHead`` and ``ImageFeatureCache``
can be used without them.
"""

from importlib import import_module
from typing import Any

_LAZY = {
    "DINOv2Extractor": "src.vision.dinov2_features",
    "CLIPExtractor": "src.vision.clip_features",
    "ImageProjectionHead": "src.vision.image_projection",
    "ImageFeatureCache": "src.vision.cache_embeddings",
    "cache_image_features": "src.vision.cache_embeddings",
}

__all__ = list(_LAZY)


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'src.vision' has no attribute '{name}'")
