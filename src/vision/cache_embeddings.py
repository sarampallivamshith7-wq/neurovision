"""
Visual Feature Extraction and Caching Utility for NeuroVision.

Computes and saves fixed DINOv2 and CLIP feature vectors for unique images
to prevent recomputing vision backbones on every epoch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data.event_parser import StimuliResolver


class ImageFeatureCache:
    """
    In-memory or on-disk feature cache for precomputed DINOv2 and CLIP image embeddings.
    """

    def __init__(
        self,
        cache_file: Union[str, Path] = "data/embeddings/image_features.npz",
        metadata_file: Union[str, Path] = "data/embeddings/image_metadata.csv",
    ) -> None:
        self.cache_file = Path(cache_file).resolve()
        self.metadata_file = Path(metadata_file).resolve()
        self.stim_id_to_idx: Dict[str, int] = {}
        self.dinov2_feats: Optional[np.ndarray] = None
        self.clip_feats: Optional[np.ndarray] = None
        self.metadata_df: Optional[pd.DataFrame] = None

        if self.cache_file.exists() and self.metadata_file.exists():
            self.load()

    @classmethod
    def from_arrays(
        cls,
        stimulus_ids: Sequence[str],
        dinov2_feats: np.ndarray,
        clip_feats: np.ndarray,
    ) -> "ImageFeatureCache":
        """Build a cache directly from arrays (used for tests / synthetic data)."""
        if not (len(stimulus_ids) == len(dinov2_feats) == len(clip_feats)):
            raise ValueError("stimulus_ids, dinov2_feats and clip_feats must have equal length.")
        obj = cls.__new__(cls)
        obj.cache_file = obj.metadata_file = Path(".")
        obj.dinov2_feats = np.asarray(dinov2_feats, dtype=np.float32)
        obj.clip_feats = np.asarray(clip_feats, dtype=np.float32)
        obj.metadata_df = pd.DataFrame({"stimulus_id": list(stimulus_ids)})
        obj.stim_id_to_idx = {sid: i for i, sid in enumerate(stimulus_ids)}
        return obj

    def load(self) -> None:
        """Load precomputed feature arrays and metadata from disk."""
        data = np.load(self.cache_file)
        self.dinov2_feats = data["dinov2"]
        self.clip_feats = data["clip"]
        self.metadata_df = pd.read_csv(self.metadata_file)
        self.stim_id_to_idx = {
            stim_id: idx for idx, stim_id in enumerate(self.metadata_df["stimulus_id"])
        }

    def get_features(
        self, stimulus_ids: Sequence[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Look up precomputed (DINOv2, CLIP) feature tensors for given stimulus_ids.

        Returns
        -------
        dinov2_tensors : torch.Tensor of shape (B, 384)
        clip_tensors : torch.Tensor of shape (B, 512)
        """
        if self.dinov2_feats is None or self.clip_feats is None:
            raise RuntimeError("ImageFeatureCache is not loaded or cache files do not exist.")

        missing = [sid for sid in stimulus_ids if sid not in self.stim_id_to_idx]
        if missing:
            raise KeyError(
                f"{len(missing)} stimulus id(s) are not in the feature cache "
                f"(first few: {missing[:3]}). Re-run scripts/cache_features.py."
            )

        indices = [self.stim_id_to_idx[sid] for sid in stimulus_ids]
        dino_sub = torch.from_numpy(self.dinov2_feats[indices]).float()
        clip_sub = torch.from_numpy(self.clip_feats[indices]).float()
        return dino_sub, clip_sub

    def is_cached(self, stimulus_id: str) -> bool:
        return stimulus_id in self.stim_id_to_idx

    def __len__(self) -> int:
        return len(self.stim_id_to_idx)


def cache_image_features(
    stimulus_ids: List[str],
    resolver: StimuliResolver,
    output_dir: Union[str, Path] = "data/embeddings",
    batch_size: int = 32,
    device: str = "cpu",
    dinov2_model: str = "dinov2_vits14",
    clip_model: str = "ViT-B-32",
    clip_pretrained: str = "openai",
    dino_extractor: Optional[Any] = None,
    clip_extractor: Optional[Any] = None,
) -> Tuple[Path, Path]:
    """
    Extract DINOv2 and CLIP features for a list of unique stimulus_ids and save to disk.

    Parameters
    ----------
    stimulus_ids : list of str
        Unique stimulus identifiers to extract.
    resolver : StimuliResolver
        Stimuli resolver for loading PIL Images.
    output_dir : str or Path
        Output directory for cache files.
    batch_size : int, default 32
    device : str, default 'cpu'
    dino_extractor, clip_extractor : optional
        Pre-built extractors exposing ``extract_from_pil_batch``. When omitted they are
        constructed from ``dinov2_model`` / ``clip_model`` (downloads weights on first use).

    Returns
    -------
    (cache_file, metadata_file) : (Path, Path)
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "image_features.npz"
    meta_path = out_dir / "image_metadata.csv"

    unique_ids = sorted(set(stimulus_ids))
    print(f"[Feature Cache] Extracting DINOv2 & CLIP for {len(unique_ids)} unique images...")

    if dino_extractor is None:
        from src.vision.dinov2_features import DINOv2Extractor

        dino_extractor = DINOv2Extractor(model_name=dinov2_model, device=device)
    if clip_extractor is None:
        from src.vision.clip_features import CLIPExtractor

        clip_extractor = CLIPExtractor(model_name=clip_model, pretrained=clip_pretrained, device=device)

    all_dino: List[np.ndarray] = []
    all_clip: List[np.ndarray] = []
    retained_records: List[Dict[str, Any]] = []

    for i in tqdm(range(0, len(unique_ids), batch_size), desc="Caching image features"):
        batch_ids = unique_ids[i : i + batch_size]
        pil_images = []
        valid_ids = []

        for sid in batch_ids:
            try:
                img = resolver.load_pil_image(sid)
                pil_images.append(img)
                valid_ids.append(sid)
            except Exception as e:
                print(f"Warning: Could not load image for {sid}: {e}")

        if not pil_images:
            continue

        dino_batch = dino_extractor.extract_from_pil_batch(pil_images)  # (B, 384)
        clip_batch = clip_extractor.extract_from_pil_batch(pil_images)  # (B, 512)

        all_dino.append(dino_batch.numpy())
        all_clip.append(clip_batch.numpy())

        for vid in valid_ids:
            retained_records.append({
                "stimulus_id": vid,
                "concept": resolver.stim_id_to_concept.get(vid, ""),
                "orig_filename": resolver.stim_id_to_orig_filename.get(vid, ""),
                "bids_filename": resolver.stim_id_to_bids_filename.get(vid, ""),
                "is_test": "test" in vid,
            })

    final_dino = np.vstack(all_dino) if all_dino else np.empty((0, 384), dtype=np.float32)
    final_clip = np.vstack(all_clip) if all_clip else np.empty((0, 512), dtype=np.float32)
    df_meta = pd.DataFrame(retained_records)
    df_meta["vector_id"] = np.arange(len(df_meta))

    np.savez_compressed(cache_path, dinov2=final_dino, clip=final_clip)
    df_meta.to_csv(meta_path, index=False)

    print(f"[Feature Cache] Saved {len(df_meta)} image representations to {cache_path}")
    return cache_path, meta_path
