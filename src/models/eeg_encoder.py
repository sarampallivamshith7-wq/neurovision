"""Config-driven EEG encoder factory so experiments can switch architectures via YAML."""

from __future__ import annotations
from typing import Any
import torch.nn as nn
from src.models.eeg_conformer import EEGConformerEncoder
from src.models.eegnet_encoder import EEGNetEncoder
from src.models.spatial_gnn import GNNConformerEncoder

ENCODER_TYPES = ("eegnet", "conformer", "gnn_conformer")

def _get(section: Any, key: str, default: Any = None) -> Any:
    try:
        return section[key]
    except (KeyError, TypeError):
        return default

def build_eeg_encoder(eeg_cfg: Any) -> nn.Module:
    enc = str(eeg_cfg["encoder_type"]).lower()
    if enc not in ENCODER_TYPES:
        raise ValueError(f"Unknown encoder_type '{enc}'")

    common = dict(
        in_channels=int(eeg_cfg["in_channels"]),
        in_samples=int(eeg_cfg["in_samples"]),
        rep_dim=int(eeg_cfg["eeg_rep_dim"]),
    )

    conf = _get(eeg_cfg, "conformer", {}) or {}

    if enc == "eegnet":
        return EEGNetEncoder(
            temporal_filters=int(_get(eeg_cfg, "temporal_filters", 16)),
            temporal_kernel=int(_get(eeg_cfg, "temporal_kernel_size", 32)),
            depth_multiplier=int(_get(eeg_cfg, "spatial_depth", 2)),
            separable_filters=int(_get(eeg_cfg, "separable_filters", 32)),
            separable_kernel=int(_get(eeg_cfg, "separable_kernel_size", 16)),
            dropout=float(_get(eeg_cfg, "dropout", 0.25)),
            **common,
        )

    transformer_kwargs = dict(
        d_model=int(_get(conf, "d_model", 256)),
        depth=int(_get(conf, "depth", 4)),
        num_heads=int(_get(conf, "num_heads", 8)),
        ff_mult=int(_get(conf, "ff_mult", 4)),
        dropout=float(_get(conf, "dropout", 0.4)),
    )

    if enc == "conformer":
        return EEGConformerEncoder(
            temporal_filters=int(_get(conf, "temporal_filters", 40)),
            temporal_kernel=int(_get(conf, "temporal_kernel", 25)),
            pool_kernel=int(_get(conf, "pool_kernel", 25)),
            pool_stride=int(_get(conf, "pool_stride", 5)),
            depthwise_spatial=bool(_get(conf, "depthwise_spatial", True)),
            **transformer_kwargs,
            **common,
        )

    raise NotImplementedError("Encoder type not usable.")