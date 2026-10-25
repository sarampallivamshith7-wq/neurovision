"""EEG projection head: maps an EEG representation into the shared embedding space."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGProjectionHead(nn.Module):
    """
    MLP projection head mapping an EEG representation (B, in_dim) to L2-normalised
    embeddings z_eeg in R^D (B, embed_dim). Mirrors ``ImageProjectionHead`` so both
    modalities land on the same unit hypersphere before InfoNCE.
    """

    def __init__(
        self,
        in_dim: int = 256,
        embed_dim: int = 512,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)
