from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageProjectionHead(nn.Module):
    def __init__(
        self, dinov2_dim: int = 384, clip_dim: int = 512, 
        embed_dim: int = 512, hidden_dim: int = 1024, dropout: float = 0.2,
    ) -> None:
        super().__init__()
        in_dim = dinov2_dim + clip_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, dinov2_feats: torch.Tensor, clip_feats: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([dinov2_feats, clip_feats], dim=-1) 
        projected = self.net(fused) 
        return F.normalize(projected, p=2, dim=-1)