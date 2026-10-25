"""
EEG Conformer encoder (README Phase 3), after Song et al. (2023).

Convolutional patch embedding (local temporal + spatial filtering) followed by a
Transformer encoder (global temporal self-attention), then temporal aggregation.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalTransformer(nn.Module):
    def __init__(
        self, d_model: int = 128, depth: int = 3, num_heads: int = 4, 
        ff_mult: int = 4, dropout: float = 0.25, max_len: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * ff_mult,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        x = x + self.pos_embed[:, :length, :]
        x = self.encoder(x)
        return self.norm(x)

class AttentionPool1d(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.scorer(x), dim=1)
        return (x * weights).sum(dim=1)             

class EEGConformerEncoder(nn.Module):
    def __init__(
        self, in_channels: int = 63, in_samples: int = 225, d_model: int = 128,
        temporal_filters: int = 40, temporal_kernel: int = 25, pool_kernel: int = 25,
        pool_stride: int = 5, depth: int = 3, num_heads: int = 4, ff_mult: int = 4,
        dropout: float = 0.25, rep_dim: int = 512, depthwise_spatial: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.in_samples = in_samples
        self.rep_dim = rep_dim

        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, temporal_filters, (1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False),
            nn.Conv2d(temporal_filters, temporal_filters, (in_channels, 1), groups=temporal_filters if depthwise_spatial else 1, bias=False),
            nn.BatchNorm2d(temporal_filters),
            nn.ELU(),
            nn.AvgPool2d((1, pool_kernel), (1, pool_stride)),
            nn.Dropout(dropout),
            nn.Conv2d(temporal_filters, d_model, (1, 1)),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_channels, in_samples)
            n_tokens = self.patch_embed(dummy).shape[-1]
        self.n_tokens = n_tokens

        self.transformer = TemporalTransformer(
            d_model=d_model, depth=depth, num_heads=num_heads,
            ff_mult=ff_mult, dropout=dropout, max_len=max(n_tokens, 8),
        )

        self.pool = AttentionPool1d(d_model)

        self.fc = nn.Sequential(
            nn.Linear(d_model, rep_dim),
            nn.LayerNorm(rep_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        tokens = self.patch_embed(x)
        tokens = tokens.squeeze(2).transpose(1, 2)
        tokens = self.transformer(tokens)
        pooled = self.pool(tokens)
        return self.fc(pooled)