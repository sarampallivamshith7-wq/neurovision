"""
EEGNet Temporal-Spatial Encoder for NeuroVision Phase 1.

Implements a compact CNN architecture tailored for EEG representation learning
inspired by Lawhern et al. (2018):
- Temporal 2D convolutions (learning frequency band filters)
- Spatial depthwise convolutions (learning spatial filters across 63 electrodes)
- Separable convolutions (combining temporal and spatial features efficiently)
- Projection to dense representation vector (256-dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Conv2dWithConstraint(nn.Conv2d):
    """Conv2d with max-norm weight constraint (used for spatial filters)."""

    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.max_norm is not None:
            with torch.no_grad():
                norm = self.weight.norm(2, dim=(1, 2, 3), keepdim=True)
                desired = torch.clamp(norm, max=self.max_norm)
                self.weight.mul_(desired / (norm + 1e-8))
        return super().forward(x)


class EEGNetEncoder(nn.Module):
    """
    EEGNet-based encoder for continuous 63-channel EEG trial segments.

    Parameters
    ----------
    in_channels : int, default 63
        Number of EEG electrodes.
    in_samples : int, default 225
        Number of time samples per trial (0.9s @ 250 Hz).
    temporal_filters : int, default 16
        Number of temporal filters (F1).
    temporal_kernel : int, default 32
        Length of temporal kernel (~128 ms @ 250 Hz).
    depth_multiplier : int, default 2
        Number of spatial filters per temporal filter (D).
    separable_filters : int, default 32
        Number of pointwise filters (F2).
    separable_kernel : int, default 16
    rep_dim : int, default 256
        Output representation dimension.
    dropout : float, default 0.25
    """

    def __init__(
        self,
        in_channels: int = 63,
        in_samples: int = 225,
        temporal_filters: int = 16,
        temporal_kernel: int = 32,
        depth_multiplier: int = 2,
        separable_filters: int = 32,
        separable_kernel: int = 16,
        rep_dim: int = 256,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.in_samples = in_samples
        self.rep_dim = rep_dim

        f1 = temporal_filters
        d = depth_multiplier
        f2 = separable_filters

        # Block 1: Temporal Conv + Depthwise Spatial Conv
        self.block1 = nn.Sequential(
            # Temporal filter (F1, 1, temporal_kernel)
            nn.Conv2d(
                1,
                f1,
                kernel_size=(1, temporal_kernel),
                padding=(0, temporal_kernel // 2),
                bias=False,
            ),
            nn.BatchNorm2d(f1),
            # Spatial filter across all electrodes (F1 * D, 1, 1)
            Conv2dWithConstraint(
                f1,
                f1 * d,
                kernel_size=(in_channels, 1),
                groups=f1,
                bias=False,
                max_norm=1.0,
            ),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        # Block 2: Separable Conv (Depthwise + Pointwise)
        self.block2 = nn.Sequential(
            nn.Conv2d(
                f1 * d,
                f1 * d,
                kernel_size=(1, separable_kernel),
                padding=(0, separable_kernel // 2),
                groups=f1 * d,
                bias=False,
            ),
            nn.Conv2d(f1 * d, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        # Calculate flattened dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_channels, in_samples)
            out1 = self.block1(dummy)
            out2 = self.block2(out1)
            flattened_dim = out2.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flattened_dim, rep_dim),
            nn.LayerNorm(rep_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            EEG trial tensor of shape (B, 63, 225) or (B, 1, 63, 225).

        Returns
        -------
        rep : torch.Tensor of shape (B, rep_dim)
        """
        if x.ndim == 3:
            # (B, 63, 225) -> (B, 1, 63, 225)
            x = x.unsqueeze(1)
        elif x.ndim != 4:
            raise ValueError(f"Expected 3D or 4D EEG tensor, got shape {x.shape}")

        b1 = self.block1(x)
        b2 = self.block2(b1)
        flat = torch.flatten(b2, start_dim=1)
        return self.fc(flat)
