"""
Subject-adversarial learning components (README section 9).

    EEG embedding ──► alignment loss
          └──► Gradient Reversal Layer ──► subject classifier

The classifier learns to predict the participant; the reversed gradient pushes the
EEG encoder to *remove* subject-identifying information.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:  # type: ignore[override]
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return -ctx.lambd * grad_output, None


class GradientReversal(nn.Module):
    """Identity in the forward pass; multiplies the gradient by ``-lambda`` backwards."""

    def __init__(self, lambd: float = 1.0) -> None:
        super().__init__()
        self.lambd = float(lambd)

    def set_lambda(self, lambd: float) -> None:
        self.lambd = float(lambd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFn.apply(x, self.lambd)


def dann_lambda(progress: float, gamma: float = 10.0, max_lambda: float = 1.0) -> float:
    """
    Standard DANN ramp-up 2 / (1 + exp(-gamma * p)) - 1, scaled by ``max_lambda``.
    ``progress`` in [0, 1]. Starts at 0 so alignment is learned before the adversary bites.
    """
    progress = min(max(progress, 0.0), 1.0)
    return max_lambda * (2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0)


class SubjectClassifier(nn.Module):
    """Gradient-reversal + MLP predicting the participant from an EEG embedding."""

    def __init__(
        self,
        in_dim: int = 512,
        num_subjects: int = 10,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        lambd: float = 1.0,
    ) -> None:
        super().__init__()
        if num_subjects < 2:
            raise ValueError("Subject-adversarial training needs at least 2 subjects.")
        self.grl = GradientReversal(lambd)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_subjects),
        )

    def set_lambda(self, lambd: float) -> None:
        self.grl.set_lambda(lambd)

    def forward(self, z_eeg: torch.Tensor) -> torch.Tensor:
        return self.net(self.grl(z_eeg))
