"""Contrastive objectives for EEG-image alignment (README section 8)."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F


def info_nce_loss(
    z_eeg: torch.Tensor,
    z_img: torch.Tensor,
    temperature: float = 0.07,
    symmetric: bool = True,
    stimulus_ids: Optional[Sequence] = None,
) -> torch.Tensor:
    """
    Symmetric InfoNCE between L2-normalised EEG and image embeddings.

    Row i of ``z_eeg`` is the positive of row i of ``z_img``; every other row is a negative.

    Parameters
    ----------
    stimulus_ids : optional sequence of length N
        THINGS-EEG2 shows each image several times, so a batch can contain the same image
        twice. Those entries are *not* real negatives; when ids are given, off-diagonal
        pairs sharing an id are masked out of the softmax (false-negative removal).
    """
    if z_eeg.shape != z_img.shape:
        raise ValueError(f"Shape mismatch: {tuple(z_eeg.shape)} vs {tuple(z_img.shape)}")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    n = z_eeg.size(0)
    logits = (z_eeg @ z_img.t()).float() / temperature  # (N, N): EEG rows x image cols

    if stimulus_ids is not None:
        if len(stimulus_ids) != n:
            raise ValueError("stimulus_ids must have one entry per batch element.")
        if torch.is_tensor(stimulus_ids):
            ids = stimulus_ids.to(logits.device)
            same = ids.view(-1, 1) == ids.view(1, -1)
        else:
            _, inverse = _unique_inverse(stimulus_ids)
            ids = torch.as_tensor(inverse, device=logits.device)
            same = ids.view(-1, 1) == ids.view(1, -1)
        false_neg = same & ~torch.eye(n, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(false_neg, float("-inf"))

    targets = torch.arange(n, device=logits.device)
    loss_e2i = F.cross_entropy(logits, targets)
    if not symmetric:
        return loss_e2i
    loss_i2e = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_e2i + loss_i2e)


def _unique_inverse(items: Sequence):
    mapping, inverse = {}, []
    for it in items:
        inverse.append(mapping.setdefault(it, len(mapping)))
    return mapping, inverse


@torch.no_grad()
def batch_retrieval_accuracy(
    z_eeg: torch.Tensor, z_img: torch.Tensor, stimulus_ids: Optional[Sequence] = None
) -> float:
    """
    In-batch EEG->image top-1 accuracy (cheap training-time signal).

    With ``stimulus_ids`` a hit means the retrieved image *is the same stimulus* - required
    because a batch may hold the same image several times (identical z_img rows).
    """
    pred = (z_eeg @ z_img.t()).argmax(dim=1)
    if stimulus_ids is None:
        return (pred == torch.arange(len(pred), device=pred.device)).float().mean().item()
    _, inverse = _unique_inverse(stimulus_ids)
    ids = torch.as_tensor(inverse, device=pred.device)
    return (ids[pred] == ids).float().mean().item()
