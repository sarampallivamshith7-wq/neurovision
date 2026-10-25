"""Retrieval metrics (README section 11)."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch


def compute_ranks(similarity: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    """
    1-based rank of the correct gallery item for each query.

    Ties are resolved by *average rank* (1 + #strictly-better + 0.5 * #other-equal). This is
    deliberate: with optimistic tie handling a collapsed model that outputs identical
    embeddings for everything would score a perfect Recall@1.
    """
    sim = similarity.float()
    correct = sim.gather(1, target_idx.view(-1, 1))  # (Q, 1)
    greater = (sim > correct).sum(dim=1).float()
    equal_others = (sim == correct).sum(dim=1).float() - 1.0
    return 1.0 + greater + 0.5 * equal_others


def retrieval_metrics(
    similarity: torch.Tensor,
    target_idx: torch.Tensor,
    ks: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    """Recall@K, mean reciprocal rank, median / mean rank and mean correct-pair cosine."""
    ranks = compute_ranks(similarity, target_idx)
    out: Dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = (ranks <= k).float().mean().item()
    out["mrr"] = (1.0 / ranks).mean().item()
    out["median_rank"] = ranks.median().item()
    out["mean_rank"] = ranks.mean().item()
    out["mean_pair_cosine"] = similarity.float().gather(1, target_idx.view(-1, 1)).mean().item()
    out["chance_recall@1"] = 1.0 / similarity.shape[1]
    return out


def to_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
