"""FAISS-backed image-embedding index for EEG->image retrieval (README section 12.1).

Uses ``IndexFlatIP`` over L2-normalised vectors (== cosine similarity). Falls back to a
NumPy brute-force search if FAISS is not installed, so results are identical either way.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - import guard
    import faiss

    _HAS_FAISS = True
except Exception:  # pragma: no cover
    faiss = None
    _HAS_FAISS = False


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class ImageIndex:
    """Maps stimulus ids <-> embedding rows and answers top-K cosine queries."""

    def __init__(self, embeddings: np.ndarray, stimulus_ids: Sequence[str]) -> None:
        if len(embeddings) != len(stimulus_ids):
            raise ValueError("embeddings and stimulus_ids must have equal length.")
        self.stimulus_ids: List[str] = list(stimulus_ids)
        self._emb = _l2_normalize(embeddings)
        self.dim = self._emb.shape[1]
        self._index = None
        if _HAS_FAISS:
            self._index = faiss.IndexFlatIP(self.dim)
            self._index.add(self._emb)

    @property
    def backend(self) -> str:
        return "faiss" if self._index is not None else "numpy"

    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, List[List[str]]]:
        """Return (scores (Q,k), ids: list of Q lists of stimulus ids), best first."""
        k = min(k, len(self.stimulus_ids))
        q = _l2_normalize(queries)
        if self._index is not None:
            scores, idx = self._index.search(q, k)
        else:
            sims = q @ self._emb.T
            idx = np.argsort(-sims, axis=1)[:, :k]
            scores = np.take_along_axis(sims, idx, axis=1)
        return scores, [[self.stimulus_ids[j] for j in row] for row in idx]
