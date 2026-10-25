"""
PyTorch Dataset definition for paired THINGS-EEG2 trials.

Guarantees strict pairing between EEG trial signals and corresponding image stimuli identifiers.
The EEG array may be an in-memory ndarray, a ``np.load(..., mmap_mode="r")`` memmap, or a
``ConcatArray`` over several memmaps; trials are only read from disk when indexed, so the
full dataset is never required to fit in RAM.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ConcatArray:
    """Lazy concatenation of several arrays/memmaps along axis 0 (supports int indexing)."""

    def __init__(self, arrays: Sequence[np.ndarray]) -> None:
        if not arrays:
            raise ValueError("ConcatArray needs at least one array.")
        tail = arrays[0].shape[1:]
        if any(a.shape[1:] != tail for a in arrays):
            raise ValueError("All arrays must share the same trailing shape.")
        self.arrays = list(arrays)
        self._starts = np.cumsum([0] + [len(a) for a in arrays])
        self.shape = (int(self._starts[-1]),) + tuple(tail)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, idx: int) -> np.ndarray:
        idx = int(idx)
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        part = int(np.searchsorted(self._starts, idx, side="right") - 1)
        return self.arrays[part][idx - self._starts[part]]


class ThingsEEGDataset(Dataset):
    """
    PyTorch Dataset yielding paired (EEG trial, stimulus_id, metadata).

    Parameters
    ----------
    eeg_data : np.ndarray, np.memmap, ConcatArray or torch.Tensor
        Array of shape (n_trials, n_channels, n_samples).
    metadata_df : pd.DataFrame
        Trial metadata with 'stimulus_id', 'tot_img_number', 'category_name'.
        Optional columns: 'trial_idx', 'subject_idx' (int, for subject-adversarial training).
    transform : callable, optional
        Optional transformation applied to the EEG tensor.
    """

    def __init__(
        self,
        eeg_data: Union[np.ndarray, torch.Tensor, ConcatArray],
        metadata_df: pd.DataFrame,
        transform: Optional[Any] = None,
    ) -> None:
        if len(eeg_data) != len(metadata_df):
            raise ValueError(
                f"Length mismatch: eeg_data has {len(eeg_data)} trials, "
                f"metadata_df has {len(metadata_df)} rows."
            )

        self.eeg = eeg_data.float() if torch.is_tensor(eeg_data) else eeg_data
        self.metadata = metadata_df.reset_index(drop=True)
        self.transform = transform

        # Pre-extract columns once: per-item pandas .iloc is very slow at 10^5 trials.
        m = self.metadata
        self._stim_ids = m["stimulus_id"].astype(str).tolist()
        self._tot_img = m["tot_img_number"].astype(int).to_numpy()
        self._category = m["category_name"].astype(str).tolist()
        self._trial_idx = (
            m["trial_idx"].astype(int).to_numpy() if "trial_idx" in m else np.arange(len(m))
        )
        self._subject_idx = (
            m["subject_idx"].astype(int).to_numpy() if "subject_idx" in m else np.zeros(len(m), dtype=int)
        )

    def __len__(self) -> int:
        return len(self.metadata)

    @property
    def eeg_tensor(self) -> torch.Tensor:
        """Backward-compatible full tensor view (materialises the data - avoid for large sets)."""
        return torch.stack([self._load(i) for i in range(len(self))])

    def _load(self, idx: int) -> torch.Tensor:
        item = self.eeg[idx]
        if torch.is_tensor(item):
            return item
        return torch.from_numpy(np.array(item, dtype=np.float32))  # copy: memmaps are read-only

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        eeg = self._load(idx)  # shape: (n_channels, n_samples)
        if self.transform is not None:
            eeg = self.transform(eeg)
        return {
            "eeg": eeg,
            "stimulus_id": self._stim_ids[idx],
            "tot_img_number": int(self._tot_img[idx]),
            "category_name": self._category[idx],
            "trial_idx": int(self._trial_idx[idx]),
            "subject_idx": int(self._subject_idx[idx]),
        }
