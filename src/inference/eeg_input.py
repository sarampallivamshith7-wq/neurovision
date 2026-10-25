"""Validation of user-supplied EEG. Pure numpy: no torch / streamlit dependency.

Only *processed THINGS-EEG2 epochs* are accepted: (63 channels, 225 samples) at 250 Hz,
-0.1..0.8 s, already baseline-corrected and trial-wise channel z-scored (this is exactly the
format of data/test_eeg.npy). No filtering / resampling / z-scoring is applied here, because
the benchmark path does not apply them either: it feeds these epochs straight to whitening.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

N_CHANNELS = 63
N_SAMPLES = 225
SFREQ = 250.0
T_START_MS = -100.0
SUPPORTED_EXTENSIONS = (".npy",)


class EEGValidationError(ValueError):
    """Raised with a user-readable message when an uploaded EEG cannot be used."""


@dataclass
class ValidatedEEG:
    data: np.ndarray            # (63, 225) float32
    original_shape: tuple
    transposed: bool
    n_trials_averaged: int      # 1 for a single trial

    @property
    def duration_s(self) -> float:
        return self.data.shape[1] / SFREQ


def load_npy_bytes(raw: bytes, filename: str = "") -> np.ndarray:
    if filename and not filename.lower().endswith(SUPPORTED_EXTENSIONS):
        raise EEGValidationError(
            f"Unsupported file type '{filename}'. Only .npy files (processed 63x225 epochs) are supported.")
    try:
        arr = np.load(io.BytesIO(raw), allow_pickle=False)   # never unpickle uploads
    except Exception as e:  # corrupted / not an npy / pickled object array
        raise EEGValidationError(f"Could not read '{filename or 'file'}' as a valid .npy array ({type(e).__name__}: {e}).")
    if not isinstance(arr, np.ndarray):
        raise EEGValidationError("The file is not a single .npy array (.npz archives are not supported).")
    return arr


def validate_eeg(arr: np.ndarray) -> ValidatedEEG:
    """Return a clean (63, 225) float32 array or raise EEGValidationError.

    Accepted: (63, 225); (225, 63) (transposed, unambiguous because 63 != 225);
    (n, 63, 225) / (n, 225, 63) -> the n trials are averaged (n >= 1).
    """
    if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.complexfloating):
        raise EEGValidationError(f"EEG must be real-valued numeric, got dtype {arr.dtype}.")
    shape = tuple(arr.shape)
    expected = f"Expected EEG with {N_CHANNELS} channels and {N_SAMPLES} samples"

    if arr.ndim == 3:
        trials = arr
        n = shape[0]
        if n < 1:
            raise EEGValidationError(f"{expected}, but received an empty array of shape {shape}.")
        inner = shape[1:]
        if inner == (N_CHANNELS, N_SAMPLES):
            pass
        elif inner == (N_SAMPLES, N_CHANNELS):
            trials = np.swapaxes(arr, 1, 2)
        else:
            raise EEGValidationError(f"{expected} per trial, but received shape {shape}.")
        transposed = inner == (N_SAMPLES, N_CHANNELS)
        data, n_avg = trials.astype(np.float64), n
    elif arr.ndim == 2:
        if shape == (N_CHANNELS, N_SAMPLES):
            data, transposed = arr, False
        elif shape == (N_SAMPLES, N_CHANNELS):
            data, transposed = arr.T, True
        else:
            raise EEGValidationError(f"{expected}, but received shape {shape}.")
        data, n_avg = data[None].astype(np.float64), 1
    else:
        raise EEGValidationError(f"{expected}, but received a {arr.ndim}-D array of shape {shape}.")

    if not np.isfinite(data).all():
        n_nan, n_inf = int(np.isnan(data).sum()), int(np.isinf(data).sum())
        raise EEGValidationError(f"EEG contains non-finite values ({n_nan} NaN, {n_inf} Inf). Clean the data first.")

    out = data.mean(axis=0).astype(np.float32)
    if float(out.std()) < 1e-8:
        raise EEGValidationError("EEG signal is flat (zero variance); there is nothing to decode.")
    return ValidatedEEG(out, shape, transposed, n_avg)
