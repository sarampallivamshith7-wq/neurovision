"""
EEG Preprocessing and Trial Epoching Pipeline for THINGS-EEG2.

Implements:
1. Zero-phase FIR band-pass filtering (0.5 - 40.0 Hz)
2. Notch filtering (50.0 Hz powerline)
3. Anti-aliased decimation / resampling (1000 Hz -> 250 Hz)
4. Event-aligned trial segmentation ([-0.1s, 0.8s] = 225 samples @ 250 Hz)
5. Pre-stimulus baseline correction ([-0.1s, 0.0s])
6. Trial-wise channel z-score normalization
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import mne
import numpy as np
import pandas as pd


EXPECTED_63_CHANNELS = [
    "Fp1", "F3", "F7", "FT9", "FC5", "FC1", "C3", "T7", "TP9", "CP5",
    "CP1", "Pz", "P3", "P7", "O1", "Oz", "O2", "P4", "P8", "TP10",
    "CP6", "CP2", "Cz", "C4", "T8", "FT10", "FC6", "FC2", "F4", "F8",
    "Fp2", "AF7", "AF3", "AFz", "F1", "F5", "FT7", "FC3", "FCz", "C1", "C5",
    "TP7", "CP3", "P1", "P5", "PO7", "PO3", "POz", "PO4", "PO8", "P6",
    "P2", "CPz", "CP4", "TP8", "C6", "C2", "FC4", "FT8", "F6", "F2",
    "AF4", "AF8",
]


def canonicalize_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    Make channel identity/order identical across sessions/subjects (README 4.1).

    Matches names case-insensitively, renames to the canonical spelling, and reorders to
    ``EXPECTED_63_CHANNELS`` (non-EEG channels are dropped). Raises ``ValueError`` listing the
    problem if the recording does not contain exactly the expected electrodes - important
    because the spatial graph assigns coordinates by position in this list.
    """
    eeg_names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)]
    by_lower = {n.lower(): n for n in eeg_names}
    expected_lower = {c.lower() for c in EXPECTED_63_CHANNELS}
    missing = [c for c in EXPECTED_63_CHANNELS if c.lower() not in by_lower]
    unexpected = [n for n in eeg_names if n.lower() not in expected_lower]
    if missing or unexpected:
        raise ValueError(
            "EEG channels do not match the expected 63-electrode layout. "
            f"Missing: {missing}; unexpected: {unexpected}"
        )
    rename = {by_lower[c.lower()]: c for c in EXPECTED_63_CHANNELS if by_lower[c.lower()] != c}
    if rename:
        raw.rename_channels(rename)
    raw.reorder_channels(EXPECTED_63_CHANNELS)
    return raw


def preprocess_raw(
    raw: mne.io.BaseRaw,
    l_freq: float = 0.5,
    h_freq: float = 40.0,
    notch_freq: float = 50.0,
    target_sfreq: float = 250.0,
    copy: bool = True,
    verbose: bool = False,
) -> mne.io.BaseRaw:
    """
    Apply band-pass, notch, and anti-aliasing resampling to a continuous EEG recording.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Continuous EEG Raw object.
    l_freq : float, default 0.5
        Lower cutoff frequency in Hz.
    h_freq : float, default 40.0
        Upper cutoff frequency in Hz.
    notch_freq : float, default 50.0
        Powerline notch frequency in Hz.
    target_sfreq : float, default 250.0
        Target sampling rate in Hz.
    copy : bool, default True
        Whether to operate on a copy of raw.
    verbose : bool, default False

    Returns
    -------
    raw_proc : mne.io.BaseRaw
        Preprocessed and resampled continuous EEG.
    """
    raw_proc = raw.copy() if copy else raw

    # Validate + canonicalise channel identity/order (always, not only when the count differs)
    canonicalize_channels(raw_proc)

    # Notch filter (50 Hz powerline)
    nyquist = raw_proc.info["sfreq"] / 2.0
    if notch_freq < nyquist:
        raw_proc.notch_filter(
            freqs=[notch_freq],
            picks="eeg",
            method="fir",
            phase="zero",
            verbose=verbose,
        )

    # Zero-phase FIR band-pass filter
    raw_proc.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        picks="eeg",
        method="fir",
        phase="zero",
        fir_window="hamming",
        fir_design="firwin",
        verbose=verbose,
    )

    # Resample with built-in anti-aliasing filter
    if target_sfreq < raw_proc.info["sfreq"]:
        raw_proc.resample(sfreq=target_sfreq, verbose=verbose)

    return raw_proc


def extract_epochs_from_events(
    raw: mne.io.BaseRaw,
    df_trials: pd.DataFrame,
    epoch_start: float = -0.1,
    epoch_end: float = 0.8,
    baseline_start: float = -0.1,
    baseline_end: float = 0.0,
    normalization: str = "trial_zscore",
    expected_channels: int = 63,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Segment continuous EEG data into fixed-window epochs aligned to trial onsets,
    apply baseline subtraction, and normalize.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Filtered and resampled continuous EEG.
    df_trials : pd.DataFrame
        Filtered trials manifest returned by parse_events_and_stimuli.
    epoch_start : float, default -0.1
        Start of epoch relative to stimulus onset in seconds.
    epoch_end : float, default 0.8
        End of epoch relative to stimulus onset in seconds.
    baseline_start : float, default -0.1
        Start of baseline window in seconds.
    baseline_end : float, default 0.0
        End of baseline window in seconds.
    normalization : str, default 'trial_zscore'
        Normalization strategy: 'trial_zscore' or 'none'.
    expected_channels : int, default 63
        Expected channel count.

    Returns
    -------
    epochs_data : np.ndarray
        Array of shape (n_trials, n_channels, n_samples).
    retained_df : pd.DataFrame
        Metadata rows corresponding to the retained valid trials.
    """
    sfreq = float(raw.info["sfreq"])
    raw_data = raw.get_data(picks="eeg")  # shape: (n_channels, total_samples)

    if raw_data.shape[0] != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} channels, got {raw_data.shape[0]} in raw data."
        )

    # Calculate exact sample offsets
    # Length in samples: round((epoch_end - epoch_start) * sfreq)
    # E.g. (0.8 - (-0.1)) * 250 = 0.9 * 250 = 225 samples
    epoch_samples = int(round((epoch_end - epoch_start) * sfreq))
    baseline_offset_start = int(round((baseline_start - epoch_start) * sfreq))
    baseline_offset_end = int(round((baseline_end - epoch_start) * sfreq))

    extracted_epochs: List[np.ndarray] = []
    valid_indices: List[int] = []

    total_time_samples = raw_data.shape[1]

    for idx, (_, row) in enumerate(df_trials.iterrows()):
        onset_sec = float(row["onset"])
        start_sample = int(round((onset_sec + epoch_start) * sfreq))
        end_sample = start_sample + epoch_samples

        # Boundary validation
        if start_sample < 0 or end_sample > total_time_samples:
            continue

        trial_slice = raw_data[:, start_sample:end_sample].copy()  # (63, epoch_samples)

        if trial_slice.shape[1] != epoch_samples:
            continue

        # Baseline subtraction
        if baseline_offset_end > baseline_offset_start >= 0:
            b_slice = trial_slice[:, baseline_offset_start:baseline_offset_end]
            baseline_mean = np.mean(b_slice, axis=-1, keepdims=True)
            trial_slice -= baseline_mean

        # Normalization
        if normalization == "trial_zscore":
            ch_mean = np.mean(trial_slice, axis=-1, keepdims=True)
            ch_std = np.std(trial_slice, axis=-1, keepdims=True)
            # Avoid divide-by-zero on flat channels
            ch_std = np.where(ch_std < 1e-8, 1.0, ch_std)
            trial_slice = (trial_slice - ch_mean) / ch_std

        # Check for NaN / Inf
        if np.isnan(trial_slice).any() or np.isinf(trial_slice).any():
            continue

        extracted_epochs.append(trial_slice)
        valid_indices.append(idx)

    if not extracted_epochs:
        raise ValueError("No valid epochs could be extracted from recording.")

    epochs_array = np.stack(extracted_epochs, axis=0).astype(np.float32)
    retained_df = df_trials.iloc[valid_indices].copy().reset_index(drop=True)
    retained_df["epoch_idx"] = np.arange(len(retained_df))

    return epochs_array, retained_df
