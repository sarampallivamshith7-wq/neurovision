"""
Sanity check: Verify EEG preprocessing, resampling to 250 Hz, and epoching to (63, 225).
"""

from pathlib import Path
import mne
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.event_parser import parse_events_and_stimuli
from src.data.preprocessing import extract_epochs_from_events, preprocess_raw
from src.data.eeg_dataset import ThingsEEGDataset


@pytest.mark.data
def test_preprocessing_and_epoching_pipeline():
    bdf_path = Path("data/sub-01/sub-01/ses-01/eeg/sub-01_ses-01_task-train_run-01_eeg.bdf")
    events_path = Path("data/sub-01/sub-01/ses-01/eeg/sub-01_ses-01_task-train_run-01_events.tsv")

    if not (bdf_path.exists() and events_path.exists() and bdf_path.stat().st_size > 1000):
        pytest.skip("THINGS-EEG2 data not available (see tests/test_preprocessing_synthetic.py for dataset-free tests)")

    # 1. Parse first 10 image presentation trials
    df_trials, _ = parse_events_and_stimuli(
        events_tsv_path=events_path,
        max_trials=10,
    )
    assert len(df_trials) == 10

    # Read only first 30 seconds of raw EEG for fast testing (the 10 events occur between 11s and 15s)
    raw = mne.io.read_raw_bdf(bdf_path, preload=False, verbose=False)
    # Crop to first 30s to keep unit test sub-second
    raw_cropped = raw.crop(tmin=0.0, tmax=30.0).load_data()

    assert raw_cropped.info["sfreq"] == 1000.0
    assert len(raw_cropped.ch_names) == 63

    # 2. Preprocess: Bandpass (0.5-40 Hz), Notch (50 Hz), Resample (250 Hz)
    raw_proc = preprocess_raw(
        raw_cropped,
        l_freq=0.5,
        h_freq=40.0,
        notch_freq=50.0,
        target_sfreq=250.0,
        copy=True,
    )
    assert raw_proc.info["sfreq"] == 250.0

    # 3. Extract epochs: [-0.1s, 0.8s] = 0.9s = 225 samples
    epochs_data, retained_df = extract_epochs_from_events(
        raw_proc,
        df_trials,
        epoch_start=-0.1,
        epoch_end=0.8,
        baseline_start=-0.1,
        baseline_end=0.0,
        normalization="trial_zscore",
        expected_channels=63,
    )

    assert len(epochs_data) == 10
    assert len(retained_df) == 10
    # Shape must be exactly (10, 63, 225)
    assert epochs_data.shape == (10, 63, 225)
    assert not np.isnan(epochs_data).any(), "Found NaNs in epoched EEG data"
    assert not np.isinf(epochs_data).any(), "Found Infs in epoched EEG data"

    # Verify trial-wise z-score normalization (mean ~0, std ~1 per channel)
    mean_per_ch = np.mean(epochs_data, axis=-1)
    std_per_ch = np.std(epochs_data, axis=-1)
    np.testing.assert_allclose(mean_per_ch, 0.0, atol=1e-5)
    np.testing.assert_allclose(std_per_ch, 1.0, atol=1e-4)

    # 4. PyTorch Dataset & DataLoader
    dataset = ThingsEEGDataset(epochs_data, retained_df)
    assert len(dataset) == 10

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))

    assert "eeg" in batch
    assert "stimulus_id" in batch
    assert batch["eeg"].shape == (4, 63, 225)
    assert len(batch["stimulus_id"]) == 4
