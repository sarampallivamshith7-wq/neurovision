"""
Dataset-free tests of preprocessing / epoching / pairing using synthetic MNE recordings.

The key check is *timing*: a known transient injected at (onset + 0.3 s) must land at the
right sample of the 250 Hz epoch, i.e. filtering + resampling must not shift the signal
relative to the stimulus and trials must not be mis-paired.
"""

import mne
import numpy as np
import pandas as pd
import pytest

from src.data.eeg_dataset import ConcatArray, ThingsEEGDataset
from src.data.preprocessing import (
    EXPECTED_63_CHANNELS,
    canonicalize_channels,
    extract_epochs_from_events,
    preprocess_raw,
)

SFREQ = 1000.0


def make_raw(duration=30.0, bumps=(), seed=0):
    """63-ch noise recording; ``bumps`` = [(time_sec, channel_idx)] smooth 40 uV transients."""
    rng = np.random.default_rng(seed)
    n = int(duration * SFREQ)
    data = rng.normal(scale=2e-6, size=(63, n))
    t = np.arange(n) / SFREQ
    for t0, ch in bumps:
        data[ch] += 40e-6 * np.exp(-0.5 * ((t - t0) / 0.03) ** 2)
    info = mne.create_info(EXPECTED_63_CHANNELS, SFREQ, "eeg")
    return mne.io.RawArray(data, info, verbose=False)


def make_trials(onsets):
    return pd.DataFrame({
        "trial_idx": np.arange(len(onsets)),
        "onset": onsets,
        "stimulus_id": [f"stim-train{i + 1:05d}" for i in range(len(onsets))],
        "tot_img_number": np.arange(len(onsets)) + 1,
        "category_name": ["c"] * len(onsets),
    })


@pytest.fixture(scope="module")
def processed():
    onsets = [5.0, 8.0, 11.0, 14.0]
    raw = make_raw(bumps=[(o + 0.3, 3) for o in onsets])  # transient 300 ms after each onset, channel 3
    proc = preprocess_raw(raw, 0.5, 40.0, 50.0, 250.0)
    return raw, proc, make_trials(onsets)


def test_resampled_to_250hz(processed):
    _, proc, _ = processed
    assert proc.info["sfreq"] == 250.0
    assert proc.n_times == 30 * 250


def test_epoch_shape_and_zscore(processed):
    _, proc, trials = processed
    eeg, kept = extract_epochs_from_events(proc, trials)
    assert eeg.shape == (4, 63, 225) and eeg.dtype == np.float32
    assert len(kept) == 4 and np.isfinite(eeg).all()
    np.testing.assert_allclose(eeg.mean(-1), 0.0, atol=1e-5)
    np.testing.assert_allclose(eeg.std(-1), 1.0, atol=1e-4)


def test_transient_lands_at_correct_sample(processed):
    """0.3 s after onset with epoch_start=-0.1 s -> sample (0.3+0.1)*250 = 100."""
    _, proc, trials = processed
    eeg, _ = extract_epochs_from_events(proc, trials)
    peaks = np.abs(eeg[:, 3, :]).argmax(axis=-1)
    assert np.all(np.abs(peaks - 100) <= 2), f"peaks at {peaks}, expected ~100 (timing shift!)"


def test_pairing_preserved_and_deterministic(processed):
    _, proc, trials = processed
    eeg1, kept1 = extract_epochs_from_events(proc, trials)
    eeg2, kept2 = extract_epochs_from_events(proc, trials)
    np.testing.assert_array_equal(eeg1, eeg2)
    assert kept1["stimulus_id"].tolist() == trials["stimulus_id"].tolist()
    assert kept1["epoch_idx"].tolist() == [0, 1, 2, 3]


def test_out_of_bounds_trials_dropped_and_metadata_stays_aligned():
    raw = make_raw(duration=30.0, bumps=[(10.3, 5), (20.3, 5)])
    proc = preprocess_raw(raw, 0.5, 40.0, 50.0, 250.0)
    trials = make_trials([0.05, 10.0, 20.0, 29.9])  # first starts before t=0, last runs past the end
    eeg, kept = extract_epochs_from_events(proc, trials)
    assert len(eeg) == len(kept) == 2
    assert kept["stimulus_id"].tolist() == ["stim-train00002", "stim-train00003"]  # correct rows survive
    # the surviving epochs really are the ones centred on the injected transients (ch 5)
    assert np.all(np.abs(np.abs(eeg[:, 5, :]).argmax(-1) - 100) <= 2)


def test_wrong_channel_count_raises():
    raw = make_raw().pick(list(range(60)))
    proc = raw.copy().resample(250.0, verbose=False)
    with pytest.raises(ValueError):
        extract_epochs_from_events(proc, make_trials([5.0]))


def test_missing_expected_channel_raises():
    raw = make_raw()
    raw.rename_channels({"Cz": "Cz_renamed"})
    with pytest.raises(ValueError):
        preprocess_raw(raw)


def test_channel_order_is_canonicalised_so_graph_positions_stay_valid():
    """A recording with shuffled channel order must come out in canonical order, data following names."""
    cz = EXPECTED_63_CHANNELS.index("Cz")
    raw = make_raw(bumps=[(10.3, cz)])
    perm = list(np.random.default_rng(1).permutation(63))
    shuffled = raw.copy().reorder_channels([EXPECTED_63_CHANNELS[i] for i in perm])
    assert shuffled.ch_names != EXPECTED_63_CHANNELS
    proc = preprocess_raw(shuffled, 0.5, 40.0, 50.0, 250.0)
    assert proc.ch_names == EXPECTED_63_CHANNELS
    eeg, _ = extract_epochs_from_events(proc, make_trials([10.0]))
    assert np.abs(eeg[0, cz]).argmax() in range(98, 103)      # transient still on Cz
    # ...and NOT on the other channels (z-scoring equalises amplitudes, so compare peak timing)
    others = np.delete(np.abs(eeg[0]).argmax(-1), cz)
    assert np.mean(np.isin(others, range(98, 103))) < 0.2


def test_channel_names_matched_case_insensitively():
    raw = make_raw()
    raw.rename_channels({"Fp1": "FP1", "Cz": "CZ"})
    assert canonicalize_channels(raw).ch_names == EXPECTED_63_CHANNELS


# ---- Dataset class ---------------------------------------------------------------------
def _meta(n):
    return pd.DataFrame({"stimulus_id": [f"s{i}" for i in range(n)], "tot_img_number": range(n), "category_name": "c"})


def test_dataset_length_mismatch_raises():
    with pytest.raises(ValueError):
        ThingsEEGDataset(np.zeros((3, 63, 225), np.float32), _meta(4))


def test_dataset_reads_memmap_lazily_and_defaults_subject(tmp_path):
    arr = np.random.default_rng(0).normal(size=(6, 63, 225)).astype(np.float32)
    np.save(tmp_path / "x.npy", arr)
    mm = np.load(tmp_path / "x.npy", mmap_mode="r")
    ds = ThingsEEGDataset(mm, _meta(6))
    item = ds[4]
    np.testing.assert_array_equal(item["eeg"].numpy(), arr[4])
    assert item["subject_idx"] == 0 and item["stimulus_id"] == "s4"
    assert item["eeg"].flags.writeable if hasattr(item["eeg"], "flags") else True


def test_concat_array_indexing():
    a, b = np.arange(6).reshape(3, 2, 1), np.arange(100, 108).reshape(4, 2, 1)
    c = ConcatArray([a, b])
    assert len(c) == 7 and c.shape == (7, 2, 1)
    np.testing.assert_array_equal(c[2], a[2])
    np.testing.assert_array_equal(c[3], b[0])
    np.testing.assert_array_equal(c[-1], b[-1])
    with pytest.raises(IndexError):
        c[7]
