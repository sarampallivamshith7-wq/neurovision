"""
Integration test of the *real-data* code path on a miniature fake BIDS dataset.

Exercises the actual scripts and loaders end to end:
    scripts/build_epochs.py  ->  processed .npy/.csv  ->  memmap loading  ->  training
    ->  test evaluation  ->  scripts/evaluate.py (FAISS top-K)

Each image n gets a Gaussian transient on channel 0 at latency 0.15 + 0.02 * (n % 20) s after
onset. That fingerprint lets us verify that every saved epoch is paired with the right image.
"""

import subprocess
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image

pytest.importorskip("edfio", reason="edfio is required to write synthetic BDF files")

from src.data.event_parser import StimuliResolver
from src.data.preprocessing import EXPECTED_63_CHANNELS
from src.training.train_alignment import load_checkpoint, load_processed_split, run_training
from src.utils.config import load_config
from src.vision.cache_embeddings import cache_image_features

ROOT = Path(__file__).resolve().parents[1]
N_TRAIN, N_TEST, TEST_REPS, SOA = 30, 20, 2, 1.0


def latency(n):
    return 0.15 + 0.02 * (n % 20)


class FakeExtractor:
    def __init__(self, dim): self.dim = dim

    def extract_from_pil_batch(self, images):
        return torch.stack([torch.tensor(np.asarray(i).reshape(-1, 3).mean(0), dtype=torch.float32).repeat(self.dim // 3 + 1)[: self.dim] for i in images])


def write_recording(path, onsets, numbers):
    rng = np.random.default_rng(0)
    n = int((onsets[-1] + 3) * 1000)
    data = rng.normal(scale=1e-6, size=(63, n))
    t = np.arange(n) / 1000.0
    for o, k in zip(onsets, numbers):
        data[0] += 5e-5 * np.exp(-0.5 * ((t - (o + latency(k))) / 0.015) ** 2)
    raw = mne.io.RawArray(data, mne.create_info(EXPECTED_63_CHANNELS, 1000.0, "eeg"), verbose=False)
    mne.export.export_raw(str(path), raw, fmt="bdf", overwrite=True, verbose=False)


def write_events(path, onsets, numbers):
    pd.DataFrame({
        "onset": onsets, "sample": (np.array(onsets) * 1000).astype(int), "duration": 0.1,
        "trial_type": "image", "tot_img_number": numbers, "img_category": [k % 10 + 1 for k in numbers],
        "category_name": [f"cat{k % 10}" for k in numbers],
    }).to_csv(path, sep="\t", index=False)


@pytest.fixture(scope="module")
def fake_bids(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bids")
    bids = tmp / "data" / "sub-01"
    stim = bids / "stimuli"
    eeg = bids / "sub-01" / "ses-01" / "eeg"
    eeg.mkdir(parents=True)
    rng = np.random.default_rng(1)

    rows = []
    for split, count in (("train", N_TRAIN), ("test", N_TEST)):
        for n in range(1, count + 1):
            sid = f"stim-train{n:05d}" if split == "train" else f"stim-test{n:03d}"
            concept, fname = f"{n:05d}_thing{n}", f"thing{n}_01b.jpg"
            rel = f"{split}_images/{concept}/{fname}" if split == "test" else f"training_images/{concept}/{fname}"
            (stim / rel).parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16), tuple(int(v) for v in rng.integers(0, 255, 3))).save(stim / rel)
            rows.append({"stimulus_id": sid, "filename": rel,
                         "description": f'concept "{concept}"; original filename "{fname}".'})
    pd.DataFrame(rows).to_csv(stim / "stimuli.tsv", sep="\t", index=False)

    tr_nums = list(range(1, N_TRAIN + 1))
    tr_on = [2.0 + SOA * i for i in range(len(tr_nums))]
    write_recording(eeg / "sub-01_ses-01_task-train_run-01_eeg.bdf", tr_on, tr_nums)
    write_events(eeg / "sub-01_ses-01_task-train_run-01_events.tsv", tr_on, tr_nums)

    te_nums = [n for _ in range(TEST_REPS) for n in range(1, N_TEST + 1)]
    te_on = [2.0 + SOA * i for i in range(len(te_nums))]
    write_recording(eeg / "sub-01_ses-01_task-test_eeg.bdf", te_on, te_nums)
    write_events(eeg / "sub-01_ses-01_task-test_events.tsv", te_on, te_nums)

    cfg = load_config(ROOT / "configs" / "phase1.yaml")
    d = cfg["dataset"]
    d.update(bids_root=str(bids), stimuli_dir=str(stim), stimuli_tsv=str(stim / "stimuli.tsv"),
             training_zip=str(stim / "training_images.zip"), processed_dir=str(tmp / "processed"),
             embeddings_dir=str(tmp / "embeddings"))
    cfg["training"].update(epochs=2, batch_size=8, patience=5, device_auto=False, device="cpu",
                           checkpoint_dir=str(tmp / "ckpt"), run_name="itest", val_fraction=0.1)
    cfg["retrieval"]["results_dir"] = str(tmp / "res")
    cfg_path = tmp / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg.to_dict()))
    return {"tmp": tmp, "cfg": cfg, "cfg_path": cfg_path}


def run_script(*args):
    return subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True, timeout=300)


@pytest.mark.slow
def test_full_real_data_path(fake_bids):
    cfg, cfg_path, tmp = fake_bids["cfg"], fake_bids["cfg_path"], fake_bids["tmp"]

    # 1. build_epochs.py -> processed files
    proc = run_script("scripts/build_epochs.py", "--config", cfg_path)
    assert proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-1500:]
    train_eeg, train_meta, n_sub = load_processed_split(cfg, "train")
    test_eeg, test_meta, _ = load_processed_split(cfg, "test")
    assert isinstance(train_eeg, np.memmap) and n_sub == 1                     # lazily loaded, not read into RAM
    assert train_eeg.shape == (N_TRAIN, 63, 225) and test_eeg.shape == (N_TEST * TEST_REPS, 63, 225)
    assert len(train_meta) == N_TRAIN and len(test_meta) == N_TEST * TEST_REPS
    assert train_meta["stimulus_id"].iloc[0] == "stim-train00001" and test_meta["stimulus_id"].iloc[0] == "stim-test001"

    # 2. pairing: the latency fingerprint in each saved epoch must match the image it is paired with
    for eeg, meta in ((train_eeg, train_meta), (test_eeg, test_meta)):
        got = np.abs(np.asarray(eeg)[:, 0, :]).argmax(-1)
        want = np.round((np.array([latency(k) for k in meta["tot_img_number"]]) + 0.1) * 250).astype(int)
        assert np.all(np.abs(got - want) <= 2), "EEG epoch is paired with the wrong image!"

    # 3. feature cache (fake extractors) for every stimulus, then real (non-synthetic) training
    d = cfg["dataset"]
    res = StimuliResolver(d["stimuli_tsv"], d["stimuli_dir"], d["training_zip"])
    cache_image_features(res.df_stimuli["stimulus_id"].tolist(), res, output_dir=d["embeddings_dir"],
                         dino_extractor=FakeExtractor(384), clip_extractor=FakeExtractor(512))
    out = run_training(cfg, synthetic=False)
    assert out["test_metrics"]["n_images"] == N_TEST and out["test_metrics"]["n_trials"] == N_TEST * TEST_REPS
    assert (tmp / "res" / "itest" / "test_metrics.json").exists()
    model, _, _ = load_checkpoint(out["checkpoint"])
    assert model is not None

    # 4. evaluate.py incl. FAISS top-K
    ev = run_script("scripts/evaluate.py", "--checkpoint", out["checkpoint"], "--config", cfg_path, "--topk-demo", 2)
    assert ev.returncode == 0, ev.stdout[-1500:] + ev.stderr[-1500:]
    assert "e2i_recall@1" in ev.stdout and "FAISS backend" in ev.stdout and "top5=" in ev.stdout
