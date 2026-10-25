"""End-to-end alignment training on synthetic data (no dataset required)."""

import copy
import json
import logging

import numpy as np
import pytest
import torch
from torch.utils.data import Subset

from src.data.eeg_dataset import ThingsEEGDataset
from src.data.synthetic import make_synthetic_alignment_data
from src.evaluation import evaluate_retrieval
from src.training.train_alignment import (
    load_checkpoint,
    run_training,
    split_by_stimulus,
    train_model,
)
from src.utils.config import load_config

logging.getLogger("neurovision.train").setLevel(logging.WARNING)


def make_cfg(tmp_path, encoder="eegnet", epochs=10, **train_kw):
    cfg = load_config("configs/phase1.yaml")
    cfg["eeg_model"]["encoder_type"] = encoder
    t = cfg["training"]
    t.update(epochs=epochs, patience=99, learning_rate=1e-3, batch_size=64,
             checkpoint_dir=str(tmp_path / "ckpt"), run_name="t", device_auto=False, device="cpu")
    t.update(train_kw)
    cfg["retrieval"]["results_dir"] = str(tmp_path / "res")
    return cfg


@pytest.mark.slow
def test_eegnet_learns_to_retrieve_unseen_images_and_checkpoint_roundtrips(tmp_path):
    cfg = make_cfg(tmp_path)
    res = run_training(cfg, synthetic=True)
    m = res["test_metrics"]
    # 20 unseen test images: chance R@1 = 5%
    assert m["e2i_recall@1"] >= 0.6, m
    assert m["e2i_recall@5"] >= 0.9 and m["i2e_recall@1"] >= 0.5

    # artefacts required by the README's reproducibility rules
    out = tmp_path / "ckpt" / "t"
    assert (out / "best.pt").exists() and (out / "config.yaml").exists() and (out / "history.csv").exists()
    saved = json.loads((tmp_path / "res" / "t" / "test_metrics.json").read_text())
    assert saved["metrics"]["e2i_recall@1"] == m["e2i_recall@1"]

    # a reloaded checkpoint reproduces the reported metrics exactly
    syn = make_synthetic_alignment_data(n_subjects=1, seed=int(cfg["seed"]))
    model, cfg2, ckpt = load_checkpoint(out / "best.pt")
    again = evaluate_retrieval(model, ThingsEEGDataset(syn.test_eeg, syn.test_meta), syn.cache)
    assert again["e2i_recall@1"] == pytest.approx(m["e2i_recall@1"])
    assert again["e2i_mrr"] == pytest.approx(m["e2i_mrr"], abs=1e-5)
    assert cfg2["eeg_model"]["encoder_type"] == "eegnet" and ckpt["epoch"] >= 1


@pytest.mark.slow
def test_negative_control_shuffled_pairings_stay_at_chance(tmp_path):
    """Break the EEG<->image pairing in training. If test retrieval still looked good, the pipeline leaks."""
    syn = make_synthetic_alignment_data(seed=2)
    rng = np.random.default_rng(0)
    train_ids = sorted(syn.train_meta["stimulus_id"].unique())
    perm = dict(zip(train_ids, rng.permutation(train_ids)))
    meta = syn.train_meta.copy()
    meta["stimulus_id"] = meta["stimulus_id"].map(perm)              # wrong image for every EEG trial

    cfg = make_cfg(tmp_path, epochs=10)
    full = ThingsEEGDataset(syn.train_eeg, meta)
    tr, _ = split_by_stimulus(meta["stimulus_id"].tolist(), 0.1, 2)
    r = train_model(cfg, Subset(full, tr.tolist()), None, syn.cache, 1, tmp_path / "ctrl")
    model, _, _ = load_checkpoint(r["checkpoint"])
    m = evaluate_retrieval(model, ThingsEEGDataset(syn.test_eeg, syn.test_meta), syn.cache)
    assert m["e2i_recall@1"] < 0.35, f"control reached {m['e2i_recall@1']:.2f} (chance 0.05) -> leakage?"


@pytest.mark.parametrize("encoder", ["conformer", "gnn_conformer"])
def test_advanced_encoders_train_a_few_steps_without_nan(tmp_path, encoder):
    syn = make_synthetic_alignment_data(n_train_images=24, n_test_images=8, reps_train=2, reps_test=2, seed=0)
    cfg = make_cfg(tmp_path, encoder=encoder, epochs=2, batch_size=16)
    full = ThingsEEGDataset(syn.train_eeg, syn.train_meta)
    tr, va = split_by_stimulus(syn.train_meta["stimulus_id"].tolist(), 0.2, 0)
    r = train_model(cfg, Subset(full, tr.tolist()), Subset(full, va.tolist()), syn.cache, 1, tmp_path / encoder)
    assert len(r["history"]) == 2
    assert all(np.isfinite(h["train_loss"]) for h in r["history"])
    assert r["history"][-1]["train_loss"] < r["history"][0]["train_loss"] + 0.5   # not diverging
    assert r["checkpoint"].exists()


def test_subject_adversarial_training_runs_and_logs_adversary(tmp_path):
    syn = make_synthetic_alignment_data(n_train_images=24, n_test_images=8, reps_train=2, reps_test=2, n_subjects=2, seed=0)
    assert set(syn.train_meta["subject_idx"]) == {0, 1}
    cfg = make_cfg(tmp_path, epochs=2, batch_size=16)
    cfg["subject_adversarial"]["enabled"] = True
    full = ThingsEEGDataset(syn.train_eeg, syn.train_meta)
    tr, va = split_by_stimulus(syn.train_meta["stimulus_id"].tolist(), 0.2, 0)
    r = train_model(cfg, Subset(full, tr.tolist()), Subset(full, va.tolist()), syn.cache, syn.num_subjects, tmp_path / "adv")
    h = r["history"][-1]
    assert np.isfinite(h["train_adv"]) and h["train_adv"] > 0 and 0.0 <= h["train_subj_acc"] <= 1.0
    model, _, _ = load_checkpoint(r["checkpoint"])
    assert model.subject_classifier is not None


def test_adversarial_without_multiple_subjects_fails_loudly(tmp_path):
    syn = make_synthetic_alignment_data(n_train_images=12, n_test_images=4, reps_train=2, reps_test=2)
    cfg = make_cfg(tmp_path, epochs=1, batch_size=8)
    cfg["subject_adversarial"]["enabled"] = True
    full = ThingsEEGDataset(syn.train_eeg, syn.train_meta)
    with pytest.raises(ValueError, match="one subject"):
        train_model(cfg, full, None, syn.cache, 1, tmp_path / "x")


def test_training_is_reproducible_given_the_seed(tmp_path):
    syn = make_synthetic_alignment_data(n_train_images=20, n_test_images=4, reps_train=2, reps_test=2, seed=5)
    losses = []
    for i in range(2):
        cfg = make_cfg(tmp_path, epochs=2, batch_size=16)
        full = ThingsEEGDataset(syn.train_eeg, syn.train_meta)
        r = train_model(cfg, full, None, syn.cache, 1, tmp_path / f"rep{i}")
        losses.append([h["train_loss"] for h in r["history"]])
    np.testing.assert_allclose(losses[0], losses[1], rtol=1e-5)
