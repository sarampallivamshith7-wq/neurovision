"""
Synthetic paired EEG / image-feature data with a *learnable* EEG<->image relationship.

Used by the test-suite and ``--synthetic`` runs to validate the whole alignment pipeline
(model, InfoNCE, evaluation, checkpointing) on machines without THINGS-EEG2. Nothing about
these numbers says anything about real-brain performance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.vision.cache_embeddings import ImageFeatureCache


@dataclass
class SyntheticData:
    train_eeg: np.ndarray
    train_meta: pd.DataFrame
    test_eeg: np.ndarray
    test_meta: pd.DataFrame
    cache: ImageFeatureCache
    num_subjects: int


def _meta(ids, reps, subject_of_trial, offset=0):
    rows = []
    for i, sid in enumerate(ids):
        for r in range(reps):
            rows.append({"stimulus_id": sid, "tot_img_number": i + 1, "category_name": f"cat{i % 10}"})
    df = pd.DataFrame(rows)
    df["trial_idx"] = np.arange(len(df)) + offset
    df["subject_idx"] = subject_of_trial(len(df))
    return df


def make_synthetic_alignment_data(
    n_train_images: int = 120,
    n_test_images: int = 20,
    reps_train: int = 3,
    reps_test: int = 6,
    n_channels: int = 63,
    n_samples: int = 225,
    dino_dim: int = 384,
    clip_dim: int = 512,
    n_subjects: int = 1,
    noise: float = 1.0,
    signal: float = 1.0,
    subject_strength: float = 0.5,
    seed: int = 0,
) -> SyntheticData:
    """EEG = signal * pattern(image features) + subject_pattern + noise, then trial z-scored."""
    rng = np.random.default_rng(seed)
    n_images = n_train_images + n_test_images
    ids = [f"stim-train{i + 1:05d}" for i in range(n_train_images)] + [
        f"stim-test{i + 1:03d}" for i in range(n_test_images)
    ]

    # Shared low-dimensional "semantic code" per image. Image features and the EEG pattern are
    # both linear read-outs of it, so the EEG<->image mapping is genuinely learnable and
    # generalises to unseen images (as opposed to a random, unrecoverable projection).
    code_dim = 16
    code = rng.normal(size=(n_images, code_dim))
    dino = (code @ rng.normal(size=(code_dim, dino_dim)) + 0.3 * rng.normal(size=(n_images, dino_dim))).astype(np.float32)
    clip = (code @ rng.normal(size=(code_dim, clip_dim)) + 0.3 * rng.normal(size=(n_images, clip_dim))).astype(np.float32)
    decode = rng.normal(size=(code_dim, n_channels * n_samples)) / np.sqrt(code_dim)
    patterns = (code @ decode).reshape(n_images, n_channels, n_samples)
    subj_patterns = rng.normal(size=(max(n_subjects, 1), n_channels, n_samples))

    def build(image_ids_slice, reps):
        sel = list(image_ids_slice)
        n = len(sel) * reps
        subj = rng.integers(0, n_subjects, size=n) if n_subjects > 1 else np.zeros(n, dtype=int)
        eeg = np.empty((n, n_channels, n_samples), dtype=np.float32)
        t = 0
        for ii in sel:
            for _ in range(reps):
                x = signal * patterns[ii] + subject_strength * subj_patterns[subj[t]]
                x = x + noise * rng.normal(size=x.shape)
                x = (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-8)
                eeg[t] = x
                t += 1
        return eeg, subj

    train_eeg, train_subj = build(range(n_train_images), reps_train)
    test_eeg, test_subj = build(range(n_train_images, n_images), reps_test)

    train_meta = _meta(ids[:n_train_images], reps_train, lambda n: train_subj)
    test_meta = _meta(ids[n_train_images:], reps_test, lambda n: test_subj)
    cache = ImageFeatureCache.from_arrays(ids, dino, clip)
    return SyntheticData(train_eeg, train_meta, test_eeg, test_meta, cache, max(n_subjects, 1))
