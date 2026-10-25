"""InfoNCE, retrieval metrics, evaluation protocol and FAISS index tests."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from src.data.eeg_dataset import ThingsEEGDataset
from src.evaluation import ImageIndex, compute_ranks, evaluate_retrieval, retrieval_metrics
from src.training.losses import batch_retrieval_accuracy, info_nce_loss
from src.training.train_alignment import split_by_stimulus
from src.vision.cache_embeddings import ImageFeatureCache


# ---- InfoNCE --------------------------------------------------------------------------------
def test_infonce_perfect_alignment_near_zero_and_random_near_log_n():
    n = 32
    eye = torch.eye(n)
    assert info_nce_loss(eye, eye, temperature=0.05).item() < 1e-3
    torch.manual_seed(0)
    a, b = (F.normalize(torch.randn(n, 128), dim=-1) for _ in range(2))
    assert info_nce_loss(a, b, temperature=1.0).item() == pytest.approx(math.log(n), rel=0.15)


def test_infonce_symmetric_is_mean_of_both_directions():
    torch.manual_seed(1)
    a, b = (F.normalize(torch.randn(8, 16), dim=-1) for _ in range(2))
    e2i = info_nce_loss(a, b, symmetric=False)
    i2e = info_nce_loss(b, a, symmetric=False)
    assert info_nce_loss(a, b, symmetric=True).item() == pytest.approx(0.5 * (e2i + i2e).item(), rel=1e-5)


def test_infonce_masks_false_negatives_from_repeated_images():
    z = torch.eye(4)[[0, 0, 1, 2]]  # image 0 appears twice
    ids = ["a", "a", "b", "c"]
    unmasked = info_nce_loss(z, z, temperature=0.05)
    masked = info_nce_loss(z, z, temperature=0.05, stimulus_ids=ids)
    assert unmasked.item() > 0.3          # duplicates are wrongly penalised as negatives
    assert masked.item() < 1e-3           # ...and correctly ignored when ids are supplied


def test_infonce_gradients_flow_and_validate_inputs():
    a = F.normalize(torch.randn(6, 8), dim=-1).requires_grad_()
    b = F.normalize(torch.randn(6, 8), dim=-1)
    info_nce_loss(a, b).backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()
    with pytest.raises(ValueError):
        info_nce_loss(a, b[:, :4])
    with pytest.raises(ValueError):
        info_nce_loss(a, b, temperature=0)
    with pytest.raises(ValueError):
        info_nce_loss(a, b, stimulus_ids=["x"])


def test_batch_accuracy_counts_duplicate_images_as_correct():
    z_img = torch.eye(3)[[0, 0, 1]]
    z_eeg = z_img.clone()
    assert batch_retrieval_accuracy(z_eeg, z_img) < 1.0                       # naive: 2nd duplicate "wrong"
    assert batch_retrieval_accuracy(z_eeg, z_img, ["a", "a", "b"]) == 1.0     # id-aware: correct


# ---- ranking / metrics ----------------------------------------------------------------------
def test_ranks_hand_computed():
    sim = torch.tensor([[0.9, 0.1, 0.5],    # target 0 -> rank 1
                        [0.9, 0.1, 0.5],    # target 2 -> rank 2
                        [0.9, 0.1, 0.5]])   # target 1 -> rank 3
    ranks = compute_ranks(sim, torch.tensor([0, 2, 1]))
    assert ranks.tolist() == [1.0, 2.0, 3.0]
    m = retrieval_metrics(sim, torch.tensor([0, 2, 1]), ks=(1, 2, 3))
    assert m["recall@1"] == pytest.approx(1 / 3) and m["recall@2"] == pytest.approx(2 / 3) and m["recall@3"] == 1.0
    assert m["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 3) / 3)
    assert m["median_rank"] == 2.0 and m["chance_recall@1"] == pytest.approx(1 / 3)


def test_collapsed_model_cannot_score_perfect_recall():
    """All-identical embeddings tie everywhere; optimistic tie-breaking would report Recall@1 = 1.0."""
    sim = torch.ones(50, 50)
    m = retrieval_metrics(sim, torch.arange(50), ks=(1, 5))
    assert m["recall@1"] == 0.0
    assert m["median_rank"] == pytest.approx(25.5)


# ---- evaluation protocol --------------------------------------------------------------------
class OracleModel:
    """Embeds EEG and images identically per stimulus -> retrieval must be perfect."""

    def __init__(self, dim): self.dim = dim
    def eval(self): return self
    def encode_eeg(self, eeg): return F.normalize(eeg[:, 0, : self.dim], dim=-1)
    def encode_image(self, dino, clip): return F.normalize(dino[:, : self.dim], dim=-1)


def _oracle_setup(n_img=15, reps=4, dim=32, noise=0.0, seed=0):
    import pandas as pd

    rng = np.random.default_rng(seed)
    codes = rng.normal(size=(n_img, dim)).astype(np.float32)
    ids = [f"stim-test{i:03d}" for i in range(n_img)]
    eeg = np.zeros((n_img * reps, 2, 225), np.float32)
    rows = []
    for i in range(n_img):
        for r in range(reps):
            eeg[i * reps + r, 0, :dim] = codes[i] + noise * rng.normal(size=dim)
            rows.append({"stimulus_id": ids[i], "tot_img_number": i, "category_name": "c"})
    cache = ImageFeatureCache.from_arrays(ids, codes, np.zeros((n_img, 4), np.float32))
    return ThingsEEGDataset(eeg, pd.DataFrame(rows)), cache


def test_evaluate_retrieval_oracle_is_perfect_and_reports_all_groups():
    ds, cache = _oracle_setup()
    m = evaluate_retrieval(OracleModel(32), ds, cache, ks=(1, 5))
    for k in ("trial_e2i_recall@1", "e2i_recall@1", "i2e_recall@1", "e2i_mrr", "i2e_mrr"):
        assert m[k] == 1.0, k
    assert m["n_images"] == 15 and m["n_trials"] == 60
    assert m["e2i_chance_recall@1"] == pytest.approx(1 / 15)


def test_evaluate_retrieval_random_embeddings_are_near_chance():
    ds, cache = _oracle_setup(n_img=40, dim=32)

    class Random(OracleModel):
        def encode_eeg(self, eeg): return F.normalize(torch.randn(len(eeg), 32), dim=-1)

    torch.manual_seed(0)
    m = evaluate_retrieval(Random(32), ds, cache)
    assert m["e2i_recall@1"] < 0.25  # chance = 0.025


def test_repetition_averaging_beats_single_trials_under_noise():
    ds, cache = _oracle_setup(n_img=40, reps=16, dim=32, noise=2.0, seed=3)
    m = evaluate_retrieval(OracleModel(32), ds, cache)
    assert m["e2i_mrr"] > m["trial_e2i_mrr"]


# ---- splitting (leakage) --------------------------------------------------------------------
def test_split_by_stimulus_never_leaks_an_image_across_splits():
    ids = [f"img{i}" for i in range(50) for _ in range(4)]
    tr, va = split_by_stimulus(ids, 0.2, seed=0)
    tr_imgs, va_imgs = {ids[i] for i in tr}, {ids[i] for i in va}
    assert tr_imgs.isdisjoint(va_imgs)
    assert len(va_imgs) == 10 and len(tr) + len(va) == len(ids)
    assert all(ids[i] in va_imgs for i in va) and all(sum(x == v for x in (ids[i] for i in va)) == 4 for v in va_imgs)
    tr2, va2 = split_by_stimulus(ids, 0.2, seed=0)
    assert (va == va2).all()                                  # deterministic
    assert split_by_stimulus(ids, 0.0, 0)[1].size == 0        # no validation requested


# ---- FAISS index ----------------------------------------------------------------------------
def test_image_index_matches_bruteforce_and_returns_ids():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(200, 64)).astype(np.float32)
    ids = [f"s{i}" for i in range(200)]
    index = ImageIndex(emb, ids)
    q = emb[[3, 77, 150]] + 0.01 * rng.normal(size=(3, 64)).astype(np.float32)
    scores, top = index.search(q, k=5)
    assert [t[0] for t in top] == ["s3", "s77", "s150"]
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    en = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    expected = np.argsort(-(qn @ en.T), axis=1)[:, :5]
    assert [[ids[j] for j in row] for row in expected] == top
    assert np.all(np.diff(scores, axis=1) <= 1e-6)            # best first
    assert index.backend in ("faiss", "numpy")
    assert len(index.search(q, k=999)[1][0]) == 200            # k is clipped to gallery size
