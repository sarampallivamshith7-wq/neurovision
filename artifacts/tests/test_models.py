"""Model-component tests: shapes, graph construction, batch independence, adversary gradients."""

import copy

import numpy as np
import pytest
import torch

from src.data.preprocessing import EXPECTED_63_CHANNELS
from src.models import (
    ENCODER_TYPES,
    EEGProjectionHead,
    GNNConformerEncoder,
    GradientReversal,
    SubjectClassifier,
    build_eeg_encoder,
    build_electrode_graph,
    build_model,
    dann_lambda,
)
from src.models.spatial_gnn import SpatialGraphEncoder, get_electrode_positions
from src.utils.config import load_config

torch.manual_seed(0)


@pytest.fixture(scope="module")
def cfg():
    return load_config("configs/phase1.yaml")


def with_encoder(cfg, name):
    c = copy.deepcopy(cfg)
    c["eeg_model"]["encoder_type"] = name
    return c


# ---- encoders -------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ENCODER_TYPES)
def test_encoder_output_shape_and_finite(cfg, name):
    enc = build_eeg_encoder(with_encoder(cfg, name)["eeg_model"])
    x = torch.randn(5, 63, 225)
    y = enc(x)
    assert y.shape == (5, cfg["eeg_model"]["eeg_rep_dim"]) == (5, enc.rep_dim)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("name", ENCODER_TYPES)
def test_encoder_accepts_4d_input_and_backprops(cfg, name):
    enc = build_eeg_encoder(with_encoder(cfg, name)["eeg_model"])
    x = torch.randn(3, 1, 63, 225)
    enc(x).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in enc.parameters() if p.requires_grad)


@pytest.mark.parametrize("name", ["conformer", "gnn_conformer"])
def test_samples_in_a_batch_do_not_influence_each_other(cfg, name):
    """Eval-mode output for a sample must not depend on its batch-mates (catches batched-graph offset bugs)."""
    enc = build_eeg_encoder(with_encoder(cfg, name)["eeg_model"]).eval()
    x = torch.randn(4, 63, 225)
    with torch.no_grad():
        together = enc(x)
        alone = torch.cat([enc(x[i : i + 1]) for i in range(4)])
    torch.testing.assert_close(together, alone, atol=1e-4, rtol=1e-4)


def test_unknown_encoder_raises(cfg):
    with pytest.raises(ValueError):
        build_eeg_encoder(with_encoder(cfg, "transformer_xl")["eeg_model"])


# ---- electrode graph ------------------------------------------------------------------------
def test_electrode_graph_properties():
    pos = get_electrode_positions(EXPECTED_63_CHANNELS)
    assert pos.shape == (63, 3) and np.isfinite(pos).all()
    ei = build_electrode_graph(pos, k=6)
    edges = set(map(tuple, ei.t().tolist()))
    assert all((j, i) in edges for i, j in edges), "graph must be undirected"
    assert all(i != j for i, j in edges), "no self loops"
    assert torch.bincount(ei[0], minlength=63).min() >= 6
    assert set(ei.flatten().tolist()) == set(range(63)), "every electrode is connected"


def test_electrode_graph_is_spatially_meaningful():
    """Neighbours must follow scalp geometry: Oz connects to occipital neighbours, not frontal ones."""
    ei = build_electrode_graph(get_electrode_positions(EXPECTED_63_CHANNELS), k=6)
    nbrs = lambda name: {EXPECTED_63_CHANNELS[j] for j in ei[1][ei[0] == EXPECTED_63_CHANNELS.index(name)].tolist()}
    assert {"O1", "O2", "POz"} <= nbrs("Oz")
    assert not (nbrs("Oz") & {"Fp1", "Fp2", "AFz", "Fz"})
    assert {"Cz"} & nbrs("C1") and {"Cz"} & nbrs("C2")


def test_electrode_positions_from_bids_tsv_and_fallback(tmp_path):
    import pandas as pd

    names = EXPECTED_63_CHANNELS
    df = pd.DataFrame({"name": names, "x": np.arange(63.0), "y": 0.0, "z": 0.0})
    f = tmp_path / "electrodes.tsv"
    df.to_csv(f, sep="\t", index=False)
    np.testing.assert_allclose(get_electrode_positions(names, f)[:, 0], np.arange(63.0))
    df.iloc[5, 1] = np.nan  # incomplete file -> falls back to the standard montage
    df.to_csv(f, sep="\t", index=False)
    assert np.isfinite(get_electrode_positions(names, f)).all()
    with pytest.raises(ValueError):
        get_electrode_positions(["NotAnElectrode"])


@pytest.mark.parametrize("conv", ["gatv2", "gat", "gcn"])
def test_graph_encoder_variants_swap_via_config(conv):
    ei = build_electrode_graph(get_electrode_positions(EXPECTED_63_CHANNELS), k=4)
    g = SpatialGraphEncoder(ei, 63, dim=16, num_layers=2, heads=4, conv_type=conv)
    x = torch.randn(2, 63, 7, 16)
    assert g(x).shape == x.shape
    with pytest.raises(ValueError):
        SpatialGraphEncoder(ei, 63, conv_type="nope")


def test_gnn_actually_mixes_neighbouring_electrodes_only():
    """A perturbation on one electrode must change its graph neighbours' features but not distant ones."""
    ei = build_electrode_graph(get_electrode_positions(EXPECTED_63_CHANNELS), k=4)
    g = SpatialGraphEncoder(ei, 63, dim=16, num_layers=1, heads=4, dropout=0.0).eval()
    x = torch.randn(1, 63, 3, 16)
    x2 = x.clone(); x2[0, 10] += 5.0
    with torch.no_grad():
        d = (g(x2) - g(x)).abs().sum(dim=(0, 2, 3))
    neigh = set(ei[1][ei[0] == 10].tolist()) | {10}
    assert all(d[i] > 1e-6 for i in neigh)
    assert all(d[i] < 1e-6 for i in range(63) if i not in neigh)


def test_gnn_requires_names_for_non_63_channels():
    with pytest.raises(ValueError):
        GNNConformerEncoder(in_channels=32)
    enc = GNNConformerEncoder(in_channels=8, in_samples=100, ch_names=EXPECTED_63_CHANNELS[:8], knn=3, pool_kernel=10, pool_stride=5)
    assert enc(torch.randn(2, 8, 100)).shape == (2, 256)


# ---- projection / adversary -----------------------------------------------------------------
def test_projection_head_unit_norm():
    z = EEGProjectionHead(256, 512)(torch.randn(6, 256))
    torch.testing.assert_close(z.norm(dim=-1), torch.ones(6), atol=1e-5, rtol=0)


def test_gradient_reversal_flips_and_scales_gradient():
    x = torch.ones(3, requires_grad=True)
    GradientReversal(0.5)(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.full((3,), -0.5))
    y = torch.ones(3, requires_grad=True)
    GradientReversal(0.5)(y).sum().detach()  # forward is identity
    assert torch.equal(GradientReversal(2.0)(y), y)


def test_dann_lambda_schedule():
    assert dann_lambda(0.0) == pytest.approx(0.0, abs=1e-9)
    assert dann_lambda(1.0) == pytest.approx(1.0, abs=1e-3)
    assert dann_lambda(0.5, max_lambda=0.3) < 0.3
    vals = [dann_lambda(p / 10) for p in range(11)]
    assert vals == sorted(vals)


def test_adversary_pushes_encoder_away_from_subject_information():
    """Classifier params descend the CE loss while the upstream encoder ascends it."""
    enc = torch.nn.Linear(4, 4, bias=False)
    clf = SubjectClassifier(4, num_subjects=2, dropout=0.0, lambd=1.0)
    x, y = torch.randn(16, 4), torch.randint(0, 2, (16,))
    loss = torch.nn.functional.cross_entropy(clf(enc(x)), y)
    loss.backward()
    g_rev = enc.weight.grad.clone()
    enc.zero_grad(); clf.zero_grad()
    clf.set_lambda(-1.0)  # lambda=-1 turns the reversal into an identity -> true gradient of the loss
    torch.nn.functional.cross_entropy(clf(enc(x)), y).backward()
    torch.testing.assert_close(g_rev, -enc.weight.grad, atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError):
        SubjectClassifier(4, num_subjects=1)


# ---- full model assembly --------------------------------------------------------------------
def test_build_model_forward_and_adversary_rules(cfg):
    m = build_model(cfg)
    out = m(torch.randn(4, 63, 225), torch.randn(4, 384), torch.randn(4, 512))
    assert out["z_eeg"].shape == out["z_img"].shape == (4, 512) and out["subject_logits"] is None

    adv = copy.deepcopy(cfg); adv["subject_adversarial"]["enabled"] = True
    with pytest.raises(ValueError, match="one subject"):
        build_model(adv, num_subjects=1)
    m2 = build_model(adv, num_subjects=3)
    assert m2(torch.randn(4, 63, 225), torch.randn(4, 384), torch.randn(4, 512))["subject_logits"].shape == (4, 3)

    bad = copy.deepcopy(cfg); bad["eeg_model"]["eeg_embed_dim"] = 256
    with pytest.raises(ValueError, match="must equal"):
        build_model(bad)
