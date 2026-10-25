"""Tests for the EEG-upload path: validation (numpy only) and shared-pipeline inference (needs artifacts)."""
import io

import numpy as np
import pytest

from src.inference.eeg_input import EEGValidationError, load_npy_bytes, validate_eeg


def _npy(a):
    b = io.BytesIO(); np.save(b, a); return b.getvalue()


def test_valid_trial():
    r = validate_eeg(np.random.randn(63, 225).astype(np.float32))
    assert r.data.shape == (63, 225) and not r.transposed


def test_transposed_trial():
    x = np.random.randn(63, 225)
    r = validate_eeg(x.T)
    assert r.transposed and np.allclose(r.data, x, atol=1e-6)


def test_stack_is_averaged():
    r = validate_eeg(np.random.randn(5, 63, 225))
    assert r.data.shape == (63, 225) and r.n_trials_averaged == 5


@pytest.mark.parametrize("shape", [(64, 225), (63, 250), (63,), (2, 3, 63, 225), (0, 63, 225)])
def test_bad_shapes(shape):
    with pytest.raises(EEGValidationError, match="Expected EEG with 63 channels and 225 samples"):
        validate_eeg(np.zeros(shape))


def test_nan_inf_flat():
    x = np.random.randn(63, 225); x[0, 0] = np.nan
    with pytest.raises(EEGValidationError, match="non-finite"):
        validate_eeg(x)
    x[0, 0] = np.inf
    with pytest.raises(EEGValidationError, match="non-finite"):
        validate_eeg(x)
    with pytest.raises(EEGValidationError, match="flat"):
        validate_eeg(np.ones((63, 225)))


def test_bad_files():
    with pytest.raises(EEGValidationError, match="Unsupported file type"):
        load_npy_bytes(b"abc", "x.edf")
    with pytest.raises(EEGValidationError, match="valid .npy"):
        load_npy_bytes(b"not an npy", "x.npy")
    obj = io.BytesIO(); np.save(obj, np.array([{"a": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(EEGValidationError):
        load_npy_bytes(obj.getvalue(), "x.npy")       # pickles are refused
    assert load_npy_bytes(_npy(np.zeros((63, 225))), "x.npy").shape == (63, 225)


def test_upload_reaches_shared_pipeline():
    torch = pytest.importorskip("torch")
    from src.inference import pipeline as nv
    if not nv.CKPT_PATH.exists():
        pytest.skip("artifacts not present")
    model, _, _ = nv.load_model()
    W, sess = nv.load_mvnn()
    gal = nv.load_gallery(model)
    rng = np.random.default_rng(0)
    x = validate_eeg(rng.standard_normal((63, 225))).data
    scores, order, _ = nv.predict_from_eeg(x, model, gal, W, sess["ses-01"])
    assert scores.shape == (len(gal.names),) and np.isfinite(scores).all()
    assert (np.abs(scores) <= 1.0001).all()
    # identical to the benchmark-style call: whiten -> retrieve
    ref = nv.retrieve(nv.whiten(x, W, sess["ses-01"]), gal.z, model)
    np.testing.assert_allclose(scores, ref)
