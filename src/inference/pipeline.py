"""NeuroVision inference pipeline shared by the upload tab and the benchmark tab.

    EEG epoch (63, 225) -> MVNN whitening (session matrix) -> model.encode_eeg -> L2 norm
    -> cosine vs. gallery embeddings (model.encode_image of artifacts/gallery_features.npz)

Artifacts in artifacts/ are only ever READ here, never written.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.models.aligner import build_model

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
CKPT_PATH = ARTIFACTS / "best.pt"
MVNN_PATH = ARTIFACTS / "mvnn_whitening.npz"
GALLERY_PATH = ARTIFACTS / "gallery_features.npz"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ArtifactError(RuntimeError):
    """A required artifact is missing or unreadable."""


def _require(path: Path) -> Path:
    if not path.exists():
        raise ArtifactError(f"Missing artifact: {path.relative_to(ROOT)}")
    return path


def load_model():
    ckpt = torch.load(_require(CKPT_PATH), map_location=DEVICE, weights_only=False)
    model = build_model(ckpt["config"], num_subjects=ckpt.get("num_subjects", 1))
    model.load_state_dict(ckpt["model"])
    return model.to(DEVICE).eval(), ckpt.get("epoch"), ckpt["config"]


def load_mvnn():
    z = np.load(_require(MVNN_PATH))
    W = z["W"].astype(np.float32)
    sessions = [str(s) for s in z["sessions"]]
    return W, {s: k for k, s in enumerate(sessions)}


def concept_from_filename(name: str) -> str:
    """'golf_cart_06s.jpg' -> 'golf cart' (THINGS test-image naming); fallback when no mapping CSV."""
    stem = Path(name).stem
    stem = re.sub(r"_\d+[a-z]?$", "", stem)
    return stem.replace("_", " ")


@dataclass
class Gallery:
    names: list
    z: torch.Tensor          # (N, D) L2-normalised image embeddings

    def concept(self, i: int) -> str:
        return concept_from_filename(self.names[i])


def load_gallery(model) -> Gallery:
    zf = np.load(_require(GALLERY_PATH), allow_pickle=False)
    names = [str(s) for s in zf["names"]]
    with torch.no_grad():
        z_img = model.encode_image(torch.from_numpy(zf["dino"]).to(DEVICE),
                                   torch.from_numpy(zf["clip"]).to(DEVICE))
    return Gallery(names, F.normalize(z_img.float(), dim=-1))


def whiten(x: np.ndarray, W: np.ndarray, session_index) -> np.ndarray:
    """Same operation for both paths: W[session] @ x.

    x: (63, T) with an int session_index, or (n, 63, T) with one session index per trial.
    """
    return np.matmul(W[session_index], x)


@torch.no_grad()
def retrieve(eeg_input: np.ndarray, z_gallery: torch.Tensor, model) -> np.ndarray:
    """Whitened (63, T) EEG -> cosine similarity to every gallery image."""
    x = torch.from_numpy(np.ascontiguousarray(eeg_input, dtype=np.float32))[None].to(DEVICE)
    z = F.normalize(model.encode_eeg(x).float(), dim=-1)
    return (z @ z_gallery.t()).squeeze(0).cpu().numpy()


@torch.no_grad()
def embed_eeg(eeg_input: np.ndarray, model) -> np.ndarray:
    x = torch.from_numpy(np.ascontiguousarray(eeg_input, dtype=np.float32))[None].to(DEVICE)
    return F.normalize(model.encode_eeg(x).float(), dim=-1).squeeze(0).cpu().numpy()


def predict_from_eeg(epoch: np.ndarray, model, gallery: Gallery, W: np.ndarray, session_index: int):
    """Full upload path. epoch: validated (63, 225). Returns (scores, order, whitened, embedding)."""
    whitened = whiten(epoch, W, session_index)
    scores = retrieve(whitened, gallery.z, model)
    return scores, np.argsort(-scores), whitened
