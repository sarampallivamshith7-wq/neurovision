"""
NeuroVision Phase 1 prototype: EEG -> image retrieval on the THINGS-EEG2 test set.

Run from the project root:
    streamlit run app.py

Expects:
    artifacts/best.pt, artifacts/mvnn_whitening.npz, artifacts/test_metrics_signal_averaged.json
    data/test_eeg.npy, data/test_trial_image_mapping.csv, data/test_images/
    src/  (models, vision) from the training notebook
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from src.models.aligner import build_model
from src.vision.clip_features import CLIPExtractor
from src.vision.dinov2_features import DINOv2Extractor

# ----------------------------------------------------------------- paths / constants
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"
CKPT_PATH = ARTIFACTS / "best.pt"
MVNN_PATH = ARTIFACTS / "mvnn_whitening.npz"
METRICS_PATH = ARTIFACTS / "test_metrics_signal_averaged.json"
GALLERY_CACHE = ARTIFACTS / "gallery_features.npz"   # created on first launch
EEG_PATH = DATA / "test_eeg.npy"
MAPPING_PATH = DATA / "test_trial_image_mapping.csv"
IMAGE_DIR = DATA / "test_images"

SFREQ = 250.0
T_START_MS = -100.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="NeuroVision", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
    .nv-badge {display:inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
               font-weight: 700; font-size: 0.85rem; color: white;}
    .nv-ok {background: #16a34a;} .nv-bad {background: #dc2626;} .nv-rank {background: #4b5563;}
    .nv-true {background: #2563eb;}
    .nv-sub {opacity: 0.75; font-size: 0.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------- cached loaders
@st.cache_resource(show_spinner="Loading EEG data and mapping...")
def load_data():
    eeg = np.load(EEG_PATH, mmap_mode="r")
    mapping = pd.read_csv(MAPPING_PATH)
    required = {"session", "image_filename", "image_concept"}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"{MAPPING_PATH.name} is missing columns: {missing}")
    if len(mapping) != len(eeg):
        raise ValueError(f"Row mismatch: {len(mapping)} mapping rows vs {len(eeg)} EEG trials")
    mapping["image_filename"] = mapping["image_filename"].astype(str)
    mapping["session"] = mapping["session"].astype(str)

    image_lookup = {
        p.name: p for p in IMAGE_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    unique_mapped = list(dict.fromkeys(mapping["image_filename"]))
    gallery = [f for f in unique_mapped if f in image_lookup]
    if len(gallery) != len(unique_mapped):
        st.warning(f"{len(unique_mapped) - len(gallery)} mapped test images are missing on disk "
                   f"and were left out of the gallery.")

    # row positions of every repeat, per gallery image (row i of the CSV == eeg[i])
    reps = mapping.groupby("image_filename").indices
    reps = {f: np.sort(reps[f]) for f in gallery}
    concept = mapping.drop_duplicates("image_filename").set_index("image_filename")["image_concept"].to_dict()
    return eeg, mapping, image_lookup, gallery, reps, concept


@st.cache_resource(show_spinner="Loading whitening matrices...")
def load_mvnn():
    z = np.load(MVNN_PATH)
    W = z["W"].astype(np.float32)
    sessions = [str(s) for s in z["sessions"]]
    return W, {s: k for k, s in enumerate(sessions)}


@st.cache_resource(show_spinner="Loading NeuroVision model...")
def load_model():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model = build_model(ckpt["config"], num_subjects=ckpt.get("num_subjects", 1))
    model.load_state_dict(ckpt["model"])
    return model.to(DEVICE).eval(), ckpt.get("epoch"), ckpt["config"]


@st.cache_resource(show_spinner="Embedding the 200-image gallery (first launch only)...")
def load_gallery_embeddings(gallery: tuple[str, ...]):
    model, _, cfg = load_model()
    image_lookup = load_data()[2]

    feats = None
    if GALLERY_CACHE.exists():
        z = np.load(GALLERY_CACHE, allow_pickle=False)
        if [str(s) for s in z["names"]] == list(gallery):
            feats = (z["dino"], z["clip"])

    if feats is None:
        vis = cfg["vision"]
        dino_ex = DINOv2Extractor(vis.get("dinov2_model", "dinov2_vits14"), DEVICE)
        clip_ex = CLIPExtractor(vis.get("clip_model", "ViT-B-32"), vis.get("clip_pretrained", "openai"), DEVICE)
        dino_all, clip_all = [], []
        for s in range(0, len(gallery), 32):
            imgs = []
            for name in gallery[s:s + 32]:
                with Image.open(image_lookup[name]) as im:
                    imgs.append(im.convert("RGB").copy())
            dino_all.append(dino_ex.extract_from_pil_batch(imgs).numpy())
            clip_all.append(clip_ex.extract_from_pil_batch(imgs).numpy())
        feats = (np.concatenate(dino_all).astype(np.float32), np.concatenate(clip_all).astype(np.float32))
        np.savez(GALLERY_CACHE, names=np.array(gallery), dino=feats[0], clip=feats[1])
        del dino_ex, clip_ex

    with torch.no_grad():
        z_img = model.encode_image(torch.from_numpy(feats[0]).to(DEVICE), torch.from_numpy(feats[1]).to(DEVICE))
    return F.normalize(z_img.float(), dim=-1)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return {}
    data = json.loads(METRICS_PATH.read_text())
    return data.get("all_repeats", data)


@st.cache_data(show_spinner=False)
def load_thumb(path: str, size: int = 384):
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size))
    return im


# ----------------------------------------------------------------- core logic
def whiten(raw: np.ndarray, rows: np.ndarray, mapping: pd.DataFrame, W: np.ndarray, sess_idx: dict) -> np.ndarray:
    """raw: (n, 63, T) -> each trial whitened with the matrix of the session it was recorded in."""
    ks = np.array([sess_idx[s] for s in mapping["session"].values[rows]])
    return np.matmul(W[ks], raw)


@torch.no_grad()
def retrieve(eeg_input: np.ndarray, z_gallery: torch.Tensor, model) -> np.ndarray:
    x = torch.from_numpy(np.ascontiguousarray(eeg_input, dtype=np.float32))[None].to(DEVICE)
    z = F.normalize(model.encode_eeg(x).float(), dim=-1)
    return (z @ z_gallery.t()).squeeze(0).cpu().numpy()


def butterfly(x: np.ndarray, title: str) -> go.Figure:
    t = T_START_MS + np.arange(x.shape[1]) * 1000.0 / SFREQ
    fig = go.Figure()
    for ch in range(x.shape[0]):
        fig.add_trace(go.Scattergl(x=t, y=x[ch], mode="lines", line=dict(width=0.8),
                                   opacity=0.55, name=f"ch {ch + 1}", showlegend=False,
                                   hovertemplate=f"ch {ch + 1}<br>%{{x:.0f}} ms: %{{y:.2f}}<extra></extra>"))
    gfp = x.std(axis=0)
    fig.add_trace(go.Scatter(x=t, y=gfp, mode="lines", line=dict(width=2.5, color="black"),
                             name="Global field power"))
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="stimulus onset")
    fig.update_layout(title=title, height=380, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis_title="Time (ms)", yaxis_title="Amplitude (a.u.)",
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
    return fig


# ----------------------------------------------------------------- app
try:
    eeg, mapping, image_lookup, gallery, reps, concept = load_data()
    W, sess_idx = load_mvnn()
    model, best_epoch, _ = load_model()
    z_gallery = load_gallery_embeddings(tuple(gallery))
except FileNotFoundError as e:
    st.error(f"Missing file: {e}. Check the artifacts/ and data/ folders.")
    st.stop()

st.title("🧠 NeuroVision: EEG → Image Retrieval")
st.caption(f"THINGS-EEG2, subject 01 · EEG Conformer + DINOv2/CLIP · MVNN whitening · "
           f"{len(gallery)}-image zero-shot test gallery · checkpoint epoch {best_epoch}")

# KPI row
m = load_metrics()
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Recall@1", f"{m.get('recall@1', float('nan')):.1%}")
k2.metric("Recall@5", f"{m.get('recall@5', float('nan')):.1%}")
k3.metric("Recall@10", f"{m.get('recall@10', float('nan')):.1%}")
k4.metric("Median rank", f"{m.get('median_rank', float('nan')):.0f} / {len(gallery)}")
k5.metric("Chance Recall@1", f"{1 / len(gallery):.1%}")
st.caption("Benchmark metrics: EEG averaged over all ~81 test repeats per image before encoding.")

st.divider()

# ---- controls
with st.sidebar:
    st.header("Query")
    options = sorted(gallery, key=lambda f: (str(concept.get(f, "")).lower(), f))
    def _pick_random():
        st.session_state["target_sel"] = str(np.random.choice(gallery))

    st.button("🎲 Random image", width="stretch", on_click=_pick_random)
    target = st.selectbox(
        "Test image (concept)", options, key="target_sel",
        format_func=lambda f: f"{concept.get(f, '?')}  ·  {f}",
    )

    target_rows = reps[target]
    mode = st.radio("Mode", ["Single-Trial", "Signal-Averaged"], horizontal=True)

    if mode == "Single-Trial":
        rep_i = st.selectbox(
            f"Repetition ({len(target_rows)} recorded)", range(len(target_rows)),
            format_func=lambda i: f"#{i + 1}  ·  row {target_rows[i]}  ·  {mapping['session'].iloc[target_rows[i]]}",
        )
        rows = target_rows[[rep_i]]
    else:
        k = st.slider("Trials averaged (k)", 2, len(target_rows), min(20, len(target_rows)),
                      help=f"Averages k of the {len(target_rows)} repeats. Use the maximum to match "
                           f"the benchmark numbers above.")
        seed = st.session_state.setdefault("seed", 0)
        if st.button("🔀 Resample trials", width="stretch"):
            st.session_state["seed"] = seed = seed + 1
        rng = np.random.default_rng(abs(hash((target, k, seed))) % (2 ** 32))
        rows = np.sort(rng.choice(target_rows, size=k, replace=False))

    top_k = st.slider("Top-K candidates", 1, 10, 5)

# ---- inference
raw = np.asarray(eeg[rows], dtype=np.float32)                      # (n, 63, 225)
whitened = whiten(raw, rows, mapping, W, sess_idx)                 # per-trial session whitening
model_input = whitened.mean(axis=0)                                 # == notebook: whiten, then average
raw_avg = raw.mean(axis=0)

scores = retrieve(model_input, z_gallery, model)
order = np.argsort(-scores)
true_idx = gallery.index(target)
true_rank = int(np.where(order == true_idx)[0][0]) + 1
top1 = gallery[order[0]]
correct = top1 == target

# ---- ground truth vs top-1
left, right = st.columns(2, gap="large")
with left.container(border=True):
    st.subheader("Ground truth")
    st.image(load_thumb(str(image_lookup[target])), width="stretch")
    st.markdown(f"**{concept.get(target, '?')}**  \n<span class='nv-sub'>{target} · "
                f"{'1 trial' if len(rows) == 1 else f'{len(rows)} trials averaged'}</span>",
                unsafe_allow_html=True)
with right.container(border=True):
    st.subheader("Top-1 retrieved")
    st.image(load_thumb(str(image_lookup[top1])), width="stretch")
    badge = "<span class='nv-badge nv-ok'>✓ CORRECT</span>" if correct else "<span class='nv-badge nv-bad'>✗ WRONG</span>"
    st.markdown(f"{badge} &nbsp; **{concept.get(top1, '?')}** · cosine {scores[order[0]]:.3f}  \n"
                f"<span class='nv-sub'>True image ranked <b>{true_rank} / {len(gallery)}</b> "
                f"(cosine {scores[true_idx]:.3f})</span>", unsafe_allow_html=True)

# ---- top-k strip
st.subheader(f"Top-{top_k} candidates")
cols = st.columns(top_k)
for r, (col, gi) in enumerate(zip(cols, order[:top_k]), start=1):
    name = gallery[gi]
    with col:
        st.image(load_thumb(str(image_lookup[name]), 256), width="stretch")
        cls = "nv-true" if name == target else "nv-rank"
        tag = " · target" if name == target else ""
        st.markdown(f"<span class='nv-badge {cls}'>#{r}{tag}</span> <span class='nv-sub'>{scores[gi]:.3f}</span>"
                    f"<br><span class='nv-sub'>{concept.get(name, '?')}</span>", unsafe_allow_html=True)

# ---- EEG inspector
with st.expander("Electrophysiology & MVNN Diagnostics", expanded=False):
    view = st.radio("Signal", ["Raw scalp EEG", "Whitened EEG (W · x)"], horizontal=True)
    sig = raw_avg if view.startswith("Raw") else model_input
    label = "single trial" if len(rows) == 1 else f"average of {len(rows)} trials"
    st.plotly_chart(butterfly(sig, f"{view} · {label} · 63 channels"), width="stretch")

    sessions_used = mapping["session"].values[rows]
    c1, c2, c3 = st.columns(3)
    c1.metric("Trials used", len(rows))
    c2.metric("Sessions", ", ".join(sorted(set(sessions_used))))
    c3.metric("Signal std (raw → whitened)", f"{raw_avg.std():.3f} → {model_input.std():.3f}")
    st.caption("Whitening uses the session-specific noise covariance fitted on training trials only. "
               "Each trial is whitened with its own session's matrix before averaging, exactly as in "
               "the evaluation notebook. The black line is global field power (std across channels).")
