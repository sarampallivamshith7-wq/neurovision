"""
NeuroVision: EEG-based visual concept retrieval (THINGS-EEG2 zero-shot gallery).

Run from the project root:
    streamlit run app.py

Tabs:
    "EEG Inference"  - upload a processed EEG epoch (.npy, 63 x 225 @ 250 Hz) and retrieve visual concepts
    "Evaluation"     - the original THINGS-EEG2 benchmark/demo (needs data/ folder)

Artifacts used (read-only): artifacts/best.pt, artifacts/mvnn_whitening.npz,
artifacts/gallery_features.npz, artifacts/test_metrics_signal_averaged.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.inference import pipeline as nv
from src.inference.eeg_input import (EEGValidationError, N_CHANNELS, N_SAMPLES, SFREQ, T_START_MS,
                                     load_npy_bytes, validate_eeg)

# ----------------------------------------------------------------- paths
ARTIFACTS = nv.ARTIFACTS
DATA = ROOT / "data"
METRICS_PATH = ARTIFACTS / "test_metrics_signal_averaged.json"
EEG_PATH = DATA / "test_eeg.npy"
MAPPING_PATH = DATA / "test_trial_image_mapping.csv"
IMAGE_DIR = DATA / "test_images"

st.set_page_config(page_title="NeuroVision", page_icon="🧠", layout="wide")
st.markdown(
    """
    <style>
    .nv-badge {display:inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
               font-weight: 700; font-size: 0.85rem; color: white;}
    .nv-ok {background: #16a34a;} .nv-bad {background: #dc2626;} .nv-rank {background: #4b5563;}
    .nv-true {background: #2563eb;}
    .nv-sub {opacity: 0.75; font-size: 0.85rem;}
    .nv-flow {text-align:center; opacity:0.8; font-size:0.9rem; padding:0.4rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------- cached loaders (shared pipeline)
@st.cache_resource(show_spinner="Loading NeuroVision model...")
def get_model():
    return nv.load_model()


@st.cache_resource(show_spinner="Loading whitening matrices...")
def get_mvnn():
    return nv.load_mvnn()


@st.cache_resource(show_spinner="Loading gallery embeddings...")
def get_gallery():
    model, _, _ = get_model()
    return nv.load_gallery(model)


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return {}
    data = json.loads(METRICS_PATH.read_text())
    return data.get("all_repeats", data)


@st.cache_resource(show_spinner=False)
def get_image_lookup():
    if not IMAGE_DIR.exists():
        return {}
    return {p.name: p for p in IMAGE_DIR.rglob("*")
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}}


@st.cache_data(show_spinner=False)
def load_thumb(path: str, size: int = 384):
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size))
    return im


@st.cache_resource(show_spinner="Loading EEG data and mapping...")
def load_benchmark_data():
    eeg = np.load(EEG_PATH, mmap_mode="r")
    mapping = pd.read_csv(MAPPING_PATH)
    missing = {"session", "image_filename", "image_concept"} - set(mapping.columns)
    if missing:
        raise ValueError(f"{MAPPING_PATH.name} is missing columns: {missing}")
    if len(mapping) != len(eeg):
        raise ValueError(f"Row mismatch: {len(mapping)} mapping rows vs {len(eeg)} EEG trials")
    mapping["image_filename"] = mapping["image_filename"].astype(str)
    mapping["session"] = mapping["session"].astype(str)
    gallery_names = list(dict.fromkeys(mapping["image_filename"]))
    reps = {f: np.sort(v) for f, v in mapping.groupby("image_filename").indices.items()}
    concept = mapping.drop_duplicates("image_filename").set_index("image_filename")["image_concept"].to_dict()
    return eeg, mapping, gallery_names, reps, concept


# ----------------------------------------------------------------- plots
def eeg_figure(x: np.ndarray, view: str, channel: int | None) -> go.Figure:
    t = T_START_MS + np.arange(x.shape[1]) * 1000.0 / SFREQ
    fig = go.Figure()
    if view == "Single channel":
        fig.add_trace(go.Scatter(x=t, y=x[channel], mode="lines", name=f"Channel {channel + 1}"))
    elif view == "All channels (butterfly)":
        for ch in range(x.shape[0]):
            fig.add_trace(go.Scattergl(x=t, y=x[ch], mode="lines", line=dict(width=0.8), opacity=0.5,
                                       showlegend=False, hovertemplate=f"ch {ch + 1}<br>%{{x:.0f}} ms: %{{y:.2f}}<extra></extra>"))
        fig.add_trace(go.Scatter(x=t, y=x.std(axis=0), mode="lines", line=dict(width=2.5, color="black"),
                                 name="Global field power"))
    else:  # average
        fig.add_trace(go.Scatter(x=t, y=x.mean(axis=0), mode="lines", name="Mean across channels"))
        fig.add_trace(go.Scatter(x=t, y=x.std(axis=0), mode="lines", line=dict(dash="dot"),
                                 name="Global field power (std across channels)"))
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="stimulus onset")
    fig.update_layout(title="EEG Signal", height=340, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis_title="Time (ms)", yaxis_title="Amplitude (a.u.)",
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right"))
    return fig


# ================================================================= TAB 1: upload inference
def render_inference_tab():
    st.markdown("Upload a processed EEG epoch and NeuroVision retrieves the visual concepts from its "
                "200-image THINGS gallery whose representation is closest to the EEG representation.")
    st.info("NeuroVision estimates visual concepts associated with the EEG representation. "
            "It does not decode unrestricted thoughts.")

    try:
        model, best_epoch, _ = get_model()
        W, sess_idx = get_mvnn()
        gallery = get_gallery()
    except (nv.ArtifactError, FileNotFoundError) as e:
        st.error(f"{e}. Check the artifacts/ folder.")
        return
    except Exception as e:
        st.error(f"Could not load the NeuroVision model/artifacts: {type(e).__name__}: {e}")
        return

    st.subheader("1 · Upload EEG Signal")
    up = st.file_uploader("Upload EEG Signal", type=["npy"],
                          help=f"Processed epoch, .npy, shape ({N_CHANNELS}, {N_SAMPLES}): 63 channels, 225 samples, "
                               "250 Hz, -0.1 to 0.8 s, baseline-corrected and channel z-scored per trial "
                               "(same format as the THINGS-EEG2 test epochs).")
    c1, c2 = st.columns(2)
    sessions = list(sess_idx)
    session = c1.selectbox("Whitening session", sessions, index=0,
                           help="The MVNN whitening matrix is fitted per recording session. An uploaded epoch has no "
                                "session label, so choose the matrix to apply (ses-01 by default).")
    top_k = c2.slider("Top-K candidates", 1, 10, 5)

    if up is None:
        st.caption("Supported format: .npy · expected single-trial shape (63, 225).")
        return

    raw = up.getvalue()
    try:
        eeg = validate_eeg(load_npy_bytes(raw, up.name))
    except EEGValidationError as e:
        st.error(str(e))
        return
    x = eeg.data

    # ---- EEG information
    st.subheader("2 · EEG information")
    m = st.columns(6)
    m[0].metric("Filename", up.name if len(up.name) < 22 else up.name[:19] + "...")
    m[1].metric("Channels", x.shape[0])
    m[2].metric("Samples", x.shape[1])
    m[3].metric("Sampling rate", f"{SFREQ:.0f} Hz")
    m[4].metric("Duration", f"{eeg.duration_s:.2f} s")
    m[5].metric("Shape", str(tuple(x.shape)))
    notes = ["Sampling rate is fixed by the processed-epoch format (a .npy carries no metadata)."]
    if eeg.transposed:
        notes.append(f"Input shape {eeg.original_shape} was transposed to (channels, samples).")
    if eeg.n_trials_averaged > 1:
        notes.append(f"{eeg.n_trials_averaged} trials in the file were averaged into one epoch.")
    st.caption(" ".join(notes))

    # ---- waveform
    st.subheader("3 · EEG Signal")
    v1, v2 = st.columns([3, 1])
    view = v1.radio("View", ["Average across channels", "Single channel", "All channels (butterfly)"], horizontal=True)
    ch = None
    if view == "Single channel":
        ch = int(v2.number_input("Channel", 1, N_CHANNELS, 1)) - 1
    st.plotly_chart(eeg_figure(x, view, ch), width="stretch")

    # ---- process
    st.markdown("<div class='nv-flow'>EEG Signal ↓ NeuroVision model (MVNN whitening → EEG encoder) ↓ "
                "EEG representation ↓ cosine retrieval over gallery ↓ Top-K predictions</div>",
                unsafe_allow_html=True)
    key = hashlib.sha1(raw).hexdigest() + "|" + session
    if st.button("Process EEG", type="primary"):
        try:
            with st.spinner("Running NeuroVision..."):
                scores, order, whitened = nv.predict_from_eeg(x, model, gallery, W, sess_idx[session])
            st.session_state["result"] = dict(key=key, scores=scores, order=order)
        except Exception as e:
            st.session_state.pop("result", None)
            st.error(f"Inference failed: {type(e).__name__}: {e}")
            return

    res = st.session_state.get("result")
    if not res or res["key"] != key:
        st.caption("Click **Process EEG** to run inference on this signal.")
        return

    # ---- prediction
    scores, order = res["scores"], res["order"]
    images = get_image_lookup()
    st.subheader("NeuroVision Prediction")
    top1 = int(order[0])
    st.markdown(f"**Predicted visual concept (Top-1):** &nbsp; <span style='font-size:1.6rem'>"
                f"{gallery.concept(top1).title()}</span> &nbsp; "
                f"<span class='nv-sub'>retrieval score {scores[top1]:.3f}</span>", unsafe_allow_html=True)

    cols = st.columns(top_k)
    for r, (col, gi) in enumerate(zip(cols, order[:top_k]), start=1):
        gi = int(gi)
        with col:
            if gallery.names[gi] in images:
                st.image(load_thumb(str(images[gallery.names[gi]]), 256), width="stretch")
            st.markdown(f"<span class='nv-badge nv-rank'>#{r}</span> **{gallery.concept(gi).title()}**<br>"
                        f"<span class='nv-sub'>Retrieval score {scores[gi]:.3f}</span>", unsafe_allow_html=True)
    if not images:
        st.caption("Gallery thumbnails unavailable (data/test_images/ not found); concepts are shown by name.")

    st.dataframe(pd.DataFrame({"Rank": range(1, top_k + 1),
                               "Predicted visual concept": [gallery.concept(int(i)).title() for i in order[:top_k]],
                               "Gallery image": [gallery.names[int(i)] for i in order[:top_k]],
                               "Similarity score (cosine)": [round(float(scores[i]), 4) for i in order[:top_k]]}),
                 hide_index=True, width="stretch")
    st.caption(f"Scores are cosine similarities between the EEG and image embeddings, not calibrated probabilities. "
               f"Gallery: {len(gallery.names)} images · checkpoint epoch {best_epoch}. "
               f"The model's benchmark accuracy is modest (see Evaluation tab); a prediction is not evidence of correctness.")


# ================================================================= TAB 2: original benchmark
def render_evaluation_tab():
    if not (EEG_PATH.exists() and MAPPING_PATH.exists() and IMAGE_DIR.exists()):
        st.warning("The Evaluation tab needs the THINGS-EEG2 test data in `data/` "
                   "(test_eeg.npy, test_trial_image_mapping.csv, test_images/). It was not found, "
                   "so only the stored benchmark metrics are shown.")
    try:
        model, best_epoch, _ = get_model()
        W, sess_idx = get_mvnn()
        gal = get_gallery()
    except Exception as e:
        st.error(f"Could not load the model/artifacts: {type(e).__name__}: {e}")
        return

    m = load_metrics()
    n_gal = len(gal.names)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Recall@1", f"{m.get('recall@1', float('nan')):.1%}")
    k2.metric("Recall@5", f"{m.get('recall@5', float('nan')):.1%}")
    k3.metric("Recall@10", f"{m.get('recall@10', float('nan')):.1%}")
    k4.metric("Median rank", f"{m.get('median_rank', float('nan')):.0f} / {n_gal}")
    k5.metric("Chance Recall@1", f"{1 / n_gal:.1%}")
    st.caption("Benchmark metrics: EEG averaged over all ~81 test repeats per image before encoding. "
               f"THINGS-EEG2 subject 01 · checkpoint epoch {best_epoch}.")

    if not (EEG_PATH.exists() and MAPPING_PATH.exists() and IMAGE_DIR.exists()):
        return
    try:
        eeg, mapping, names, reps, concept = load_benchmark_data()
    except Exception as e:
        st.error(f"Could not load benchmark data: {type(e).__name__}: {e}")
        return
    image_lookup = get_image_lookup()
    names = [n for n in names if n in image_lookup]
    pos = {n: i for i, n in enumerate(gal.names)}
    if any(n not in pos for n in names):
        st.error("Benchmark images are not all present in artifacts/gallery_features.npz.")
        return
    # restrict scoring to this benchmark gallery, in artifact embedding order
    idx = np.array([pos[n] for n in names])
    z_gallery = gal.z[idx]

    st.divider()
    left_c, right_c = st.columns([1, 2])
    with left_c:
        options = sorted(names, key=lambda f: (str(concept.get(f, "")).lower(), f))

        def _pick_random():
            st.session_state["target_sel"] = str(np.random.choice(names))
        st.button("🎲 Random image", on_click=_pick_random)
        target = st.selectbox("Test image (concept)", options, key="target_sel",
                              format_func=lambda f: f"{concept.get(f, '?')}  ·  {f}")
        target_rows = reps[target]
        mode = st.radio("Mode", ["Single-Trial", "Signal-Averaged"], horizontal=True)
        if mode == "Single-Trial":
            rep_i = st.selectbox(f"Repetition ({len(target_rows)} recorded)", range(len(target_rows)),
                                 format_func=lambda i: f"#{i + 1}  ·  row {target_rows[i]}  ·  {mapping['session'].iloc[target_rows[i]]}")
            rows = target_rows[[rep_i]]
        else:
            k = st.slider("Trials averaged (k)", 2, len(target_rows), min(20, len(target_rows)))
            seed = st.session_state.setdefault("seed", 0)
            if st.button("🔀 Resample trials"):
                st.session_state["seed"] = seed = seed + 1
            rng = np.random.default_rng(abs(hash((target, k, seed))) % (2 ** 32))
            rows = np.sort(rng.choice(target_rows, size=k, replace=False))
        top_k = st.slider("Top-K candidates", 1, 10, 5, key="bench_topk")

    raw = np.asarray(eeg[rows], dtype=np.float32)
    ks = np.array([sess_idx[s] for s in mapping["session"].values[rows]])
    whitened = nv.whiten(raw, W, ks)                  # per-trial session whitening (shared function)
    model_input = whitened.mean(axis=0)
    scores = nv.retrieve(model_input, z_gallery, model)  # shared encoder + retrieval
    order = np.argsort(-scores)
    true_idx = names.index(target)
    true_rank = int(np.where(order == true_idx)[0][0]) + 1
    top1 = names[order[0]]

    with right_c:
        a, b = st.columns(2)
        with a.container(border=True):
            st.subheader("Ground truth")
            st.image(load_thumb(str(image_lookup[target])), width="stretch")
            st.markdown(f"**{concept.get(target, '?')}**")
        with b.container(border=True):
            st.subheader("Top-1 retrieved")
            st.image(load_thumb(str(image_lookup[top1])), width="stretch")
            badge = "<span class='nv-badge nv-ok'>✓ CORRECT</span>" if top1 == target else "<span class='nv-badge nv-bad'>✗ WRONG</span>"
            st.markdown(f"{badge} **{concept.get(top1, '?')}** · cosine {scores[order[0]]:.3f}  \n"
                        f"<span class='nv-sub'>True image ranked <b>{true_rank} / {len(names)}</b></span>",
                        unsafe_allow_html=True)

    st.subheader(f"Top-{top_k} candidates")
    for r, (col, gi) in enumerate(zip(st.columns(top_k), order[:top_k]), start=1):
        name = names[gi]
        with col:
            st.image(load_thumb(str(image_lookup[name]), 256), width="stretch")
            cls, tag = ("nv-true", " · target") if name == target else ("nv-rank", "")
            st.markdown(f"<span class='nv-badge {cls}'>#{r}{tag}</span> <span class='nv-sub'>{scores[gi]:.3f}</span>"
                        f"<br><span class='nv-sub'>{concept.get(name, '?')}</span>", unsafe_allow_html=True)

    with st.expander("Electrophysiology & MVNN Diagnostics"):
        view = st.radio("Signal", ["Raw scalp EEG", "Whitened EEG (W · x)"], horizontal=True)
        sig = raw.mean(axis=0) if view.startswith("Raw") else model_input
        st.plotly_chart(eeg_figure(sig, "All channels (butterfly)", None), width="stretch")


# ================================================================= main
st.title("🧠 NeuroVision")
st.caption("EEG-based visual concept retrieval · THINGS-EEG2 · EEG Conformer + DINOv2/CLIP · MVNN whitening")
tab_infer, tab_eval = st.tabs(["EEG Inference", "Evaluation / Benchmark"])
with tab_infer:
    render_inference_tab()
with tab_eval:
    render_evaluation_tab()
