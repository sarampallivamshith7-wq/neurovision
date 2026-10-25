"""
EEG-image contrastive alignment training (README Phases 2-8).

    python -m src.training.train_alignment --config configs/phase1.yaml
    python -m src.training.train_alignment --config configs/phase1.yaml --encoder gnn_conformer
    python -m src.training.train_alignment --synthetic --epochs 5      # no dataset needed

Every run writes, under ``<checkpoint_dir>/<run_name>/``: ``config.yaml`` (exact config used),
``history.csv`` (per-epoch loss / validation metrics) and ``best.pt``; and, under
``<results_dir>/<run_name>/``: ``test_metrics.json``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.eeg_dataset import ConcatArray, ThingsEEGDataset
from src.evaluation.retrieval import evaluate_retrieval
from src.models.aligner import NeuroVisionModel, build_model
from src.models.subject_adversary import dann_lambda
from src.training.losses import batch_retrieval_accuracy, info_nce_loss
from src.utils.config import ConfigDict, load_config
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger("neurovision.train")


# --------------------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------------------
def load_processed_split(cfg: Any, split: str) -> Tuple[Any, pd.DataFrame, int]:
    """
    Load epochs written by ``scripts/build_epochs.py`` for every configured subject/session.

    Returns (eeg array [memmap / ConcatArray], metadata DataFrame with ``subject_idx``,
    number of subjects). Nothing is read into RAM until trials are indexed.
    """
    ds = cfg["dataset"]
    subjects = ds.get("subjects") or [ds["subject"]]
    sessions = ds.get("sessions") or [ds["session"]]
    arrays, metas = [], []
    for s_idx, sub in enumerate(subjects):
        for ses in sessions:
            base = Path(ds["processed_dir"]) / f"{sub}_{ses}_{split}"
            eeg_path, meta_path = Path(f"{base}_eeg.npy"), Path(f"{base}_meta.csv")
            if not (eeg_path.exists() and meta_path.exists()):
                raise FileNotFoundError(
                    f"Processed epochs not found: {eeg_path}. "
                    f"Run: python scripts/build_epochs.py --config <cfg>"
                )
            arrays.append(np.load(eeg_path, mmap_mode="r"))
            meta = pd.read_csv(meta_path)
            meta["subject_idx"] = s_idx
            metas.append(meta)
    eeg = arrays[0] if len(arrays) == 1 else ConcatArray(arrays)
    return eeg, pd.concat(metas, ignore_index=True), len(subjects)


def split_by_stimulus(stimulus_ids: List[str], val_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split trial indices into train/val **by image**, never by trial, so every validation image
    is completely unseen during training (no repetition leakage across the split).
    """
    unique = sorted(set(stimulus_ids))
    if val_fraction <= 0 or len(unique) < 4:
        return np.arange(len(stimulus_ids)), np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    n_val = int(min(max(2, round(val_fraction * len(unique))), len(unique) - 2))
    val_ids = set(rng.choice(unique, size=n_val, replace=False).tolist())
    is_val = np.array([s in val_ids for s in stimulus_ids])
    return np.where(~is_val)[0], np.where(is_val)[0]


def resolve_device(cfg: Any) -> torch.device:
    t = cfg["training"]
    if t.get("device_auto", True) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device(t.get("device", "cpu"))


# --------------------------------------------------------------------------------------
# Checkpoint IO
# --------------------------------------------------------------------------------------
def save_checkpoint(path: Path, model: NeuroVisionModel, cfg: Any, epoch: int, metrics: Dict, num_subjects: int) -> None:
    cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    torch.save(
        {"model": model.state_dict(), "config": cfg_dict, "epoch": epoch, "metrics": metrics, "num_subjects": num_subjects},
        path,
    )


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> Tuple[NeuroVisionModel, ConfigDict, Dict]:
    """Rebuild the model exactly as trained (from the config stored in the checkpoint)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ConfigDict(ckpt["config"])
    model = build_model(cfg, num_subjects=ckpt["num_subjects"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg, ckpt


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def train_model(
    cfg: Any,
    train_ds: Dataset,
    val_ds: Optional[Dataset],
    cache: Any,
    num_subjects: int,
    out_dir: Path,
) -> Dict[str, Any]:
    """Run contrastive training. Returns a dict with history, best epoch and checkpoint path."""
    t_cfg, adv_cfg = cfg["training"], cfg["subject_adversarial"]
    device = resolve_device(cfg)
    set_seed(int(cfg.get("seed", 42)))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg), f, sort_keys=False)

    model = build_model(cfg, num_subjects=num_subjects).to(device)
    adversarial = model.subject_classifier is not None
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Encoder={cfg['eeg_model']['encoder_type']} | params={n_params:,} | device={device} | "
        f"train_trials={len(train_ds)} | val_trials={0 if val_ds is None else len(val_ds)} | adversarial={adversarial}"
    )

    batch_size = min(int(t_cfg["batch_size"]), len(train_ds))
    gen = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=len(train_ds) > batch_size,
        num_workers=int(t_cfg.get("num_workers", 0)), generator=gen,
    )
    epochs = int(t_cfg["epochs"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(t_cfg["learning_rate"]), weight_decay=float(t_cfg["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    use_amp = bool(t_cfg.get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    monitor = str(t_cfg.get("monitor", "e2i_mrr"))
    patience = int(t_cfg.get("patience", 7))
    ks = tuple(cfg["retrieval"]["top_k"])
    temperature = float(t_cfg["temperature"])
    symmetric = bool(t_cfg.get("symmetric_loss", True))
    grad_clip = float(t_cfg.get("grad_clip", 0.0) or 0.0)

    history: List[Dict[str, float]] = []
    best_val, best_epoch, bad_epochs = -float("inf"), -1, 0
    ckpt_path = out_dir / "best.pt"
    total_steps, step = max(epochs * len(loader), 1), 0

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        agg = {"loss": 0.0, "nce": 0.0, "adv": 0.0, "acc": 0.0, "subj_acc": 0.0}
        n_batches = 0
        for batch in loader:
            eeg = batch["eeg"].to(device)
            dino, clip = (t.to(device) for t in cache.get_features(batch["stimulus_id"]))
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(eeg, dino, clip)
            nce = info_nce_loss(out["z_eeg"], out["z_img"], temperature, symmetric, batch["stimulus_id"])
            loss = nce
            if adversarial:
                model.subject_classifier.set_lambda(dann_lambda(step / total_steps, max_lambda=float(adv_cfg["lambda_max"])))
                subj = batch["subject_idx"].to(device)
                adv = F.cross_entropy(out["subject_logits"].float(), subj)
                loss = nce + float(adv_cfg["weight"]) * adv
                agg["adv"] += adv.item()
                agg["subj_acc"] += (out["subject_logits"].argmax(1) == subj).float().mean().item()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            agg["loss"] += loss.item(); agg["nce"] += nce.item()
            agg["acc"] += batch_retrieval_accuracy(out["z_eeg"].detach(), out["z_img"].detach(), batch["stimulus_id"])
            n_batches += 1
            step += 1
        sched.step()

        row: Dict[str, float] = {"epoch": epoch, "lr": opt.param_groups[0]["lr"]}
        row.update({f"train_{k}": v / max(n_batches, 1) for k, v in agg.items()})
        if val_ds is not None and len(val_ds) > 0:
            val = evaluate_retrieval(model, val_ds, cache, ks, device)
            row.update({f"val_{k}": v for k, v in val.items()})
            score = val[monitor]
        else:
            score = -row["train_nce"]  # no validation split: track training loss
        row["seconds"] = time.time() - t0
        history.append(row)

        logger.info(
            f"epoch {epoch:03d}/{epochs} loss={row['train_loss']:.4f} batch_acc={row['train_acc']:.3f} "
            + (f"val_{monitor}={score:.4f} val_e2i_R@1={row.get('val_e2i_recall@1', float('nan')):.3f} " if val_ds is not None else "")
            + f"({row['seconds']:.1f}s)"
        )

        if score > best_val:
            best_val, best_epoch, bad_epochs = score, epoch, 0
            save_checkpoint(ckpt_path, model, cfg, epoch, {monitor: score}, num_subjects)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.info(f"Early stopping at epoch {epoch} (best epoch {best_epoch}).")
                break

    keys = sorted({k for r in history for k in r}, key=lambda k: (k != "epoch", k))
    with open(out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(history)

    return {"history": history, "best_epoch": best_epoch, "best_score": best_val, "checkpoint": ckpt_path, "out_dir": out_dir}


def run_training(cfg: Any, synthetic: bool = False) -> Dict[str, Any]:
    """Full pipeline: load data -> split by image -> train -> evaluate best checkpoint on test set."""
    set_seed(int(cfg.get("seed", 42)))
    run_name = cfg["training"].get("run_name") or f"{cfg['eeg_model']['encoder_type']}_{datetime.now():%Y%m%d_%H%M%S}"
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"]) / run_name
    res_dir = Path(cfg["retrieval"]["results_dir"]) / run_name

    if synthetic:
        from src.data.synthetic import make_synthetic_alignment_data

        syn = make_synthetic_alignment_data(n_subjects=2 if cfg["subject_adversarial"]["enabled"] else 1, seed=int(cfg.get("seed", 42)))
        train_eeg, train_meta, test_eeg, test_meta, cache, num_subjects = (
            syn.train_eeg, syn.train_meta, syn.test_eeg, syn.test_meta, syn.cache, syn.num_subjects,
        )
    else:
        from src.vision.cache_embeddings import ImageFeatureCache

        emb = Path(cfg["dataset"]["embeddings_dir"])
        cache = ImageFeatureCache(emb / "image_features.npz", emb / "image_metadata.csv")
        if len(cache) == 0:
            raise FileNotFoundError(f"No cached image features in {emb}. Run: python scripts/cache_features.py")
        train_eeg, train_meta, num_subjects = load_processed_split(cfg, "train")
        test_eeg, test_meta, _ = load_processed_split(cfg, "test")

    full_train = ThingsEEGDataset(train_eeg, train_meta)
    tr_idx, va_idx = split_by_stimulus(train_meta["stimulus_id"].astype(str).tolist(), float(cfg["training"]["val_fraction"]), int(cfg.get("seed", 42)))
    train_ds = Subset(full_train, tr_idx.tolist())
    val_ds = Subset(full_train, va_idx.tolist()) if len(va_idx) else None
    test_ds = ThingsEEGDataset(test_eeg, test_meta)
    logger.info(f"Images: train={train_meta.iloc[tr_idx]['stimulus_id'].nunique()} "
                f"val={0 if not len(va_idx) else train_meta.iloc[va_idx]['stimulus_id'].nunique()} "
                f"test={test_meta['stimulus_id'].nunique()}")

    result = train_model(cfg, train_ds, val_ds, cache, num_subjects, ckpt_dir)

    # Final evaluation on the held-out test images with the best checkpoint.
    device = resolve_device(cfg)
    model, _, _ = load_checkpoint(result["checkpoint"], device)
    test_metrics = evaluate_retrieval(model, test_ds, cache, tuple(cfg["retrieval"]["top_k"]), device)
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"run_name": run_name, "best_epoch": result["best_epoch"], "metrics": test_metrics}, f, indent=2)
    logger.info("TEST " + " | ".join(f"{k}={v:.4f}" for k, v in test_metrics.items() if k.startswith("e2i_") and ("recall" in k or "mrr" in k)))
    result["test_metrics"] = test_metrics
    result["results_dir"] = res_dir
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Train EEG-image contrastive alignment.")
    p.add_argument("--config", default="configs/phase1.yaml")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic data (pipeline check; no dataset needed).")
    p.add_argument("--encoder", choices=["eegnet", "conformer", "gnn_conformer"])
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--adversarial", action="store_true", help="Enable subject-adversarial learning.")
    p.add_argument("--run-name")
    a = p.parse_args()

    cfg = load_config(a.config)
    if a.encoder: cfg["eeg_model"]["encoder_type"] = a.encoder
    if a.epochs: cfg["training"]["epochs"] = a.epochs
    if a.batch_size: cfg["training"]["batch_size"] = a.batch_size
    if a.run_name: cfg["training"]["run_name"] = a.run_name
    if a.adversarial: cfg["subject_adversarial"]["enabled"] = True
    run_training(cfg, synthetic=a.synthetic)


if __name__ == "__main__":
    main()
