"""
Evaluate a trained checkpoint on the held-out test images (EEG->image and image->EEG).

    python scripts/evaluate.py --checkpoint checkpoints/<run>/best.pt --config configs/phase1.yaml
    python scripts/evaluate.py --checkpoint ... --topk-demo 5     # FAISS top-K retrieved images
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.eeg_dataset import ThingsEEGDataset  # noqa: E402
from src.evaluation.faiss_index import ImageIndex  # noqa: E402
from src.evaluation.retrieval import embed_eeg_dataset, embed_images, evaluate_retrieval  # noqa: E402
from src.training.train_alignment import load_checkpoint, load_processed_split, resolve_device  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.vision.cache_embeddings import ImageFeatureCache  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/phase1.yaml", help="Used only for data locations.")
    ap.add_argument("--topk-demo", type=int, default=0, help="Print FAISS top-K images for the first test images.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg)
    model, train_cfg, _ = load_checkpoint(args.checkpoint, device)
    emb = Path(cfg["dataset"]["embeddings_dir"])
    cache = ImageFeatureCache(emb / "image_features.npz", emb / "image_metadata.csv")
    eeg, meta, _ = load_processed_split(cfg, "test")
    ds = ThingsEEGDataset(eeg, meta)

    metrics = evaluate_retrieval(model, ds, cache, tuple(cfg["retrieval"]["top_k"]), device)
    print(json.dumps(metrics, indent=2))

    if args.topk_demo:
        z_eeg, ids = embed_eeg_dataset(model, ds, device)
        gallery = sorted(set(ids))
        index = ImageIndex(embed_images(model, cache, gallery, device).numpy(), gallery)
        _, top = index.search(z_eeg[: args.topk_demo].numpy(), k=5)
        print(f"\nFAISS backend: {index.backend}")
        for true_id, retrieved in zip(ids[: args.topk_demo], top):
            print(f"  true={true_id}  top5={retrieved}")


if __name__ == "__main__":
    main()
