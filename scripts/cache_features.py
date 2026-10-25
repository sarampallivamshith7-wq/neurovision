"""
Cache frozen DINOv2 + CLIP features for every stimulus image (README Phase 5).

    python scripts/cache_features.py --config configs/phase1.yaml [--split train|test|both]

Reads stimulus ids from the stimuli manifest, so it does not need the EEG to be processed
first. Output: <embeddings_dir>/image_features.npz + image_metadata.csv. NOTE: first run
downloads DINOv2 (torch.hub / GitHub) and CLIP (Hugging Face) weights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.event_parser import StimuliResolver  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.vision.cache_embeddings import cache_image_features  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None, help="Only the first N ids (smoke test).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d, v = cfg["dataset"], cfg["vision"]
    resolver = StimuliResolver(d["stimuli_tsv"], d["stimuli_dir"], d["training_zip"])
    ids = resolver.df_stimuli["stimulus_id"].astype(str).tolist()
    if args.split != "both":
        ids = [i for i in ids if i.startswith(f"stim-{args.split}")]
    if args.limit:
        ids = ids[: args.limit]

    cache_image_features(
        ids, resolver, output_dir=d["embeddings_dir"], batch_size=int(v["cache_batch_size"]),
        device="cuda" if cfg["training"].get("device_auto", True) and __import__("torch").cuda.is_available() else cfg["training"]["device"],
        dinov2_model=v["dinov2_model"], clip_model=v["clip_model"], clip_pretrained=v["clip_pretrained"],
    )
    resolver.close()


if __name__ == "__main__":
    main()
