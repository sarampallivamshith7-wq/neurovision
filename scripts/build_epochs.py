"""
Preprocess THINGS-EEG2 BDF recordings into model-ready epochs (README Phase 1).

For each subject/session it finds ``*_task-train_run-*_events.tsv`` and ``*_task-test_events.tsv``
next to their ``*_eeg.bdf``, then: parse events -> filter/notch/resample -> epoch -> baseline ->
z-score, and writes

    <processed_dir>/<sub>_<ses>_<split>_eeg.npy    float32 (n_trials, 63, 225)  [memmap-loadable]
    <processed_dir>/<sub>_<ses>_<split>_meta.csv   one row per retained trial, same order

Only one run's recording is held in memory at a time.

    python scripts/build_epochs.py --config configs/phase1.yaml
    python scripts/build_epochs.py --subjects sub-01 sub-02 --sessions ses-01 ses-02
    python scripts/build_epochs.py --max-trials 200        # quick smoke run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.event_parser import parse_events_and_stimuli  # noqa: E402
from src.data.preprocessing import extract_epochs_from_events, preprocess_raw  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("neurovision.build_epochs")


def process_recording(bdf_path: Path, events_path: Path, cfg, max_trials=None):
    p, d = cfg["preprocessing"], cfg["dataset"]
    df_trials, resolver = parse_events_and_stimuli(
        events_tsv_path=events_path,
        stimuli_tsv_path=d["stimuli_tsv"],
        stimuli_dir=d["stimuli_dir"],
        training_zip_path=d["training_zip"],
        max_trials=max_trials,
    )
    resolver.close()
    raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)
    if abs(raw.info["sfreq"] - p["original_sfreq"]) > 1e-6:
        raise ValueError(f"{bdf_path.name}: sfreq {raw.info['sfreq']} != config {p['original_sfreq']}")
    raw = preprocess_raw(
        raw, l_freq=p["filter_low"], h_freq=p["filter_high"], notch_freq=p["notch_freq"],
        target_sfreq=p["target_sfreq"], copy=False,
    )
    epochs, kept = extract_epochs_from_events(
        raw, df_trials, epoch_start=p["epoch_start"], epoch_end=p["epoch_end"],
        baseline_start=p["baseline_start"], baseline_end=p["baseline_end"],
        normalization=p["normalization"], expected_channels=p["expected_channels"],
    )
    if len(kept) < len(df_trials):
        logger.warning(f"{bdf_path.name}: dropped {len(df_trials) - len(kept)} / {len(df_trials)} trials (out of bounds / non-finite).")
    return epochs, kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--subjects", nargs="+")
    ap.add_argument("--sessions", nargs="+")
    ap.add_argument("--max-trials", type=int, default=None, help="Limit trials per recording (smoke tests).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = cfg["dataset"]
    subjects = args.subjects or d.get("subjects") or [d["subject"]]
    sessions = args.sessions or d.get("sessions") or [d["session"]]
    out_dir = Path(d["processed_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    for sub in subjects:
        for ses in sessions:
            eeg_dir = Path(d["bids_root"]) / sub / ses / "eeg"
            for split in ("train", "test"):
                events = sorted(eeg_dir.glob(f"{sub}_{ses}_task-{split}*_events.tsv"))
                if not events:
                    logger.warning(f"No {split} events in {eeg_dir} - skipping.")
                    continue
                all_epochs, all_meta = [], []
                for ev in events:
                    bdf = ev.with_name(ev.name.replace("_events.tsv", "_eeg.bdf"))
                    if not bdf.exists() or bdf.stat().st_size <= 1000:
                        logger.warning(f"{bdf.name} missing or a git-annex stub - skipping (fetch the real file).")
                        continue
                    logger.info(f"Processing {bdf.name}")
                    epochs, meta = process_recording(bdf, ev, cfg, args.max_trials)
                    meta["run_file"] = bdf.name
                    all_epochs.append(epochs); all_meta.append(meta)
                if not all_epochs:
                    continue
                eeg = np.concatenate(all_epochs, axis=0).astype(np.float32)
                meta = pd.concat(all_meta, ignore_index=True)
                meta["trial_idx"] = np.arange(len(meta))  # global index within this file
                base = out_dir / f"{sub}_{ses}_{split}"
                np.save(f"{base}_eeg.npy", eeg)
                meta.to_csv(f"{base}_meta.csv", index=False)
                logger.info(f"Saved {eeg.shape} -> {base}_eeg.npy")


if __name__ == "__main__":
    main()
