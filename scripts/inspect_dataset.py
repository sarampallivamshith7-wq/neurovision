"""
THINGS-EEG2 Dataset Inspection Script.

Inspects:
- Subjects and sessions
- Raw BDF file availability and sizes
- Channel count, names, and sampling rates via MNE
- Event TSV structures (task-train and task-test)
- Image stimulus files (test_images and training_images)
- Trial-to-image mapping consistency
"""

import json
import os
from pathlib import Path

import mne
import numpy as np
import pandas as pd


def inspect_dataset(bids_root: str = "data/sub-01"):
    root = Path(bids_root)
    print("=" * 70)
    print("            THINGS-EEG2 DATASET INSPECTION")
    print("=" * 70)
    print(f"Dataset root path: {root.resolve()}\n")

    # 1. Dataset description
    desc_path = root / "dataset_description.json"
    if desc_path.exists():
        with open(desc_path, "r", encoding="utf-8") as f:
            desc = json.load(f)
            print(f"Dataset Name: {desc.get('Name')}")
            print(f"BIDS Version: {desc.get('BIDSVersion')}")
            print(f"Dataset Version: {desc.get('Version')}")

    # 2. Participants
    participants_path = root / "participants.tsv"
    if participants_path.exists():
        df_p = pd.read_csv(participants_path, sep="\t")
        print(f"\nTotal Participants in participants.tsv: {len(df_p)}")
        print(df_p.to_string(index=False))

    # 3. Subject directories and file sizes
    sub_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("sub-")])
    print(f"\nFound {len(sub_dirs)} subject directories: {[d.name for d in sub_dirs]}")

    print("\n--- BDF File Availability per Subject ---")
    for s_dir in sub_dirs:
        bdf_files = list(s_dir.glob("**/*.bdf"))
        real_bdfs = [f for f in bdf_files if f.stat().st_size > 1000]
        stub_bdfs = [f for f in bdf_files if f.stat().st_size <= 1000]
        total_size_mb = sum(f.stat().st_size for f in real_bdfs) / (1024 * 1024)
        print(
            f"  {s_dir.name}: {len(real_bdfs)} real BDFs ({total_size_mb:.1f} MB), "
            f"{len(stub_bdfs)} git-annex stubs"
        )

    # 4. Sessions for sub-01
    sub01_dir = root / "sub-01"
    sessions = sorted([d.name for d in sub01_dir.iterdir() if d.is_dir() and d.name.startswith("ses-")])
    print(f"\nsub-01 Sessions: {sessions}")
    for ses in sessions:
        eeg_files = list((sub01_dir / ses / "eeg").glob("*_eeg.bdf"))
        print(f"  {ses}: {len(eeg_files)} BDF files")
        for ef in sorted(eeg_files):
            print(f"    - {ef.name} ({ef.stat().st_size / (1024*1024):.1f} MB)")

    # 5. MNE Inspection of a sample recording (sub-01 ses-01 task-train run-01)
    sample_bdf = sub01_dir / "ses-01" / "eeg" / "sub-01_ses-01_task-train_run-01_eeg.bdf"
    if sample_bdf.exists() and sample_bdf.stat().st_size > 1000:
        print(f"\n--- MNE Inspection of {sample_bdf.name} ---")
        raw = mne.io.read_raw_bdf(sample_bdf, preload=False, verbose=False)
        print(f"  Sampling frequency (sfreq): {raw.info['sfreq']} Hz")
        print(f"  Channel count: {len(raw.ch_names)}")
        print(f"  Total samples: {raw.n_times}")
        print(f"  Recording duration: {raw.n_times / raw.info['sfreq']:.2f} seconds ({raw.n_times / raw.info['sfreq'] / 60:.2f} minutes)")
        print(f"  Channel names (first 10): {raw.ch_names[:10]}")
        print(f"  Channel names (last 10): {raw.ch_names[-10:]}")
        ch_types = set(raw.get_channel_types())
        print(f"  Channel types: {ch_types}")

    # 6. Event TSV inspection (task-train and task-test)
    train_events_file = sub01_dir / "ses-01" / "eeg" / "sub-01_ses-01_task-train_run-01_events.tsv"
    test_events_file = sub01_dir / "ses-01" / "eeg" / "sub-01_ses-01_task-test_events.tsv"

    print("\n--- Event Files Structure ---")
    if train_events_file.exists():
        df_train_events = pd.read_csv(train_events_file, sep="\t")
        print(f"\nTask-Train (ses-01, run-01): {len(df_train_events)} total event rows")
        print(f"Columns: {list(df_train_events.columns)}")
        print("Trial types breakdown:")
        print(df_train_events["trial_type"].value_counts().to_string())
        img_train = df_train_events[df_train_events["trial_type"] == "image"]
        print(f"  Image presentation trials: {len(img_train)}")
        print(f"  Duration min/mean/max: {img_train['duration'].min():.4f}s / {img_train['duration'].mean():.4f}s / {img_train['duration'].max():.4f}s")
        print(f"  SOA min/mean/max (excluding n/a): {pd.to_numeric(img_train['soa'], errors='coerce').dropna().min():.4f}s / {pd.to_numeric(img_train['soa'], errors='coerce').dropna().mean():.4f}s / {pd.to_numeric(img_train['soa'], errors='coerce').dropna().max():.4f}s")
        print("\nSample Train Event Rows (first 3):")
        print(img_train.head(3)[["onset", "sample", "trial_type", "tot_img_number", "category_name", "block", "sequence", "img_in_sequence"]].to_string(index=False))

    if test_events_file.exists():
        df_test_events = pd.read_csv(test_events_file, sep="\t")
        print(f"\nTask-Test (ses-01): {len(df_test_events)} total event rows")
        img_test = df_test_events[df_test_events["trial_type"] == "image"]
        print(f"  Image presentation trials: {len(img_test)}")
        print(f"  Unique test images (tot_img_number): {img_test['tot_img_number'].nunique()}")
        print(f"  Repetitions per test image: {img_test['tot_img_number'].value_counts().iloc[0]}")

    # 7. Stimuli inspection
    stimuli_dir = root / "stimuli"
    stimuli_tsv = stimuli_dir / "stimuli.tsv"
    print("\n--- Stimuli & Image Database ---")
    if stimuli_tsv.exists():
        df_stim = pd.read_csv(stimuli_tsv, sep="\t")
        print(f"Total entries in stimuli.tsv: {len(df_stim)}")
        print(f"Columns: {list(df_stim.columns)}")
        train_stim = df_stim[df_stim["stimulus_id"].str.startswith("stim-train")]
        test_stim = df_stim[df_stim["stimulus_id"].str.startswith("stim-test")]
        print(f"  Train stimuli count: {len(train_stim)}")
        print(f"  Test stimuli count: {len(test_stim)}")

    # Check image files
    test_img_dir = stimuli_dir / "test_images"
    if test_img_dir.exists():
        test_img_cats = [d for d in test_img_dir.iterdir() if d.is_dir()]
        test_img_files = list(test_img_dir.glob("**/*_image.jpg"))
        print(f"Test images folder: {len(test_img_cats)} category folders, {len(test_img_files)} image files found")

    train_img_dir = stimuli_dir / "training_images"
    train_zip = stimuli_dir / "training_images.zip"
    if train_img_dir.exists():
        train_img_cats = [d for d in train_img_dir.iterdir() if d.is_dir()]
        print(f"Training images folder: {len(train_img_cats)} category folders extracted")
    if train_zip.exists():
        print(f"Training images zip archive: {train_zip.name} ({train_zip.stat().st_size / (1024*1024):.1f} MB)")

    # 8. Check EEG-to-Image Pairing Verification
    print("\n--- EEG-to-Image Mapping Verification ---")
    # Take first 5 train events and verify path in stimuli.tsv and on disk
    sample_train_img_events = img_train.head(5)
    for _, row in sample_train_img_events.iterrows():
        tot_num = int(row["tot_img_number"])
        cat_name = row["category_name"]
        stim_id = f"stim-train{tot_num:05d}"
        stim_row = df_stim[df_stim["stimulus_id"] == stim_id]
        if not stim_row.empty:
            rel_filename = stim_row.iloc[0]["filename"]
            full_img_path = stimuli_dir / rel_filename
            print(f"  Trial tot_num={tot_num} ({cat_name}) -> {stim_id} -> {rel_filename} (Exists extracted: {full_img_path.exists()})")

    print("\nSample Test Image Mapping:")
    sample_test_img_events = img_test.head(5)
    for _, row in sample_test_img_events.iterrows():
        tot_num = int(row["tot_img_number"])
        cat_name = row["category_name"]
        stim_id = f"stim-test{tot_num:03d}"
        stim_row = df_stim[df_stim["stimulus_id"] == stim_id]
        if not stim_row.empty:
            rel_filename = stim_row.iloc[0]["filename"]
            full_img_path = stimuli_dir / rel_filename
            print(f"  Trial tot_num={tot_num} ({cat_name}) -> {stim_id} -> {rel_filename} (Exists: {full_img_path.exists()})")

    print("\n" + "=" * 70)
    print("         INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    inspect_dataset()
