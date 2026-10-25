"""
Event and Stimuli Alignment Parser for THINGS-EEG2.

Parses BIDS events.tsv files, filters out catch/target trials, resolves image paths
against stimuli.tsv, and provides utilities for retrieving image data from disk or
directly from the training_images.zip archive without full extraction.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from PIL import Image


class StimuliResolver:
    """Resolves and loads stimulus images from disk or zip archive."""

    def __init__(
        self,
        stimuli_tsv_path: Union[str, Path] = "data/sub-01/stimuli/stimuli.tsv",
        stimuli_dir: Union[str, Path] = "data/sub-01/stimuli",
        training_zip_path: Optional[Union[str, Path]] = "data/sub-01/stimuli/training_images.zip",
    ) -> None:
        self.stimuli_dir = Path(stimuli_dir).resolve()
        self.stimuli_tsv_path = Path(stimuli_tsv_path).resolve()
        self.training_zip_path = Path(training_zip_path).resolve() if training_zip_path else None

        if not self.stimuli_tsv_path.exists():
            raise FileNotFoundError(f"stimuli.tsv not found: {self.stimuli_tsv_path}")

        # Load stimuli manifest
        self.df_stimuli = pd.read_csv(self.stimuli_tsv_path, sep="\t")

        # Extract concept and original filename from description column
        # e.g.: ... concept "01175_roller_coaster"; original filename "roller_coaster_02s.jpg".
        pattern = r'concept\s+"*([^";]+)"*;\s*original filename\s+"*([^";]+)"*'
        extracted = self.df_stimuli["description"].str.extract(pattern)
        self.df_stimuli["concept"] = extracted[0]
        self.df_stimuli["orig_filename"] = extracted[1]

        # Build lookup tables
        self.stim_id_to_bids_filename = dict(
            zip(self.df_stimuli["stimulus_id"], self.df_stimuli["filename"])
        )
        self.stim_id_to_concept = dict(
            zip(self.df_stimuli["stimulus_id"], self.df_stimuli["concept"])
        )
        self.stim_id_to_orig_filename = dict(
            zip(self.df_stimuli["stimulus_id"], self.df_stimuli["orig_filename"])
        )

        self._zip_file: Optional[zipfile.ZipFile] = None

    def _get_zip(self) -> Optional[zipfile.ZipFile]:
        if self._zip_file is None and self.training_zip_path and self.training_zip_path.exists():
            self._zip_file = zipfile.ZipFile(self.training_zip_path, "r")
        return self._zip_file

    def get_image_path(self, stimulus_id: str) -> Tuple[Optional[Path], bool]:
        """
        Return the primary disk path to the image file and a boolean indicating if it is on disk.
        """
        bids_rel = self.stim_id_to_bids_filename.get(stimulus_id)
        concept = self.stim_id_to_concept.get(stimulus_id)
        orig_file = self.stim_id_to_orig_filename.get(stimulus_id)

        # Check candidate locations on disk
        candidates = []
        if bids_rel:
            candidates.append(self.stimuli_dir / bids_rel)
        if concept and orig_file:
            prefix = "test_images" if "test" in stimulus_id else "training_images"
            candidates.append(self.stimuli_dir / prefix / concept / orig_file)

        for path in candidates:
            if path.exists():
                return path, True

        return (candidates[0] if candidates else None), False

    def load_pil_image(self, stimulus_id: str) -> Image.Image:
        """
        Load PIL image for given stimulus_id.
        Tries disk first (both BIDS name and original name), then falls back
        to reading directly from training_images.zip.
        """
        # 1. Try disk
        disk_path, exists = self.get_image_path(stimulus_id)
        if exists and disk_path is not None:
            return Image.open(disk_path).convert("RGB")

        # 2. Try training_images.zip archive
        concept = self.stim_id_to_concept.get(stimulus_id)
        orig_file = self.stim_id_to_orig_filename.get(stimulus_id)
        bids_rel = self.stim_id_to_bids_filename.get(stimulus_id)

        zip_ref = self._get_zip()
        if zip_ref is not None:
            zip_candidates = []
            if concept and orig_file:
                zip_candidates.append(f"training_images/{concept}/{orig_file}")
            if bids_rel:
                zip_candidates.append(bids_rel.replace("\\", "/"))

            for target in zip_candidates:
                try:
                    with zip_ref.open(target) as f:
                        return Image.open(io.BytesIO(f.read())).convert("RGB")
                except KeyError:
                    continue

        raise FileNotFoundError(
            f"Image for {stimulus_id} not found on disk or in {self.training_zip_path}"
        )

    def close(self) -> None:
        if self._zip_file is not None:
            self._zip_file.close()
            self._zip_file = None


def parse_events_and_stimuli(
    events_tsv_path: Union[str, Path],
    stimuli_tsv_path: Union[str, Path] = "data/sub-01/stimuli/stimuli.tsv",
    stimuli_dir: Union[str, Path] = "data/sub-01/stimuli",
    training_zip_path: Optional[Union[str, Path]] = "data/sub-01/stimuli/training_images.zip",
    max_trials: Optional[int] = None,
) -> Tuple[pd.DataFrame, StimuliResolver]:
    """
    Parse BIDS events.tsv file, filter out target catch trials, map each trial
    to its unique stimulus_id, category, and image reference.
    """
    events_file = Path(events_tsv_path).resolve()
    if not events_file.exists():
        raise FileNotFoundError(f"events.tsv file not found: {events_file}")

    df_events = pd.read_csv(events_file, sep="\t")
    resolver = StimuliResolver(
        stimuli_tsv_path=stimuli_tsv_path,
        stimuli_dir=stimuli_dir,
        training_zip_path=training_zip_path,
    )

    # Filter for valid image presentation trials
    valid_mask = (df_events["trial_type"] == "image") & (df_events["tot_img_number"].notna())
    df_valid = df_events[valid_mask].copy()

    is_test = "task-test" in events_file.name

    records = []
    for idx, (_, row) in enumerate(df_valid.iterrows()):
        tot_num = int(float(row["tot_img_number"]))
        if is_test:
            stim_id = f"stim-test{tot_num:03d}"
        else:
            stim_id = f"stim-train{tot_num:05d}"

        concept = resolver.stim_id_to_concept.get(stim_id, str(row["category_name"]))
        orig_filename = resolver.stim_id_to_orig_filename.get(stim_id, "")
        bids_filename = resolver.stim_id_to_bids_filename.get(stim_id, "")

        records.append({
            "trial_idx": idx,
            "onset": float(row["onset"]),
            "sample": int(float(row["sample"])),
            "duration": float(row["duration"]),
            "stimulus_id": stim_id,
            "tot_img_number": tot_num,
            "img_category": int(float(row["img_category"])),
            "category_name": str(row["category_name"]),
            "concept": concept,
            "orig_filename": orig_filename,
            "filename": bids_filename,
            "is_test": is_test,
        })

        if max_trials is not None and len(records) >= max_trials:
            break

    df_trials = pd.DataFrame(records)
    return df_trials, resolver
