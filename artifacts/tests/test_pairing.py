"""
Sanity check: Verify strict EEG trial to stimulus image pairing.
"""

from pathlib import Path
from PIL import Image
import pytest

from src.data.event_parser import parse_events_and_stimuli


@pytest.mark.data
def test_test_stimuli_pairing():
    events_path = Path("data/sub-01/sub-01/ses-01/eeg/sub-01_ses-01_task-test_events.tsv")
    stimuli_path = Path("data/sub-01/stimuli/stimuli.tsv")
    stimuli_dir = Path("data/sub-01/stimuli")

    if not (events_path.exists() and stimuli_path.exists()):
        pytest.skip("THINGS-EEG2 data not available")

    # Parse first 20 test trials
    df_trials, resolver = parse_events_and_stimuli(
        events_tsv_path=events_path,
        stimuli_tsv_path=stimuli_path,
        stimuli_dir=stimuli_dir,
        max_trials=20,
    )

    assert len(df_trials) == 20
    assert "stimulus_id" in df_trials.columns
    assert "tot_img_number" in df_trials.columns

    for _, row in df_trials.iterrows():
        stim_id = row["stimulus_id"]
        assert stim_id.startswith("stim-test")

        # Verify image exists on disk and can be opened
        img = resolver.load_pil_image(stim_id)
        assert isinstance(img, Image.Image)
        assert img.size[0] > 0 and img.size[1] > 0

    resolver.close()


@pytest.mark.data
def test_train_stimuli_pairing_from_zip():
    events_path = Path("data/sub-01/sub-01/ses-01/eeg/sub-01_ses-01_task-train_run-01_events.tsv")
    stimuli_path = Path("data/sub-01/stimuli/stimuli.tsv")
    stimuli_dir = Path("data/sub-01/stimuli")
    zip_path = Path("data/sub-01/stimuli/training_images.zip")

    if not (events_path.exists() and zip_path.exists()):
        pytest.skip("THINGS-EEG2 data not available")

    # Parse first 10 training trials
    df_trials, resolver = parse_events_and_stimuli(
        events_tsv_path=events_path,
        stimuli_tsv_path=stimuli_path,
        stimuli_dir=stimuli_dir,
        training_zip_path=zip_path,
        max_trials=10,
    )

    assert len(df_trials) == 10

    for _, row in df_trials.iterrows():
        stim_id = row["stimulus_id"]
        assert stim_id.startswith("stim-train")

        # Verify image can be extracted/read on-demand from zip
        img = resolver.load_pil_image(stim_id)
        assert isinstance(img, Image.Image)
        assert img.size[0] > 0 and img.size[1] > 0

    resolver.close()
