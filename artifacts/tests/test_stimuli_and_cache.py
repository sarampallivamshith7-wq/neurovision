"""Dataset-free tests for stimulus resolution and the feature cache (fake extractors, tiny generated images)."""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.event_parser import StimuliResolver, parse_events_and_stimuli
from src.vision.cache_embeddings import ImageFeatureCache, cache_image_features

COLORS = {"stim-train00001": (255, 0, 0), "stim-train00002": (0, 255, 0), "stim-test001": (0, 0, 255)}


def build_stimuli(root: Path, with_zip=False):
    """Writes stimuli.tsv + images either on disk or only inside training_images.zip."""
    rows = []
    for i, (sid, color) in enumerate(COLORS.items(), 1):
        split = "test_images" if "test" in sid else "training_images"
        concept, fname = f"0000{i}_thing{i}", f"thing{i}_01b.jpg"
        rows.append({"stimulus_id": sid, "filename": f"{split}/{concept}/{fname}",
                     "description": f'concept "{concept}"; original filename "{fname}".'})
        rel = Path(split) / concept / fname
        img = Image.new("RGB", (32, 32), color)
        if with_zip and split == "training_images":
            with zipfile.ZipFile(root / "training_images.zip", "a") as z:
                tmp = root / "_tmp.jpg"; img.save(tmp); z.write(tmp, f"training_images/{concept}/{fname}"); tmp.unlink()
        else:
            (root / rel).parent.mkdir(parents=True, exist_ok=True); img.save(root / rel)
    pd.DataFrame(rows).to_csv(root / "stimuli.tsv", sep="\t", index=False)
    return root


class FakeExtractor:
    """Deterministic 'features' = mean RGB tiled to `dim`, so we can verify id<->feature pairing."""

    def __init__(self, dim): self.dim = dim

    def extract_from_pil_batch(self, images):
        return torch.stack([torch.tensor(np.asarray(im).reshape(-1, 3).mean(0), dtype=torch.float32).repeat(self.dim // 3 + 1)[: self.dim] for im in images])


def test_get_image_path_returns_flat_tuple_for_known_and_unknown_ids(tmp_path):
    r = StimuliResolver(build_stimuli(tmp_path) / "stimuli.tsv", tmp_path, None)
    path, on_disk = r.get_image_path("stim-train00001")
    assert isinstance(path, Path) and on_disk is True
    assert r.get_image_path("stim-train99999") == (None, False)     # regression: used to be ((None, False), False)


def test_images_load_from_disk_and_from_zip_fallback(tmp_path):
    (tmp_path / "disk_root").mkdir()
    disk = build_stimuli(tmp_path / "disk_root")
    zroot = tmp_path / "zip_root"; zroot.mkdir()
    build_stimuli(zroot, with_zip=True)
    for root, zp in ((disk, None), (zroot, zroot / "training_images.zip")):
        r = StimuliResolver(root / "stimuli.tsv", root, zp)
        assert r.load_pil_image("stim-train00001").getpixel((0, 0))[0] > 200   # red
        assert r.load_pil_image("stim-test001").getpixel((0, 0))[2] > 200      # blue (always on disk)
        with pytest.raises(FileNotFoundError):
            r.load_pil_image("stim-train00003")
        r.close()


def test_cache_roundtrip_preserves_id_feature_pairing_and_is_deterministic(tmp_path):
    root = build_stimuli(tmp_path)
    r = StimuliResolver(root / "stimuli.tsv", root, None)
    kw = dict(resolver=r, dino_extractor=FakeExtractor(384), clip_extractor=FakeExtractor(512))
    ids = ["stim-test001", "stim-train00002", "stim-train00001", "stim-train00001"]   # unsorted + duplicate
    c1, m1 = cache_image_features(ids, output_dir=tmp_path / "a", **kw)
    c2, m2 = cache_image_features(ids, output_dir=tmp_path / "b", **kw)
    a, b = ImageFeatureCache(c1, m1), ImageFeatureCache(c2, m2)
    assert len(a) == 3                                           # duplicates collapsed
    for sid, color in COLORS.items():
        d, c = a.get_features([sid])
        assert d.shape == (1, 384) and c.shape == (1, 512)
        assert d[0, 0].item() == pytest.approx(color[0], abs=4) and d[0, 1].item() == pytest.approx(color[1], abs=4)  # JPEG is lossy
    torch.testing.assert_close(a.get_features(list(COLORS))[0], b.get_features(list(COLORS))[0])
    assert a.is_cached("stim-test001") and not a.is_cached("stim-train00099")
    with pytest.raises(KeyError, match="not in the feature cache"):
        a.get_features(["stim-train00099"])


def test_cache_skips_unloadable_images_without_crashing(tmp_path):
    root = build_stimuli(tmp_path)
    r = StimuliResolver(root / "stimuli.tsv", root, None)
    c, m = cache_image_features(["stim-train00001", "stim-train00003"], r, output_dir=tmp_path / "o",
                                dino_extractor=FakeExtractor(384), clip_extractor=FakeExtractor(512))
    assert ImageFeatureCache(c, m).is_cached("stim-train00001") and len(ImageFeatureCache(c, m)) == 1


def test_parse_events_filters_non_image_rows_and_maps_ids(tmp_path):
    root = build_stimuli(tmp_path)
    ev = pd.DataFrame({
        "onset": [1.0, 1.2, 1.4, 1.6], "sample": [1000, 1200, 1400, 1600], "duration": 0.1,
        "trial_type": ["image", "image", "target", "image"],
        "tot_img_number": [1, 2, 2, np.nan], "img_category": [1, 2, 2, 0],
        "category_name": ["thing1", "thing2", "thing2", "n/a"],
    })
    f = tmp_path / "sub-01_ses-01_task-train_run-01_events.tsv"; ev.to_csv(f, sep="\t", index=False)
    df, res = parse_events_and_stimuli(f, root / "stimuli.tsv", root, None)
    assert df["stimulus_id"].tolist() == ["stim-train00001", "stim-train00002"]   # target + NaN rows dropped
    assert df["onset"].tolist() == [1.0, 1.2] and not df["is_test"].any()
    res.close()
