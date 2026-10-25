"""
Sanity check: Verify vision encoders, feature caching, and image projection head.
"""

from pathlib import Path
from PIL import Image
import numpy as np
import pytest
import torch

from src.data.event_parser import StimuliResolver
from src.vision.cache_embeddings import ImageFeatureCache, cache_image_features
from src.vision.clip_features import CLIPExtractor
from src.vision.dinov2_features import DINOv2Extractor
from src.vision.image_projection import ImageProjectionHead


@pytest.mark.network
def test_vision_extractors():
    # Create two synthetic RGB images
    img1 = Image.new("RGB", (224, 224), color=(255, 0, 0))
    img2 = Image.new("RGB", (224, 224), color=(0, 255, 0))

    try:  # weights are downloaded on first use (torch.hub / Hugging Face)
        dino = DINOv2Extractor(device="cpu")
        clip = CLIPExtractor(device="cpu")
    except Exception as exc:
        pytest.skip(f"Pretrained vision weights unavailable (offline?): {type(exc).__name__}")

    dino_feats = dino.extract_from_pil_batch([img1, img2])
    clip_feats = clip.extract_from_pil_batch([img1, img2])

    assert dino_feats.shape == (2, 384)
    assert clip_feats.shape == (2, 512)
    assert not torch.isnan(dino_feats).any()
    assert not torch.isnan(clip_feats).any()


def test_image_projection_head():
    head = ImageProjectionHead(
        dinov2_dim=384,
        clip_dim=512,
        embed_dim=512,
        dropout=0.0,
    )
    dino_dummy = torch.randn(4, 384)
    clip_dummy = torch.randn(4, 512)

    proj = head(dino_dummy, clip_dummy)

    assert proj.shape == (4, 512)
    # Verify L2 normalization
    norms = torch.norm(proj, p=2, dim=-1)
    np.testing.assert_allclose(norms.detach().numpy(), 1.0, atol=1e-5)


@pytest.mark.data
@pytest.mark.network
def test_caching_and_retrieval(tmp_path: Path):
    if not Path("data/sub-01/stimuli/stimuli.tsv").exists():
        pytest.skip("THINGS-EEG2 data not available")
    resolver = StimuliResolver(
        stimuli_tsv_path="data/sub-01/stimuli/stimuli.tsv",
        stimuli_dir="data/sub-01/stimuli",
    )

    sample_test_ids = ["stim-test001", "stim-test002"]
    cache_path, meta_path = cache_image_features(
        stimulus_ids=sample_test_ids,
        resolver=resolver,
        output_dir=tmp_path,
        device="cpu",
    )

    assert cache_path.exists()
    assert meta_path.exists()

    cache = ImageFeatureCache(cache_file=cache_path, metadata_file=meta_path)
    assert cache.is_cached("stim-test001")
    assert cache.is_cached("stim-test002")

    dino_t, clip_t = cache.get_features(["stim-test001", "stim-test002"])
    assert dino_t.shape == (2, 384)
    assert clip_t.shape == (2, 512)

    resolver.close()
