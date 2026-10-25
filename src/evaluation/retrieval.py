"""EEG->image and image->EEG retrieval evaluation in the shared embedding space."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.evaluation.metrics import retrieval_metrics


@torch.no_grad()
def embed_eeg_dataset(
    model, dataset: Dataset, device="cpu", batch_size: int = 128
) -> Tuple[torch.Tensor, List[str]]:
    """Encode every trial of ``dataset``; returns (z_eeg (N, D) on CPU, stimulus_ids)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embs, ids = [], []
    for batch in loader:
        embs.append(model.encode_eeg(batch["eeg"].to(device)).cpu())
        ids.extend(batch["stimulus_id"])
    return torch.cat(embs), ids


@torch.no_grad()
def embed_images(model, cache, stimulus_ids: Sequence[str], device="cpu", batch_size: int = 1024) -> torch.Tensor:
    model.eval()
    out = []
    for i in range(0, len(stimulus_ids), batch_size):
        dino, clip = cache.get_features(stimulus_ids[i : i + batch_size])
        out.append(model.encode_image(dino.to(device), clip.to(device)).cpu())
    return torch.cat(out)


@torch.no_grad()
def evaluate_retrieval(
    model,
    dataset: Dataset,
    cache,
    ks: Sequence[int] = (1, 5, 10),
    device="cpu",
    batch_size: int = 128,
) -> Dict[str, float]:
    """
    Retrieval metrics against a gallery containing exactly the images present in ``dataset``.

    Reported groups
    ---------------
    trial_e2i_* : every single EEG trial queries the image gallery
    e2i_*       : EEG embeddings averaged over repetitions of each image query the gallery
                  (the standard THINGS-EEG2 protocol; 200-way for the test set)
    i2e_*       : each image queries the gallery of repetition-averaged EEG embeddings

    The gallery is built only from images of the evaluated split, so held-out (validation /
    test) images are never mixed with training images.
    """
    z_eeg, ids = embed_eeg_dataset(model, dataset, device, batch_size)
    unique_ids = sorted(set(ids))
    index_of = {sid: i for i, sid in enumerate(unique_ids)}
    z_img = embed_images(model, cache, unique_ids, device)  # (G, D)
    targets = torch.tensor([index_of[s] for s in ids])

    results: Dict[str, float] = {"n_trials": float(len(ids)), "n_images": float(len(unique_ids))}

    def add(prefix: str, sim: torch.Tensor, tgt: torch.Tensor) -> None:
        for k, v in retrieval_metrics(sim, tgt, ks).items():
            results[f"{prefix}_{k}"] = v

    add("trial_e2i", z_eeg @ z_img.t(), targets)

    avg = torch.zeros(len(unique_ids), z_eeg.size(1)).index_add_(0, targets, z_eeg)
    avg = F.normalize(avg, dim=-1)
    sim = avg @ z_img.t()
    arange = torch.arange(len(unique_ids))
    add("e2i", sim, arange)
    add("i2e", sim.t(), arange)
    return results
