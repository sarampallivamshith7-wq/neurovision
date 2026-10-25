"""
Spatial electrode graph + GAT encoder (README Phase 4).

    (B, 63, T) EEG
        -> shared per-electrode temporal stem       (B, C, L, F)
        -> GAT over the electrode graph at each step (B, C, L, F)   [spatially enriched]
        -> electrode readout                         (B, L, d_model)
        -> Transformer (shared with EEGConformer)    -> (B, rep_dim)

The graph encoder is modular: ``conv_type`` selects GATv2 / GAT / GCN so variants can be
compared through configuration alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv

from src.models.eeg_conformer import TemporalTransformer


# --------------------------------------------------------------------------------------
# Electrode graph construction
# --------------------------------------------------------------------------------------
def get_electrode_positions(
    ch_names: Sequence[str],
    electrodes_tsv: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Return (n_channels, 3) electrode coordinates in the order of ``ch_names``.

    Uses a BIDS ``electrodes.tsv`` (columns name,x,y,z) when supplied and it covers every
    channel; otherwise falls back to MNE's built-in ``standard_1005`` montage.
    """
    if electrodes_tsv is not None and Path(electrodes_tsv).exists():
        import pandas as pd

        df = pd.read_csv(electrodes_tsv, sep="\t")
        if {"name", "x", "y", "z"}.issubset(df.columns):
            lookup = {r["name"]: (r["x"], r["y"], r["z"]) for _, r in df.iterrows()}
            if all(c in lookup for c in ch_names):
                pos = np.array([lookup[c] for c in ch_names], dtype=float)
                if np.isfinite(pos).all():
                    return pos

    import mne

    montage = mne.channels.make_standard_montage("standard_1005")
    ch_pos = montage.get_positions()["ch_pos"]
    missing = [c for c in ch_names if c not in ch_pos]
    if missing:
        raise ValueError(f"No electrode coordinates available for channels: {missing}")
    return np.array([ch_pos[c] for c in ch_names], dtype=float)


def build_electrode_graph(positions: np.ndarray, k: int = 6) -> torch.Tensor:
    """
    Undirected k-nearest-neighbour graph over electrode coordinates.

    Returns ``edge_index`` of shape (2, E) containing both directions of every edge and
    no self-loops (the GNN layers add their own). Every node has degree >= k.
    """
    n = len(positions)
    if not 1 <= k < n:
        raise ValueError(f"k must satisfy 1 <= k < n_nodes ({n}); got {k}.")
    d = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn_idx = np.argsort(d, axis=1)[:, :k]
    edges = set()
    for i in range(n):
        for j in nn_idx[i]:
            edges.add((i, int(j)))
            edges.add((int(j), i))
    edge_index = torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()
    return edge_index


# --------------------------------------------------------------------------------------
# Graph encoder
# --------------------------------------------------------------------------------------
class SpatialGraphEncoder(nn.Module):
    """
    Applies ``num_layers`` graph-conv layers over the electrode graph independently at
    every (sample, time-step). Input/Output: (B, C, L, F).
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        dim: int = 16,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        conv_type: str = "gatv2",
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.register_buffer("edge_index", edge_index.clone().long(), persistent=False)

        conv_type = conv_type.lower()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            if conv_type in ("gat", "gatv2"):
                if dim % heads != 0:
                    raise ValueError(f"node dim ({dim}) must be divisible by heads ({heads}).")
                cls = GATv2Conv if conv_type == "gatv2" else GATConv
                self.convs.append(cls(dim, dim // heads, heads=heads, dropout=dropout))
            elif conv_type == "gcn":
                self.convs.append(GCNConv(dim, dim))
            else:
                raise ValueError(f"Unknown conv_type '{conv_type}' (use gatv2 | gat | gcn).")
            self.norms.append(nn.LayerNorm(dim))
        self.dropout = nn.Dropout(dropout)

    def _batched_edge_index(self, n_graphs: int, device: torch.device) -> torch.Tensor:
        base = self.edge_index.to(device)  # (2, E)
        offsets = torch.arange(n_graphs, device=device) * self.num_nodes
        return (base.unsqueeze(1) + offsets.view(1, -1, 1)).reshape(2, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, length, f = x.shape
        if c != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} electrodes, got {c}.")
        h = x.permute(0, 2, 1, 3).reshape(b * length * c, f)  # graph-major node ordering
        ei = self._batched_edge_index(b * length, x.device)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + self.dropout(F.elu(conv(h, ei))))  # residual + norm
        return h.reshape(b, length, c, f).permute(0, 2, 1, 3)


class GNNConformerEncoder(nn.Module):
    """Spatial GAT + temporal Conformer EEG encoder. Input (B, C, T) -> (B, rep_dim)."""

    def __init__(
        self,
        in_channels: int = 63,
        in_samples: int = 225,
        edge_index: Optional[torch.Tensor] = None,
        ch_names: Optional[Sequence[str]] = None,
        electrodes_tsv: Optional[Union[str, Path]] = None,
        knn: int = 6,
        node_dim: int = 16,
        temporal_kernel: int = 25,
        pool_kernel: int = 15,
        pool_stride: int = 5,
        gnn_layers: int = 2,
        gnn_heads: int = 4,
        gnn_type: str = "gatv2",
        d_model: int = 64,
        depth: int = 3,
        num_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.25,
        rep_dim: int = 256,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.in_samples = in_samples
        self.rep_dim = rep_dim

        if edge_index is None:
            if ch_names is None:
                if in_channels != 63:
                    raise ValueError("Provide ch_names or edge_index when in_channels != 63.")
                from src.data.preprocessing import EXPECTED_63_CHANNELS

                ch_names = EXPECTED_63_CHANNELS
            if len(ch_names) != in_channels:
                raise ValueError(f"len(ch_names)={len(ch_names)} != in_channels={in_channels}")
            edge_index = build_electrode_graph(get_electrode_positions(ch_names, electrodes_tsv), k=knn)

        # Shared per-electrode temporal stem (same filters for every electrode).
        self.stem = nn.Sequential(
            nn.Conv1d(1, node_dim, temporal_kernel, padding=temporal_kernel // 2, bias=False),
            nn.BatchNorm1d(node_dim),
            nn.ELU(),
            nn.AvgPool1d(pool_kernel, pool_stride),
        )
        with torch.no_grad():
            n_tokens = self.stem(torch.zeros(1, 1, in_samples)).shape[-1]
        self.n_tokens = n_tokens

        self.graph = SpatialGraphEncoder(
            edge_index,
            num_nodes=in_channels,
            dim=node_dim,
            num_layers=gnn_layers,
            heads=gnn_heads,
            dropout=min(dropout, 0.2),
            conv_type=gnn_type,
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(in_channels * node_dim),
            nn.Linear(in_channels * node_dim, d_model),
            nn.Dropout(dropout),
        )
        self.transformer = TemporalTransformer(
            d_model=d_model,
            depth=depth,
            num_heads=num_heads,
            ff_mult=ff_mult,
            dropout=dropout,
            max_len=max(n_tokens, 8),
        )
        self.fc = nn.Sequential(nn.Linear(d_model, rep_dim), nn.LayerNorm(rep_dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.squeeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected (B, C, T) EEG tensor, got shape {tuple(x.shape)}")
        b, c, t = x.shape
        nodes = self.stem(x.reshape(b * c, 1, t))  # (B*C, F, L)
        nodes = nodes.reshape(b, c, nodes.shape[1], nodes.shape[2]).permute(0, 1, 3, 2)  # (B, C, L, F)
        nodes = self.graph(nodes)  # spatially enriched
        tokens = nodes.permute(0, 2, 1, 3).reshape(b, nodes.shape[2], -1)  # (B, L, C*F)
        tokens = self.transformer(self.readout(tokens))
        return self.fc(tokens.mean(dim=1))
