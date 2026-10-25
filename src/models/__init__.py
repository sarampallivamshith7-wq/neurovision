"""EEG architectures and projection heads for NeuroVision."""

from src.models.aligner import NeuroVisionModel, build_model
from src.models.eeg_conformer import EEGConformerEncoder, TemporalTransformer
from src.models.eeg_encoder import ENCODER_TYPES, build_eeg_encoder
from src.models.eegnet_encoder import EEGNetEncoder
from src.models.projection_head import EEGProjectionHead
from src.models.spatial_gnn import GNNConformerEncoder, SpatialGraphEncoder, build_electrode_graph
from src.models.subject_adversary import GradientReversal, SubjectClassifier, dann_lambda

__all__ = [
    "EEGNetEncoder",
    "EEGConformerEncoder",
    "TemporalTransformer",
    "GNNConformerEncoder",
    "SpatialGraphEncoder",
    "build_electrode_graph",
    "EEGProjectionHead",
    "GradientReversal",
    "SubjectClassifier",
    "dann_lambda",
    "build_eeg_encoder",
    "ENCODER_TYPES",
    "NeuroVisionModel",
    "build_model",
]
