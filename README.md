# NeuroVision

## Multimodal EEG–Vision System for Human Visual Intention / Object Recognition

NeuroVision is a research-oriented multimodal AI pipeline that learns a shared representation between **brain activity (EEG)** and the **visual stimulus presented to a participant**. The primary dataset is **THINGS-EEG2**, containing 63-channel EEG recordings collected while participants viewed natural images.

The system is designed in stages:

1. Load and validate THINGS-EEG2.
2. Preprocess raw EEG.
3. Convert the 63-electrode EEG into a spatial/graph representation.
4. Learn EEG representations with a **spatial graph neural network + temporal Transformer/Conformer**.
5. Extract complementary image representations using **DINOv2** and **CLIP ViT-B/32**.
6. Project EEG and image representations into a common embedding space.
7. Train the two modalities with **InfoNCE contrastive alignment**.
8. Add **subject-adversarial learning** to reduce subject-specific information.
9. Evaluate EEG→image and image→EEG retrieval and downstream visual recognition.
10. Extend the shared latent representation to retrieval, visual reconstruction and multimodal reasoning.

The repository is intentionally separated from the large EEG/image dataset. The `data/` directory is expected to contain the locally available THINGS-EEG2 data and is **not included in the project ZIP**.

---

# Current Implementation Status (prototype)

| README phase | Status | Where |
|---|---|---|
| 1. Data, preprocessing, pairing | Implemented, tested (synthetic + fake-BIDS); real THINGS-EEG2 **not yet run** | `src/data/`, `scripts/build_epochs.py` |
| 2. EEGNet baseline | Implemented | `src/models/eegnet_encoder.py` |
| 3. Conformer | Implemented | `src/models/eeg_conformer.py` |
| 4. Spatial GAT + Conformer | Implemented (GATv2 / GAT / GCN switchable) | `src/models/spatial_gnn.py` |
| 5. DINOv2 + CLIP feature caching | Implemented; pretrained weights **not yet downloaded/tested here** | `src/vision/`, `scripts/cache_features.py` |
| 6. Projection heads + InfoNCE | Implemented (false-negative masking for repeated images) | `src/models/`, `src/training/losses.py` |
| 7. Subject-adversarial (GRL) | Implemented; needs >= 2 subjects | `src/models/subject_adversary.py` |
| 8. Retrieval evaluation + FAISS | Implemented (Recall@K, MRR, median rank, both directions) | `src/evaluation/`, `scripts/evaluate.py` |
| ICA / bad-trial rejection (4.3) | **Not implemented** | - |
| Reconstruction, Brain-RAG, deployment | Deferred, per section 13 | - |

Changes made to the original code: fixed the broken `src.models` import, `get_image_path` return bug,
CLIP model-name (`ViT-B/32` -> open_clip `ViT-B-32`), added the missing `open_clip_torch` requirement,
made the vision package import lazily, made channel validation/ordering strict, made the dataset memmap-friendly.

## Quickstart

```bash
python -m pip install -r requirements.txt
python -m pytest -q                                   # 59+ tests; data/network tests auto-skip

# 1. Prototype run with NO dataset (synthetic EEG + image features)
python -m src.training.train_alignment --synthetic --epochs 10
python -m src.training.train_alignment --synthetic --encoder gnn_conformer --epochs 3

# 2. Real THINGS-EEG2 (data/ must exist locally)
python scripts/inspect_dataset.py
python scripts/build_epochs.py --config configs/phase1.yaml          # BDF -> data/processed/*.npy
python scripts/cache_features.py --config configs/phase1.yaml        # DINOv2 + CLIP -> data/embeddings
python -m src.training.train_alignment --config configs/phase1.yaml --encoder eegnet
python scripts/evaluate.py --checkpoint checkpoints/<run>/best.pt --topk-demo 5
```

Synthetic runs only validate the *pipeline*; they say nothing about real-brain performance.

---

# 1. Project Goal

The central research question is:

> Can EEG recorded while a person observes an image be transformed into a representation that is close to the representation of that image in a shared multimodal embedding space?

For a stimulus image `I` and its corresponding EEG segment `E`:

```text
EEG E ──► EEG Encoder ──► EEG Embedding ──┐
                                          ├──► Shared Embedding Space
Image I ─► Vision Encoder ─► Image Embedding┘
```

The model is trained so that the EEG embedding for an image is close to that image's embedding and far from embeddings belonging to other images.

This enables:

- EEG-to-image retrieval
- image-to-EEG retrieval
- visual object/category recognition from EEG
- cross-modal representation learning
- later EEG-conditioned visual reconstruction
- later multimodal reasoning over retrieved/generated visual evidence

---

# 2. Dataset: THINGS-EEG2

NeuroVision uses THINGS-EEG2 as the primary EEG dataset.

Important characteristics used by the pipeline:

- 10 participants/subjects
- 63 EEG channels
- EEG sampling rate: 1000 Hz
- approximately 16,540 distinct training images
- 200 test images
- rapid serial visual presentation at approximately 5 Hz
- multiple recording sessions per subject
- event metadata identifying the presented image/stimulus

The project should not assume that the entire dataset must be loaded into RAM.

The intended implementation is:

```text
Disk
  │
  ▼
Event metadata
  │
  ▼
Stimulus ↔ EEG trial pairing
  │
  ▼
Lazy dataset / DataLoader
  │
  ▼
Preprocessing
  │
  ▼
Model
```

Only the required subject/session/trial data should be loaded for each experiment.

---

# 3. End-to-End Architecture

The planned architecture is:

```text
                         THINGS-EEG2
                              │
             ┌────────────────┴────────────────┐
             │                                 │
             ▼                                 ▼
       63-channel EEG                    Stimulus Image
             │                                 │
             ▼                                 ▼
     EEG preprocessing                 Image preprocessing
             │                                 │
     Band-pass filtering                     │
     Artifact correction                     │
     Baseline correction                     │
     Epoch extraction                        │
     Downsampling 1000→250 Hz                │
             │                                 │
             ▼                                 ▼
     Spatial electrode graph          ┌────────┴────────┐
             │                         │                 │
             ▼                         ▼                 ▼
        Graph encoder              DINOv2           CLIP ViT-B/32
             │                    structural          semantic
             │                    features            features
             ▼                         │                 │
      Temporal Conformer                └───────┬─────────┘
             │                                 │
             ▼                                 ▼
       EEG projection                  Vision projection
             │                                 │
             ▼                                 ▼
        EEG embedding                   Image embedding
             │                                 │
             └──────────────┬──────────────────┘
                            ▼
                   Shared embedding space
                            │
                     InfoNCE alignment
                            │
             ┌──────────────┴──────────────┐
             ▼                             ▼
      EEG→Image retrieval            Image→EEG retrieval
             │
             ▼
      Downstream recognition
             │
             ▼
      Future extensions:
      FAISS retrieval → visual reconstruction
      → multimodal reasoning / Brain-RAG
```

---

# 4. EEG Preprocessing

The preprocessing pipeline will transform the raw THINGS-EEG2 recordings into stable, model-ready EEG epochs.

## 4.1 Channel handling

The original recording contains 63 EEG channels.

The implementation should preserve the channel identity/order consistently across:

- sessions
- subjects
- trials
- preprocessing
- graph construction

Channel names and electrode locations should be used whenever the dataset metadata provides them.

## 4.2 Filtering

A configurable band-pass filter will remove slow drifts and high-frequency noise.

The exact filter limits should remain configurable through the YAML configuration rather than hard-coded throughout the code.

## 4.3 Artifact correction

Artifact handling should be implemented as a reproducible preprocessing stage.

Possible artifacts include:

- eye movements
- blinks
- muscle activity
- line noise
- bad channels
- abnormal trials

ICA can be used where appropriate for separating ocular/muscle components. The implementation should keep artifact removal configurable because ICA decisions can affect downstream EEG representations.

## 4.4 Baseline correction

Stimulus-locked epochs should support baseline correction using an appropriate pre-stimulus interval when available.

The implementation should not invent a baseline interval when the event structure does not provide one.

## 4.5 Epoch extraction

Continuous EEG is converted into stimulus-aligned epochs.

The event parser will use the THINGS-EEG2 event metadata to associate:

```text
EEG segment ↔ image identifier ↔ trial metadata
```

The pairing must be deterministic and testable.

## 4.6 Downsampling

The raw sampling rate is 1000 Hz.

The planned model input rate is:

```text
1000 Hz → 250 Hz
```

Downsampling reduces the temporal dimensionality by a factor of four while retaining the frequency content needed by the intended EEG representation.

Anti-alias filtering/resampling must be performed correctly before or as part of downsampling.

---

# 5. EEG Representation

The EEG branch is intentionally more expressive than a simple CNN.

The planned encoder combines:

1. spatial graph modeling
2. temporal sequence modeling

## 5.1 Electrode graph

The 63 electrodes are represented as graph nodes.

Each node contains a temporal EEG feature sequence.

Graph edges represent spatial relationships between electrodes.

A graph can be constructed using:

- known electrode coordinates
- neighborhood relationships
- distance-based connectivity

The graph is intended to model the fact that nearby/related electrodes can contain correlated neural information.

Conceptually:

```text
Electrodes
   │
   ▼
Spatial Graph
   │
   ▼
GNN / GAT / ST-GNN
   │
   ▼
Spatially enriched temporal features
```

## 5.2 Graph Attention / Spatial GNN

The planned architecture uses a graph-based spatial encoder, with **Graph Attention Network (GAT)** or a closely related spatial-temporal GNN implementation.

Attention allows the network to learn which neighboring electrode relationships are useful for a particular representation.

The implementation should keep the graph encoder modular so that GAT/GNN variants can be compared.

## 5.3 EEG Conformer

After spatial feature extraction, the temporal dimension is modeled using an EEG Conformer-style Transformer.

The Conformer combines convolutional/local feature extraction with self-attention.

Self-attention is useful because EEG patterns can contain relationships across non-adjacent time points.

The planned sequence is:

```text
EEG
 ↓
Spatial GNN/GAT
 ↓
Temporal feature sequence
 ↓
Conformer
 ↓
Temporal aggregation
 ↓
EEG representation
```

## 5.4 EEGNet baseline

EEGNet will be retained as a lightweight baseline.

It is not the final intended architecture.

It is useful for:

- validating preprocessing
- establishing a baseline
- comparing lightweight vs. graph/Transformer architectures
- debugging the EEG branch

The baseline should be implemented so that experiments can switch encoders through configuration.

---

# 6. Vision Branch

The vision branch uses pretrained models rather than training a large image model from scratch.

Two complementary representations are planned.

## 6.1 DINOv2

DINOv2 provides a strong visual representation learned through self-supervised learning.

Its features are useful for capturing visual structure and object-level information.

The planned use is:

```text
Image
 ↓
DINOv2
 ↓
Structural visual embedding
 ↓
Projection head
```

DINOv2 will initially be frozen.

## 6.2 CLIP ViT-B/32

CLIP provides an image representation learned through image-text contrastive learning.

CLIP is useful for semantic information.

The planned use is:

```text
Image
 ↓
CLIP ViT-B/32 image encoder
 ↓
Semantic visual embedding
 ↓
Projection head
```

The CLIP encoder will initially be frozen.

## 6.3 Multi-scale/complementary vision representation

DINOv2 and CLIP are not treated as identical models.

The purpose is to combine:

```text
DINOv2 → visual structure
CLIP   → semantic representation
```

Their outputs can be projected into a common vision representation before multimodal alignment.

This creates a richer target representation for EEG.

---

# 7. Projection Heads

Raw EEG and vision features generally have different dimensions.

Projection heads map them into a common dimensionality.

```text
EEG encoder
    ↓
EEG projection head
    ↓
z_eeg ∈ R^D

Vision encoder
    ↓
Vision projection head
    ↓
z_img ∈ R^D
```

The projected vectors are normalized before contrastive learning.

The shared dimensionality `D` should be configurable.

---

# 8. Contrastive Learning with InfoNCE

The main multimodal alignment objective is **InfoNCE**.

For a batch:

```text
EEG embeddings:   E1 E2 E3 ... EN
Image embeddings: I1 I2 I3 ... IN
```

The correct pairs are:

```text
E1 ↔ I1
E2 ↔ I2
E3 ↔ I3
...
EN ↔ IN
```

The model should maximize similarity for matching pairs and minimize similarity for mismatched pairs.

A symmetric loss is planned:

```text
EEG → Image InfoNCE
+
Image → EEG InfoNCE
```

Conceptually:

```text
Similarity matrix

        I1    I2    I3    I4
E1     +     -     -     -
E2     -     +     -     -
E3     -     -     +     -
E4     -     -     -     +
```

Temperature scaling will be configurable.

This alignment is the central training objective for the shared EEG–vision space.

---

# 9. Subject-Invariant Representation

EEG is strongly affected by subject-specific characteristics.

The project therefore plans an optional adversarial subject-invariance component.

Architecture:

```text
EEG embedding
      │
      ├──────────────► EEG-image alignment
      │
      └──► Gradient Reversal Layer
                    │
                    ▼
             Subject classifier
```

The subject classifier attempts to predict the participant.

The gradient reversal mechanism causes the EEG encoder to learn features that are less useful for identifying the subject while remaining useful for EEG–image alignment.

This component should be introduced after the basic alignment pipeline works.

---

# 10. Training Strategy

The implementation should proceed incrementally rather than training the entire architecture at once.

## Phase 1 — Data and preprocessing

Implement and validate:

- dataset indexing
- event parsing
- EEG/image pairing
- filtering
- artifact handling
- baseline correction
- epoch extraction
- 1000→250 Hz resampling
- PyTorch Dataset/DataLoader

## Phase 2 — EEG baseline

Implement:

```text
EEG → EEGNet → EEG embedding
```

Validate the complete EEG pipeline.

## Phase 3 — Temporal Transformer baseline

Implement:

```text
EEG → Conformer → EEG embedding
```

Compare against EEGNet.

## Phase 4 — Spatial + temporal EEG model

Implement:

```text
EEG
 ↓
63-electrode graph
 ↓
GAT / spatial GNN
 ↓
Conformer
 ↓
EEG embedding
```

## Phase 5 — Vision features

Implement cached pretrained features:

```text
Image → DINOv2
Image → CLIP ViT-B/32
```

Caching is important because repeatedly running large vision encoders during every EEG training step is inefficient.

## Phase 6 — Cross-modal alignment

Implement:

```text
EEG encoder → projection
Vision encoder → projection
                 ↓
              InfoNCE
```

## Phase 7 — Subject invariance

Add:

```text
Gradient Reversal + Subject Classifier
```

after the basic multimodal system is stable.

## Phase 8 — Evaluation

Evaluate:

- EEG→image retrieval
- image→EEG retrieval
- Recall@K
- mean reciprocal rank where appropriate
- embedding similarity
- subject-generalization experiments
- downstream image/category recognition

---

# 11. Evaluation

The primary evaluation is retrieval in the shared embedding space.

Given an EEG segment:

```text
EEG
 ↓
EEG encoder
 ↓
EEG embedding
 ↓
similarity against image embeddings
 ↓
Top-K retrieved images
```

Useful metrics include:

- Recall@1
- Recall@5
- Recall@10
- Median rank
- Mean reciprocal rank
- cosine similarity

The evaluation must ensure that train/test leakage does not occur.

For test images that are intentionally held out, their image embeddings must be handled according to the experimental protocol.

---

# 12. Future Extensions

Once EEG–vision alignment works, the shared representation can support more advanced components.

## 12.1 FAISS retrieval

A FAISS index can store image embeddings.

```text
EEG
 ↓
EEG embedding
 ↓
FAISS nearest-neighbor search
 ↓
Top-K visual candidates
```

This makes large-scale EEG-to-image retrieval efficient.

## 12.2 EEG-conditioned visual reconstruction

An EEG adapter can transform the EEG representation into a conditioning representation for a latent diffusion / SDXL-style image generation system.

Conceptually:

```text
EEG
 ↓
EEG encoder
 ↓
EEG adapter
 ↓
latent conditioning
 ↓
diffusion model
 ↓
reconstructed visual content
```

This is a later-stage research extension and is not required for the first working version.

## 12.3 Multimodal reasoning / Brain-RAG

Retrieved or generated visual evidence can later be supplied to a multimodal reasoning system.

Conceptually:

```text
EEG
 ↓
EEG embedding
 ↓
Visual retrieval
 ↓
Retrieved visual evidence
 ↓
Multimodal reasoning
```

The goal is to allow the system to produce interpretable semantic information from neural representations.

## 12.4 Deployment interfaces

The final system can expose inference through:

- REST
- WebSocket
- MCP-compatible interfaces

These are deployment concerns and should be implemented after the research pipeline is validated.

---

# 13. Components Explicitly Deferred

Some technologies were considered but are not part of the first implementation.

### Frequency Mamba

Frequency-domain Mamba-style modeling is reserved for an ablation/experimental branch.

It should not complicate the initial pipeline.

### ControlNet

ControlNet is not required for the initial visual reconstruction stage.

The first reconstruction experiments should focus on the EEG-to-conditioning interface.

### Large multimodal LLM integration

LLM/multimodal-LLM integration is not the core EEG representation-learning contribution.

It should be treated as a later reasoning layer rather than replacing the EEG–vision alignment model.

---

# 14. Repository Structure

Current repository:

```text
NeuroVisionProject/
│
├── configs/
│   └── phase1.yaml
│
├── scripts/
│   ├── inspect_dataset.py
│   ├── build_epochs.py        # BDF -> processed epochs
│   ├── cache_features.py      # DINOv2 + CLIP cache
│   └── evaluate.py            # checkpoint -> retrieval metrics / FAISS top-K
│
├── src/
│   ├── data/
│   │   ├── eeg_dataset.py
│   │   ├── event_parser.py
│   │   ├── preprocessing.py
│   │   └── synthetic.py
│   │
│   ├── models/   # eegnet_encoder, eeg_conformer, spatial_gnn, eeg_encoder (factory),
│   │             # projection_head, subject_adversary, aligner
│   ├── training/ # losses.py (InfoNCE), train_alignment.py
│   ├── evaluation/ # metrics.py, retrieval.py, faiss_index.py
│   │
│   ├── utils/
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── seed.py
│   │
│   └── vision/
│       ├── cache_embeddings.py
│       ├── clip_features.py
│       ├── dinov2_features.py
│       └── image_projection.py
│
└── tests/
    ├── test_eeg_preprocessing.py
    ├── test_pairing.py
    └── test_vision_pipeline.py
```

The repository should evolve toward:

```text
src/
├── data/
├── models/
│   ├── eegnet.py
│   ├── eeg_conformer.py
│   ├── spatial_gnn.py
│   ├── eeg_encoder.py
│   ├── projection_head.py
│   └── subject_adversary.py
│
├── training/
│   ├── train_eeg.py
│   └── train_alignment.py
│
├── evaluation/
│   ├── retrieval.py
│   └── metrics.py
│
├── vision/
└── utils/
```

---

# 15. Engineering Requirements

The implementation should follow these principles:

- PyTorch-based model implementation.
- PyTorch Geometric for graph neural network components.
- MNE-Python for EEG-oriented preprocessing where appropriate.
- Hugging Face Transformers for compatible pretrained vision models where appropriate.
- Reproducible random seeds.
- YAML-based configuration.
- Modular Dataset/DataLoader implementation.
- No hard-coded local dataset paths.
- No loading the entire THINGS-EEG2 dataset into memory.
- Cache expensive DINOv2/CLIP features.
- Unit-test preprocessing and trial pairing.
- Keep model components independently testable.
- Log training loss and evaluation metrics.
- Save model checkpoints and configuration used for each experiment.
- Make experiments reproducible from configuration files.

---

# 16. Expected Final System

The target research pipeline is:

```text
                 THINGS-EEG2
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
      Raw EEG                  Image
          │                       │
          ▼                       ▼
  EEG preprocessing        Image preprocessing
          │                       │
          ▼                 ┌─────┴─────┐
   63-channel graph         ▼           ▼
          │               DINOv2      CLIP
          ▼                 │           │
     Spatial GNN             └─────┬─────┘
          │                         │
          ▼                         ▼
      Conformer              Vision projection
          │                         │
          ▼                         ▼
    EEG projection            Image embedding
          │                         │
          ▼                         │
      EEG embedding                 │
          │                         │
          └──────────┬──────────────┘
                     ▼
              InfoNCE Alignment
                     │
                     ▼
             Shared latent space
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Retrieval  Recognition  Future
                              Reconstruction
                                   │
                                   ▼
                           Multimodal reasoning
```

The first milestone is **not** image generation. The first milestone is a correctly paired, reproducible EEG preprocessing pipeline followed by a working EEG encoder and measurable EEG–image contrastive alignment.

---

# 17. Important Development Rule

Do not implement every advanced component simultaneously.

The system should be built and tested in this order:

```text
Data
 ↓
Preprocessing
 ↓
EEGNet baseline
 ↓
Conformer baseline
 ↓
Spatial GNN + Conformer
 ↓
DINOv2 + CLIP feature extraction
 ↓
Projection heads
 ↓
InfoNCE
 ↓
Retrieval evaluation
 ↓
Subject adversarial learning
 ↓
FAISS
 ↓
Visual reconstruction
 ↓
Multimodal reasoning
```

At every stage, existing tests should continue to pass.

The codebase should favor **correctness, reproducibility and modular experiments** over prematurely adding large models or complex deployment components.
