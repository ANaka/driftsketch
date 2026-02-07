# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Learnings

When you make a mistake or learn a lesson during development, record it in MEMORY.md (auto-memory) for future reference.

## Project Overview

DriftSketch is a Conditional Flow Matching (CFM) model for vector sketch generation. It uses a Transformer architecture to learn velocity fields that transform Gaussian noise into structured 2D point sequences, conditioned on class labels. Currently trained on Google QuickDraw data (10 categories, up to 5000 samples each).

## Commands

```bash
# Install (editable, uses hatch build backend)
pip install -e .

# Train (uses wandb for logging)
driftsketch-train --epochs 1000 --batch-size 64 --lr 1e-4 --checkpoint-dir checkpoints

# Generate sketches from a checkpoint
driftsketch-generate --checkpoint checkpoints/model.pt --num-samples 8 --output outputs/generated.png

# Download QuickDraw raw data
python scripts/download_quickdraw.py --categories cat car house --max-per-category 5000

# Can also run as modules
python -m driftsketch.train
python -m driftsketch.inference
```

No test suite, linter, or formatter is configured yet.

## Architecture

### CFM Pipeline

The core flow: **noise (x0) -> interpolate with target (xt) -> predict velocity -> Euler integration -> sketch**

1. **Training (`train.py`):** Samples x0 ~ N(0,I) and x1 from data, interpolates xt = (1-t)*x0 + t*x1, trains model to predict velocity v = x1 - x0
2. **Inference (`inference.py`):** Starts from noise, integrates the learned velocity field via Euler ODE solver (50 steps, t=0 to t=1)

### Model (`model.py` - `VectorSketchTransformer`)

Token layout: `[time_token, class_token, point_1, ..., point_N]` (N+2 total tokens)

- Point embedding: Linear(2 -> 128) + learnable positional encoding
- Time embedding: sinusoidal features -> MLP with SiLU
- Class embedding: nn.Embedding -> MLP with SiLU
- Backbone: 6-layer TransformerEncoder (pre-LN, GELU, 4 heads)
- Output: extracts last N tokens (points only), LayerNorm -> Linear -> (B, N, 2)

### Data Pipeline (`data/`)

Two data paths exist:

1. **Synthetic (`train.py:generate_batch`):** Generates circles (label 0) and squares (label 1) procedurally. Used for initial testing.
2. **QuickDraw (`data/dataset.py:QuickDrawDataset`):** PyTorch Dataset loading real sketches from .ndjson files. Processes strokes to fixed-length point sequences via arc-length resampling (`data/processing.py:strokes_to_points`), normalizes to [-1, 1], caches as .npy files in `data/processed/`.

The QuickDraw dataset auto-discovers categories from `data/raw/quickdraw/*.ndjson` files if none specified. Label mapping is alphabetical by category name.

### Key Conventions

- All point data is shaped `(B, num_points, 2)` with values in [-1, 1]
- Time values are scalars in [0, 1]
- Class labels are integer tensors
- Default num_points=64, embed_dim=128
- Checkpoints saved as `model.state_dict()` dicts
- wandb is a hard dependency in the training loop (initialized unconditionally)
