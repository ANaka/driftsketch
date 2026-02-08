# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Learnings

When you make a mistake or learn a lesson during development, record it in MEMORY.md (auto-memory) for future reference.

## Project Overview

DriftSketch is a Conditional Flow Matching (CFM) model for vector sketch generation. It uses Transformer architectures (decoder or DiT) to learn velocity fields that transform Gaussian noise into 32 cubic Bezier strokes, conditioned on class labels or CLIP image embeddings. Primarily trained on ControlSketch (SwiftSketch) data.

## Commands

```bash
# Install (editable, uses hatch build backend)
uv pip install -e .

# Install with all optional deps
uv pip install -e ".[all]"

# Train on ControlSketch data
driftsketch-train --data-dir data/raw/controlsketch --epochs 500 --batch-size 64

# Train with DiT architecture
driftsketch-train --data-dir data/raw/controlsketch --model-type dit --epochs 500

# Train with CLIP conditioning
driftsketch-train --data-dir data/raw/controlsketch --use-clip --p-uncond 0.1

# Perceptual distillation (fine-tune on arbitrary images, no paired sketches needed)
driftsketch-train --distill --checkpoint checkpoints/model.pt --distill-image-dir /path/to/images --use-clip

# Resume training from a checkpoint
driftsketch-train --data-dir data/raw/controlsketch --checkpoint checkpoints/model.pt --epochs 500

# Generate sketches from a checkpoint
driftsketch-generate --checkpoint checkpoints/model.pt --class-label 0 --num-samples 8

# Generate from image with classifier-free guidance
driftsketch-generate --checkpoint checkpoints/model.pt --class-label 0 --image photo.jpg --cfg-scale 3.0

# Export SVG
driftsketch-generate --checkpoint checkpoints/model.pt --class-label 0 --export-svg output.svg

# Download QuickDraw raw data (legacy)
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

### Models

Two architectures are available (`--model-type`):

- **`decoder`** (default) — `BezierSketchTransformer` in `model.py`: TransformerDecoder with cross-attention to conditioning memory tokens (class or CLIP). 256d, 8 heads, 8 layers.
- **`dit`** — `BezierSketchDiT` in `dit.py`: adaLN-Zero Diffusion Transformer. Conditioning fused into a single vector that modulates every layer via adaptive layer norm. SwiGLU feed-forward, zero-initialized gates.

Both operate on flattened Bezier strokes (B, 32, 8) and share the same forward signature.

Legacy: `VectorSketchTransformer` in `model.py` operates on (B, N, 2) point sequences with token layout `[time_token, class_token, point_1, ..., point_N]`.

### Training Modes

1. **Standard** (default): Velocity MSE on ControlSketch paired data.
2. **Distillation** (`--distill`): Fine-tunes pretrained model on arbitrary images via CLIP perceptual loss. Renders generated Beziers, augments (CLIPDraw-style), computes cosine similarity in CLIP space. Requires `--checkpoint`, `--use-clip`, `--distill-image-dir`. Fresh optimizer, few-step ODE with gradients.

### Data Pipeline (`data/`)

- **ControlSketch** (`data/controlsketch.py`): Primary dataset. Loads paired image + SVG Bezier sketches from .npz files. Native format: (32, 4, 2) control points. Supports train/val/test splits.
- **ImageDataset** (`data/images.py`): Unlabeled image directory for distillation. Recursively finds JPG/PNG files.
- **QuickDraw** (`data/dataset.py`): Legacy. Loads .ndjson files, processes strokes to fixed-length polylines via arc-length resampling, caches as .npy.
- **Processing** (`data/processing.py`): `strokes_to_points()` with arc-length resampling and normalization to [-1, 1].

### Loss Functions

All auxiliary losses operate on the one-step denoised estimate: `x1_hat = x_t + (1-t) * v_pred`.

- **Core** (`train.py`): velocity MSE
- **Geometric** (`losses.py`): smoothness, degenerate stroke penalty, coverage uniformity
- **Perceptual** (`losses.py`): LPIPS
- **Pixel** (`rendering.py`): differentiable rendering via pydiffvg, pixel MSE against target
- **Aesthetic** (`aesthetic.py`): LAION aesthetic predictor (CLIP ViT-L/14 + MLP)
- **Plotter** (`plotter_losses.py`): line density, curvature, total length, spatial uniformity
- **CLIP perceptual** (`perceptual.py`): render + augment + CLIP cosine similarity (distillation mode)

### Key Modules

| Module | Purpose |
|--------|---------|
| `model.py` | VectorSketchTransformer, BezierSketchTransformer, CLIPImageProjector |
| `dit.py` | BezierSketchDiT, DiTBlock, AdaLNModulation, SwiGLUFeedForward |
| `train.py` | Training loop (standard + distillation), all losses, scheduler, EMA |
| `perceptual.py` | CLIPPerceptualLoss (render → augment → CLIP cosine similarity) |
| `inference.py` | Euler ODE generation, visualization, SVG export |
| `clip_encoder.py` | Frozen CLIP ViT-B-32 patch-level feature extraction |
| `rendering.py` | Differentiable Bezier → raster via pydiffvg |
| `ema.py` | Exponential moving average of model weights |

### Key Conventions

- Bezier data shaped `(B, 32, 4, 2)` or flattened `(B, 32, 8)`, values in [-1, 1]
- Point data shaped `(B, num_points, 2)` with values in [-1, 1]
- Time values are scalars in [0, 1]
- Class labels are integer tensors
- Checkpoints are dicts with `model_state_dict`, `ema_state_dict`, `config`; can be loaded to resume training or bootstrap distillation
- wandb is optional (`--no-wandb` flag)
- Training uses AdamW with cosine LR schedule (linear warmup) and EMA
- Always use `uv` for package management, never bare `pip`
