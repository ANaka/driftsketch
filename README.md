# DriftSketch

Conditional Flow Matching (CFM) for vector sketch generation. A Transformer learns velocity fields that transform Gaussian noise into structured Bezier stroke sequences, conditioned on class labels or CLIP image embeddings.

![CFM flow](https://img.shields.io/badge/method-Conditional_Flow_Matching-blue)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-green)

## How It Works

DriftSketch learns to map noise to sketches by training on the straight-line interpolation between random noise and real sketches:

```
Training:  x_t = (1-t) * noise + t * sketch   →   model predicts velocity v = sketch - noise
Inference: start from noise, integrate velocity field via Euler ODE solver → sketch
```

The model operates on **32 cubic Bezier strokes**, each represented as 4 control points in 2D — a compact, resolution-independent representation that can be rendered to pixels or exported as SVG.

## Quick Start

```bash
# Install (editable mode)
uv pip install -e .

# With all optional dependencies
uv pip install -e ".[all]"

# Train on ControlSketch data
driftsketch-train --data-dir data/raw/controlsketch --epochs 100 --batch-size 64

# Generate sketches
driftsketch-generate --checkpoint checkpoints/model.pt --class-label 0 --num-samples 8

# Generate from an image (requires CLIP)
driftsketch-generate --checkpoint checkpoints/model.pt --class-label 0 --image photo.jpg --cfg-scale 3.0
```

## Installation

Requires Python 3.12+. Core dependencies: PyTorch, NumPy, Matplotlib.

```bash
# Core only
uv pip install -e .

# Optional dependency groups
uv pip install -e ".[wandb]"      # Weights & Biases logging
uv pip install -e ".[clip]"       # CLIP image conditioning (open_clip_torch)
uv pip install -e ".[render]"     # Differentiable rendering (pydiffvg)
uv pip install -e ".[lpips]"      # Perceptual loss
uv pip install -e ".[aesthetic]"  # LAION aesthetic scoring (open_clip_torch)
uv pip install -e ".[all]"       # Everything
```

## Datasets

### ControlSketch (primary)

Paired image + SVG Bezier sketches from [SwiftSketch](https://swiftsketch.github.io/). Each sample contains a JPEG image and 32 cubic Bezier strokes.

- Train: 15 categories, ~15k samples
- Validation: 15 categories, 3k samples
- Test: 85 categories, ~17k samples

Place data in `data/raw/controlsketch/{split}/{category}/*.npz`.

### QuickDraw (legacy)

Google [Quick, Draw!](https://quickdraw.withgoogle.com/data) simplified strokes, converted to fixed-length polylines.

```bash
python scripts/download_quickdraw.py --categories cat car house --max-per-category 5000
```

Place raw `.ndjson` files in `data/raw/quickdraw/`. Processed `.npy` caches are stored in `data/processed/`.

## Architecture

Two model variants are available, selectable via `--model-type`:

### `decoder` (default) — `BezierSketchTransformer`

TransformerDecoder with cross-attention to conditioning memory tokens.

```
Noisy strokes (B, 32, 8)
  → Linear projection + positional encoding
  → TransformerDecoder (8 layers, 256d, 8 heads)
      cross-attends to memory: [class tokens] or [CLIP patch features]
  → Output projection → predicted velocity (B, 32, 8)
```

### `dit` — `BezierSketchDiT`

Diffusion Transformer with adaptive layer norm (adaLN-Zero) conditioning. No cross-attention — conditioning is fused into a single vector that modulates every layer.

```
Noisy strokes (B, 32, 8)
  → Linear projection + positional encoding
  → Conditioning: time_emb + class_emb (or CLIP pooled)
  → N × DiTBlock (adaLN self-attention + SwiGLU MLP)
  → Final adaLN + output projection → predicted velocity (B, 32, 8)
```

Key features:
- adaLN-Zero initialization (gates start at zero for stability)
- SwiGLU feed-forward (parameter-efficient gating)
- Classifier-free guidance via learnable null conditioning

### CLIP Image Conditioning

When `--use-clip` is enabled, a frozen CLIP ViT-B-32 extracts patch-level features from input images. During training, conditioning is randomly dropped (`--p-uncond`) to enable classifier-free guidance at inference time.

## Training Modes

DriftSketch supports two training modes:

### Standard Training (velocity MSE)

Learns the CFM velocity field from paired noise/sketch data. Uses ControlSketch dataset.

### Perceptual Distillation (`--distill`)

Fine-tunes a pretrained model to generate sketches from arbitrary images using CLIP perceptual similarity, without requiring paired sketch data. Inspired by CLIPDraw.

```
Image → CLIP encoder → target features
Noise → few-step ODE integration → beziers → render → augment → CLIP encode → sketch features
Loss = 1 - cosine_similarity(sketch_features, target_features)
```

Key details:
- Requires a bootstrap checkpoint (`--checkpoint`), CLIP (`--use-clip`), and an image directory (`--distill-image-dir`)
- Uses few-step ODE integration (default 4 steps) WITH gradients flowing through the full generation path
- Renders generated Beziers to images, applies random augmentations (crop, rotate, flip) to prevent adversarial CLIP solutions
- Class labels are ignored (dummy zeros) — conditioning comes purely from CLIP features
- Fresh optimizer (does not restore optimizer state from checkpoint)
- Compatible with geometric and plotter regularization losses

## Training

```bash
# Basic training
driftsketch-train --data-dir data/raw/controlsketch --epochs 500 --batch-size 64

# DiT with full recipe
driftsketch-train \
  --data-dir data/raw/controlsketch \
  --model-type dit \
  --epochs 500 \
  --batch-size 64 \
  --lr 1e-4 \
  --weight-decay 0.05 \
  --embed-dim 256 \
  --num-layers 8 \
  --num-heads 8 \
  --save-every 50 \
  --sample-every 10

# With CLIP conditioning
driftsketch-train \
  --data-dir data/raw/controlsketch \
  --use-clip \
  --p-uncond 0.1 \
  --model-type dit \
  --epochs 500

# With auxiliary losses
driftsketch-train \
  --data-dir data/raw/controlsketch \
  --smoothness-loss-weight 0.01 \
  --degenerate-loss-weight 0.1 \
  --pixel-loss-weight 0.1 \
  --aesthetic-loss-weight 0.001

# Perceptual distillation (fine-tune on arbitrary images)
driftsketch-train \
  --distill \
  --checkpoint checkpoints/model.pt \
  --distill-image-dir /path/to/images \
  --use-clip \
  --distill-ode-steps 4 \
  --distill-augmentations 4 \
  --epochs 100 \
  --lr 1e-5

# Resume training from a checkpoint
driftsketch-train \
  --data-dir data/raw/controlsketch \
  --checkpoint checkpoints/model.pt \
  --epochs 500
```

### Training Recipe

| Feature | Flag | Default |
|---------|------|---------|
| Cosine LR schedule | `--warmup-epochs`, `--min-lr-factor` | 5% warmup, decay to 1% |
| Weight decay | `--weight-decay` | 0.05 (excluded from biases/norms/embeddings) |
| EMA | automatic | decay=0.9999, used for sampling |
| Gradient clipping | `--max-grad-norm` | 1.0 |
| Periodic checkpoints | `--save-every` | end-only (0) |

### Loss Functions

All auxiliary losses operate on the **one-step denoised estimate**: `x1_hat = x_t + (1-t) * v_pred`.

| Loss | Flag | Description |
|------|------|-------------|
| Velocity MSE | `--velocity-loss-weight` (1.0) | Core CFM loss |
| Smoothness | `--smoothness-loss-weight` | Penalizes sharp control-point angles |
| Degenerate | `--degenerate-loss-weight` | Prevents collapsed strokes |
| Coverage | `--coverage-loss-weight` | Encourages spatial spread |
| Pixel MSE | `--pixel-loss-weight` | Rendered image loss (requires `pydiffvg`) |
| LPIPS | `--lpips-loss-weight` | Perceptual similarity (requires `lpips`) |
| Aesthetic | `--aesthetic-loss-weight` | LAION aesthetic predictor score |
| Line density | `--density-loss-weight` | Prevents overlapping lines (plotter-friendly) |
| Curvature | `--curvature-loss-weight` | Limits maximum curvature |
| Total length | `--length-loss-weight` | Controls total drawing length |
| Uniformity | `--uniformity-loss-weight` | Spatial distribution of strokes |
| CLIP perceptual | `--distill` mode | Cosine similarity in CLIP space (distillation only) |

## Inference

```bash
# Basic generation
driftsketch-generate \
  --checkpoint checkpoints/model.pt \
  --class-label 0 \
  --num-samples 8 \
  --output outputs/generated.png

# Image-conditioned generation with CFG
driftsketch-generate \
  --checkpoint checkpoints/model.pt \
  --class-label 0 \
  --image photo.jpg \
  --cfg-scale 3.0 \
  --num-steps 50

# Export SVG
driftsketch-generate \
  --checkpoint checkpoints/model.pt \
  --class-label 0 \
  --num-samples 1 \
  --export-svg outputs/sketch.svg
```

The inference script auto-detects model type and architecture dimensions from the checkpoint. EMA weights are used when available.

## Project Structure

```
src/driftsketch/
  __init__.py              # Package exports
  model.py                 # VectorSketchTransformer, BezierSketchTransformer
  dit.py                   # BezierSketchDiT (adaLN-Zero architecture)
  train.py                 # Training loop with all loss functions
  inference.py             # Euler ODE generation, visualization, SVG export
  losses.py                # Geometric losses (smoothness, degenerate, coverage) + LPIPS
  aesthetic.py             # LAION aesthetic scorer (CLIP ViT-L/14 + MLP)
  plotter_losses.py        # Pen-plotter regularization (density, curvature, length)
  clip_encoder.py          # Frozen CLIP image encoder (patch features)
  perceptual.py            # CLIP perceptual loss for distillation
  rendering.py             # Differentiable Bezier rendering via pydiffvg
  ema.py                   # Exponential moving average
  data/
    __init__.py             # Data package exports
    dataset.py              # QuickDrawDataset
    controlsketch.py        # ControlSketchDataset + SVG parsing
    images.py               # ImageDataset (unlabeled images for distillation)
    processing.py           # Stroke → point-sequence conversion

scripts/
  download_quickdraw.py     # Stream QuickDraw data from Google Cloud Storage
  visualize_dataset.py      # QuickDraw grid visualization
  visualize_controlsketch.py # ControlSketch image+sketch visualization

docs/
  plans/                    # Design documents for features
  research/                 # Dataset evaluations and research notes
```

## Data Flow

```
               ControlSketch .npz files
                       │
            parse_svg_beziers()
                       │
                       ▼
              (32, 4, 2) Beziers          ─── native stroke representation
                       │
              beziers_to_points()          ─── optional polyline conversion
                       │
                       ▼
              (N, 2) point sequences       ─── for VectorSketchTransformer
```

```
               QuickDraw .ndjson
                       │
              strokes_to_points()          ─── arc-length resampling
                       │
                       ▼
              (N, 2) point sequences       ─── cached as .npy
```

## Checkpoints

Checkpoints are saved as dictionaries containing:

```python
{
    "model_state_dict": ...,
    "ema_state_dict": ...,       # EMA shadow weights
    "optimizer_state_dict": ...,
    "epoch": ...,
    "config": {
        "model_type": "decoder" | "dit",
        "num_classes": 15,
        "embed_dim": 256,
        "num_layers": 8,
        "num_heads": 8,
        "clip_dim": 0 | 768,
        ...
    }
}
```

## License

TBD
