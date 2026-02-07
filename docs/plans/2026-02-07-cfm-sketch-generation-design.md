# Conditional Flow Matching for Vector Sketch Generation

## Overview

Implement a Conditional Flow Matching (CFM) model for vector sketch generation using a Transformer architecture. The model learns to transform Gaussian noise into structured point sequences (circles, squares) conditioned on class labels.

## Model Architecture

`VectorSketchTransformer` in `src/driftsketch/model.py`:

- **Point embedding:** `Linear(2 -> embed_dim)` + learnable positional encoding for N points
- **Time embedding:** sinusoidal features -> `MLP(dim -> embed_dim)` with SiLU activation
- **Class embedding:** `nn.Embedding(num_classes, cond_dim)` -> `MLP(cond_dim -> embed_dim)` with SiLU
- **Backbone:** `nn.TransformerEncoder` with pre-LN, GELU, 6 layers, 4 heads
- **Output head:** `LayerNorm` -> `Linear(embed_dim -> 2)`, applied only to point tokens

**Token layout:** `[time_token, class_token, point_1, ..., point_N]`

**Default dims:** `embed_dim=128`, `num_heads=4`, `num_layers=6`, `num_points=64`

## Training

`src/driftsketch/train.py` with `train()` entry point.

**Synthetic data generator** - `generate_batch(batch_size, num_points, device)` returns `(points, labels)`:

- Circle: evenly spaced angles around [0, 2pi], random scale (0.3-1.0), random center offset (-0.5 to 0.5), Gaussian noise (std=0.02)
- Square: evenly distributed points across 4 edges, same augmentations
- 50/50 split per batch, labels as integer tensor
- Points normalized to roughly [-1, 1]

**OT-CFM loop:**

1. Sample `x0 ~ N(0, I)` (noise), `x1` from generator (target)
2. Sample `t ~ U(0, 1)`
3. Interpolate: `xt = (1-t)*x0 + t*x1`
4. Target velocity: `v_target = x1 - x0`
5. Loss: `MSE(model(xt, t, label), v_target)`

**Config:** batch=64, lr=1e-4, AdamW, 1000 epochs, checkpoint to `checkpoints/model.pt`, print loss every 50 epochs.

**CLI flags:** `--epochs`, `--batch-size`, `--lr`, `--checkpoint-dir`

## Inference

`src/driftsketch/inference.py` with `main()` entry point.

**Euler ODE solver** - `generate(model, class_label, num_samples, num_steps=50, device)`:

1. Start from `x0 ~ N(0, I)` shaped `(num_samples, num_points, 2)`
2. Step size `dt = 1.0 / num_steps`
3. Each step: `x = x + model(x, t, class_label) * dt`
4. Return final `x` at `t=1`

**Visualization:** generate 8 circles + 8 squares, plot in 2-column figure, save to `outputs/generated.png`.

**CLI flags:** `--checkpoint`, `--num-samples`, `--output`

## Package Structure

```
src/driftsketch/
  __init__.py      # Export VectorSketchTransformer
  model.py         # VectorSketchTransformer class
  train.py         # generate_batch() + train() entry point
  inference.py     # generate() solver + plot_sketches() + main() entry point
  py.typed
```

**pyproject.toml:**
- Dependencies: `torch>=2.0.0`, `numpy>=1.24.0`, `matplotlib>=3.7.0`
- Entry points: `driftsketch-train = "driftsketch.train:train"`, `driftsketch-generate = "driftsketch.inference:main"`

**Public API:** `__init__.py` exports only `VectorSketchTransformer`.

**.gitignore:** `checkpoints/`, `outputs/`

## Extensibility

No extra code, just shaped for growth:

- `num_classes` parameter on model - add shapes by incrementing and adding a generator case
- `generate_batch` returns `(points, labels)` tuple - swappable for Dataset/DataLoader
- `generate()` solver takes model as input - swappable for RK4 or adaptive solvers
- Standard `model.state_dict()` checkpointing

## Verification

1. `pip install -e .`
2. `driftsketch-train` (or `python -m driftsketch.train`)
3. `driftsketch-generate` (or `python -m driftsketch.inference`)
4. Verify output shows distinct circle and square shapes
