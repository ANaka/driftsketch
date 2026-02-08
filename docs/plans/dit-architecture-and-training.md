# Beef Up Architecture & Training Recipe

## Context

First training run of BezierSketchTransformer on ControlSketch (15k samples, 15 classes) went 100 epochs with loss 0.561->0.438, still decreasing. The model works but has room to improve: no LR scheduler, no weight decay, no EMA, all auxiliary losses disabled, and the architecture uses a cross-attention decoder which is less modern than DiT-style adaLN conditioning. We'll add training recipe improvements AND a new DiT architecture option, keeping capacity CLI-configurable (defaults unchanged).

## Changes Overview

### Part 1: Training Recipe (in `train.py`)
1. **Cosine LR scheduler with warmup** -- linear warmup for 5% of epochs, then cosine decay to 1% of base LR
2. **Weight decay** -- AdamW with 0.05 weight decay, excluded from biases/norms/embeddings
3. **EMA** -- exponential moving average of model weights (decay=0.9999), used for sampling & saved in checkpoint
4. **Configurable architecture dims** -- `--embed-dim`, `--num-layers`, `--num-heads`, `--dropout` CLI args
5. **Periodic checkpointing** -- `--save-every N` saves checkpoints at intervals, not just at the end

### Part 2: DiT Architecture (new files + model.py)
6. **`src/driftsketch/dit.py`** -- new file containing:
   - `SwiGLUFeedForward` -- gated linear unit with SiLU activation (replaces GELU MLP)
   - `AdaLNModulation` -- projects conditioning vector to per-layer scale/shift/gate params
   - `DiTBlock` -- transformer block with adaLN-Zero: self-attention + SwiGLU MLP, both modulated by conditioning
   - `BezierSketchDiT` -- full model: stroke embedding + sinusoidal time + class MLP -> combined conditioning -> N DiT blocks -> output. Same forward signature as BezierSketchTransformer (x, t, class_label, clip_features, cfg_mask)
7. **`--model-type` flag** in `train.py` -- `"decoder"` (current, default) or `"dit"` (new)
8. **Update `inference.py`** -- support loading either model type from checkpoint config

## Detailed Design

### DiT Architecture (`BezierSketchDiT`)

**Key difference from current model:** Instead of cross-attending to memory tokens, conditioning (time + class) is fused into a single vector that modulates every layer via adaptive layer norm.

```
Input: (B, 32, 8) noisy Bezier strokes
  |
  +- stroke_proj: Linear(8 -> embed_dim) + pos_encoding
  |
  +- Conditioning:
  |    time: sinusoidal(t) -> MLP -> (B, cond_dim)
  |    class: Embedding -> MLP -> (B, cond_dim)
  |    cond = time + class  (or time + clip for image conditioning)
  |
  +- N x DiTBlock:
  |    +- adaLN modulation: cond -> Linear -> (shift, scale, gate) x 2
  |    +- norm1 -> modulate -> self-attention -> gate -> residual
  |    +- norm2 -> modulate -> SwiGLU MLP -> gate -> residual
  |
  +- Final adaLN + LayerNorm
  +- output_proj: Linear(embed_dim -> 8), zero-initialized
```

**adaLN-Zero initialization:** The final linear in each modulation MLP is zero-initialized, so at init the model behaves like identity (gates are zero). This is critical for stable training.

**SwiGLU:** `SiLU(x @ W_gate) * (x @ W_up)` then `@ W_down`. Hidden dim is `(8/3) * embed_dim` to match parameter count of 4x GELU MLP.

**CLIP compatibility:** When `clip_dim > 0`, CLIP features are pooled to a single vector via a projection MLP, then added to the conditioning vector (same as class embedding path). CFG uses a learnable null conditioning vector.

### EMA

Simple wrapper: shadow copy of parameters, updated each step with `shadow = decay * shadow + (1-decay) * param`. Used for sampling during training and saved in checkpoint alongside training weights.

### Forward Signature Compatibility

`BezierSketchDiT.forward(x, t, class_label, clip_features=None, cfg_mask=None)` -- identical to `BezierSketchTransformer.forward()` (minus unused `memory` param). Inference code works with either model.

## Files Created/Modified

| File | Action | What |
|------|--------|------|
| `docs/plans/dit-architecture-and-training.md` | **CREATE** | Design doc (this file) |
| `src/driftsketch/dit.py` | **CREATE** | SwiGLU, AdaLN, DiTBlock, BezierSketchDiT |
| `src/driftsketch/ema.py` | **CREATE** | EMA class |
| `src/driftsketch/train.py` | MODIFY | Add scheduler, weight decay, EMA, model-type flag, configurable dims, periodic saves |
| `src/driftsketch/inference.py` | MODIFY | Load either model type from checkpoint, support EMA weights |

**Not modified:** `model.py` (existing BezierSketchTransformer stays untouched), `losses.py`, `rendering.py`

## New CLI Arguments

```
# Training recipe
--weight-decay 0.05        # AdamW weight decay (default: 0.05)
--warmup-epochs N          # Linear warmup epochs (default: auto 5% of total)
--min-lr-factor 0.01       # Cosine decay min LR as fraction of base (default: 0.01)
--use-ema                  # Enable EMA (default: off)
--ema-decay 0.9999         # EMA decay rate
--save-every 50            # Checkpoint interval in epochs

# Architecture
--model-type decoder|dit   # Model architecture (default: decoder)
--embed-dim 256            # Embedding dimension
--num-layers 8             # Transformer layers
--num-heads 8              # Attention heads
--dropout 0.0              # Dropout rate
```

## Verification

1. **Smoke test (decoder + recipe):** `driftsketch-train --epochs 5 --batch-size 16 --use-ema --weight-decay 0.05` -- should train without errors, save checkpoint with EMA state, LR should warm up
2. **Smoke test (DiT):** `driftsketch-train --epochs 5 --batch-size 16 --model-type dit --use-ema` -- should train, generate samples, save checkpoint
3. **Inference compat:** `driftsketch-generate --checkpoint checkpoints/model.pt --class-label 0 --num-samples 4` -- should auto-detect model type from checkpoint and generate
4. **Full run:** `driftsketch-train --model-type dit --epochs 500 --use-ema --weight-decay 0.05 --smoothness-loss-weight 0.01 --degenerate-loss-weight 0.1 --save-every 50` -- compare against previous run in wandb
