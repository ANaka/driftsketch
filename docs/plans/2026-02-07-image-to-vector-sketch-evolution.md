# DriftSketch: Image-to-Vector-Sketch Evolution Plan

## Context

DriftSketch is currently a class-conditional CFM model generating flat 2D polylines from QuickDraw data. Based on the SwiftSketch paper (arxiv 2502.08642) and related work survey, we're evolving it into an image-conditioned Bezier sketch generator. The changes are split into three incremental phases, each independently trainable and testable. Renderer: easydiffvg (pure PyTorch, at `/home/naka/code/easydiffvg`).

### Current → Target

| Aspect | Current | Target |
|---|---|---|
| Output representation | (B, 64, 2) flat polyline | (B, 32, 8) Bezier strokes (32 cubic Beziers, 4 ctrl pts × 2 coords) |
| Architecture | TransformerEncoder (self-attn only) | TransformerDecoder (self-attn + cross-attn) |
| Conditioning | Class label embedding | CLIP image features via cross-attention |
| Loss | MSE on velocity field | Velocity MSE + pixel-space rendering loss |
| Output format | Scatter plot PNG | SVG via easydiffvg |

### Key Existing Asset

`ControlSketchDataset` (`src/driftsketch/data/controlsketch.py`) already provides exactly the target format:
- `beziers`: (32, 4, 2) cubic Bezier control points in [-1, 1]
- `points`: (num_points, 2) resampled polyline (backward compatible)
- `label`, `caption`, and optionally `image_bytes`, `attn_map`, `mask`

---

## Phase 1: Bezier Strokes + Decoder Architecture

**Goal:** Predict Bezier control points with a TransformerDecoder, still class-conditional.

### 1.1 New Model: `BezierSketchTransformer`

**File:** `src/driftsketch/model.py` — add new class alongside existing `VectorSketchTransformer` (keep for backward compat)

```
BezierSketchTransformer(nn.Module):
  num_strokes=32, coords_per_stroke=8, embed_dim=256, num_heads=8, num_layers=8
  num_classes=10, num_memory_tokens=4

  stroke_proj:    Linear(8 → embed_dim)           # embed each stroke's 8 coords
  pos_encoding:   Parameter(1, 32, embed_dim)      # learnable per-stroke position
  time_mlp:       sinusoidal → MLP(embed_dim)      # same approach as current
  class_embed:    Embedding(num_classes, 128)       # class conditioning
  class_to_memory: Linear(128 → embed_dim×4) + SiLU  # expand to 4 memory tokens
  transformer:    TransformerDecoder(pre-LN, GELU, 8 layers, 8 heads, ff=4×embed_dim)
  output_proj:    LayerNorm → Linear(embed_dim → 8)

  forward(x: (B,32,8), t: (B,), class_label: (B,), memory: optional) → (B,32,8)
    queries: [time_emb, stroke_embs] → (B, 33, embed_dim)
    memory:  class_to_memory(class_embed(label)) → (B, 4, embed_dim)
    out:     transformer(tgt=queries, memory=memory)
    return:  output_proj(out[:, 1:, :])  # skip time token → (B, 32, 8)
```

**Key design choices:**
- 256/8 (wider + more heads) because Bezier strokes are structurally richer than raw points
- `num_memory_tokens=4` avoids degenerate cross-attention from a single class vector; mirrors CLIP patches in Phase 2
- `memory` parameter in forward() lets Phase 2 inject CLIP features without changing the signature

### 1.2 Training Loop

**File:** `src/driftsketch/train.py` — rewrite

- Replace `generate_batch()` with `ControlSketchDataset` + `DataLoader`
- Flatten beziers: `x1 = batch["beziers"].view(B, 32, 8)`
- CFM loop unchanged in structure, just operates on (B, 32, 8):
  ```
  x0 ~ N(0,I) shape (B, 32, 8)
  t ~ U(0,1) shape (B, 1, 1)
  xt = (1-t)*x0 + t*x1
  v_target = x1 - x0
  v_pred = model(xt, t.squeeze(), labels)
  loss = MSE(v_pred, v_target)
  ```
- Make wandb optional (`--no-wandb` flag)
- Add `--data-dir`, `--categories` args
- Save richer checkpoints: `{model_state_dict, optimizer_state_dict, epoch, config}`

### 1.3 Inference

**File:** `src/driftsketch/inference.py` — rewrite

- `generate_beziers()`: Euler ODE on (B, 32, 8) → reshape to (B, 32, 4, 2)
- `plot_bezier_sketches()`: evaluate cubic Bezier curves at many t values, plot as line segments (not scatter)
- `beziers_to_pydiffvg_shapes()`: convert (32, 4, 2) in [-1,1] → list of `pydiffvg.Path` objects
  - Each stroke = independent Path with `num_control_points=torch.tensor([2])`, `points=(4,2)`, `is_closed=False`
  - Coordinate mapping: `canvas = (normalized + 1) / 2 * canvas_size`
- `export_svg()`: `pydiffvg.save_svg(filename, w, h, shapes, groups)`
- CLI: `--checkpoint`, `--class-label`, `--num-samples`, `--output`, `--export-svg`

### 1.4 Package Updates

**`src/driftsketch/__init__.py`**: Export `BezierSketchTransformer`

**`pyproject.toml`**:
- Move `wandb` to optional dependencies
- Add optional groups: `render = ["pydiffvg @ file:///home/naka/code/easydiffvg"]`, `clip = ["open_clip_torch>=2.20.0"]`

### 1.5 Verification

1. Model forward: `BezierSketchTransformer()(rand(4,32,8), rand(4), randint(0,10,(4,)))` → shape (4,32,8)
2. Data loading: `ControlSketchDataset()` → batch `beziers` has shape (32, 4, 2)
3. Training smoke: 100 iterations, loss decreases
4. Inference: generate 4 samples, matplotlib shows recognizable curves
5. SVG export: open in browser, verify valid vector output

---

## Phase 2: CLIP Image Conditioning

**Goal:** Condition on input images via frozen CLIP encoder + cross-attention. Add classifier-free guidance.

### 2.1 CLIP Encoder

**New file:** `src/driftsketch/clip_encoder.py`

```
FrozenCLIPImageEncoder(nn.Module):
  Uses open_clip ViT-B-32
  Extracts patch-level features: (B, 50, 768) for 224×224 input
    (1 CLS + 49 patches on 7×7 grid)
  All parameters frozen (no_grad)
```

### 2.2 Model Changes

**File:** `src/driftsketch/model.py` — add to `BezierSketchTransformer`

- `CLIPImageProjector`: LayerNorm → Linear(768 → 256) → GELU → Linear(256 → 256)
- `null_memory`: learnable Parameter(1, 4, embed_dim) for unconditional generation / CFG
- `forward()` gains `clip_features: (B, L, 768) | None` and `cfg_mask: (B,) bool | None`
  - If `clip_features` provided: project to memory via CLIPImageProjector
  - If `cfg_mask[i]` is True: replace that sample's memory with `null_memory` (drop conditioning)
  - Falls back to class conditioning or unconditional if no clip_features

### 2.3 Training Changes

**File:** `src/driftsketch/train.py`

- Load frozen CLIP encoder, run on batch images (`return_images=True` in dataset)
- CFG: `cfg_mask = torch.rand(B) < p_uncond` (default p_uncond=0.1)
- Pass `clip_features` and `cfg_mask` to model forward

### 2.4 Dataset Changes

**File:** `src/driftsketch/data/controlsketch.py`

- When `return_images=True`, decode JPEG bytes → PIL Image → tensor (3, 224, 224) with CLIP normalization
- Add `image_transform` parameter to `__init__`

### 2.5 Inference Changes

**File:** `src/driftsketch/inference.py`

- CFG sampling: `v = v_uncond + cfg_scale * (v_cond - v_uncond)` at each Euler step
- Accept `--image` path, preprocess with CLIP transforms, encode
- CLI: add `--image`, `--cfg-scale` (default 3.0)

### 2.6 New Dependencies

- `open_clip_torch>=2.20.0`
- `Pillow` (likely already present)
- `torchvision` (for image transforms)

### 2.7 Verification

1. CLIP features: encode test image → shape (1, 50, 768)
2. Forward with CLIP: model accepts clip_features, outputs (B, 32, 8)
3. CFG: conditional ≠ unconditional outputs
4. Train 500 steps with images, loss decreases
5. Given input image → generated sketch visually relates to image content

---

## Phase 3: Pixel-Space Loss via easydiffvg

**Goal:** Add differentiable rendering loss alongside velocity MSE for perceptually better sketches.

### 3.1 Rendering Module

**New file:** `src/driftsketch/rendering.py`

- `render_beziers_differentiable(beziers: (32,4,2), canvas_size=128, stroke_width=2.0)`:
  - Map [-1,1] → [0, canvas_size]
  - Create 32 `pydiffvg.Path` objects (one per stroke, `num_control_points=[2]`)
  - Create 32 `ShapeGroup` objects (stroke_color=black, no fill)
  - `pydiffvg.render_differentiable(w, h, shapes, groups, softness=1.0)` → (H, W, 4) RGBA
  - Return grayscale: `1.0 - alpha` → (H, W) white bg, black strokes

- `render_batch_beziers(beziers: (B,32,4,2), max_render=4)`:
  - Loop over min(B, max_render) samples (no batch rendering in easydiffvg)
  - Return (n, H, W) stacked grayscale images

### 3.2 Training Changes

**File:** `src/driftsketch/train.py`

The pixel loss compares a "one-step denoised estimate" against the target, both rasterized:
```
x1_hat = xt + (1 - t) * v_pred          # model's predicted clean output
rendered_pred = render_batch(x1_hat[:K].view(-1,32,4,2))    # differentiable
rendered_target = render_batch(x1[:K].view(-1,32,4,2))      # detached
loss_pixel = MSE(rendered_pred, rendered_target)
loss = w_velocity * loss_velocity + w_pixel * loss_pixel
```

New args: `--pixel-loss-weight` (default 0.0), `--velocity-loss-weight` (default 1.0), `--pixel-batch-size` (default 4), `--pixel-canvas-size` (default 128)

**Strategy:** Train Phase 2 first with velocity-only loss. Then fine-tune with `pixel_loss_weight=0.1` to refine stroke placement.

### 3.3 Verification

1. Gradient test: `beziers.requires_grad_(True)` → render → `.backward()` → gradients are non-zero
2. Visual sanity: render a dataset sample, compare against matplotlib rendering
3. Train with `pixel_loss_weight=0.1` for 500 steps, both loss components decrease
4. Compare: model with pixel loss vs without → pixel-loss model has fewer degenerate/overlapping strokes

---

## Files Summary

| File | Phase | Change |
|---|---|---|
| `src/driftsketch/model.py` | 1, 2 | Add `BezierSketchTransformer`, then `CLIPImageProjector` + `null_memory` |
| `src/driftsketch/train.py` | 1, 2, 3 | Rewrite for ControlSketchDataset, add CLIP, add pixel loss |
| `src/driftsketch/inference.py` | 1, 2 | Rewrite for Bezier generation, SVG export, CFG sampling |
| `src/driftsketch/clip_encoder.py` | 2 | New: frozen CLIP ViT wrapper |
| `src/driftsketch/rendering.py` | 3 | New: differentiable rendering utilities |
| `src/driftsketch/data/controlsketch.py` | 2 | Add image tensor preprocessing |
| `src/driftsketch/__init__.py` | 1 | Export new model class |
| `pyproject.toml` | 1 | Optional deps (wandb, clip, render) |
