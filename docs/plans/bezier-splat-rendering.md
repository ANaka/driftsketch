# Bezier Splat: Pure-PyTorch Differentiable Bezier Rendering

**Date:** 2026-02-07
**Branch:** `bezier-splat`
**Worktree:** `.worktrees/bezier-splat`

## Motivation

The current differentiable renderer (`rendering.py`) uses pydiffvg, which has three problems:

1. **Slow**: Python loop over 32 paths, no batch dimension, C++/Python bridge overhead
2. **Fragile build**: pydiffvg/easydiffvg is a C++/CUDA extension with chronic packaging issues (see the `easydiffvg` name mismatch error)
3. **Sequential**: `render_batch_beziers` loops over the batch dimension one-by-one

Rendering is on the critical path for pixel loss, LPIPS, aesthetic loss, and especially CLIP perceptual distillation (where it runs every training step with gradients).

## Approach: Bezier Splatting in Pure PyTorch

Based on [Bezier Splatting](https://arxiv.org/abs/2503.16424) (NeurIPS 2025, [code](https://github.com/xiliu8006/Bezier_splatting)). Their method samples anisotropic 2D Gaussians along Bezier curves and splats them onto a pixel grid. Their implementation uses a custom CUDA rasterizer (gsplat fork); we implement the same math in pure PyTorch tensor ops to avoid another C++/CUDA dependency.

### Algorithm

Given cubic Bezier control points `(B, 32, 4, 2)`:

1. **Sample curve points**: Evaluate Bernstein basis at K uniform parameter values t in [0,1]. Produces positions `(B, 32, K, 2)`.

2. **Compute tangents**: Derivative of cubic Bezier at each t gives tangent vectors `(B, 32, K, 2)`. Rotation angle = `atan2(ty, tx)`.

3. **Compute Gaussian parameters**:
   - **Mean**: sampled point position, mapped from [-1,1] to [0, canvas_size]
   - **sigma_along** (tangent direction): half the distance between consecutive samples (controls overlap/coverage)
   - **sigma_across** (normal direction): controls stroke width (parameter, e.g. 1.0 pixels)
   - **Rotation**: from tangent angle
   - **Opacity**: 1.0 (or learnable per-stroke, future extension)
   - **Color**: black on white background

4. **Rasterize**: For each pixel, compute Mahalanobis distance to each Gaussian, apply exp to get alpha, composite via soft alpha blending:
   ```
   alpha_i = opacity * exp(-0.5 * mahalanobis_dist^2)
   pixel = 1.0 - clamp(sum(alpha_i), 0, 1)   # white bg, black strokes
   ```
   We use additive alpha (not front-to-back compositing) since strokes are all black — equivalent and avoids sorting.

5. **Output**: `(B, H, W)` grayscale image, same as current `render_batch_beziers`.

### Why additive alpha (not ordered compositing)

The paper uses front-to-back alpha compositing (`C = sum(c_i * alpha_i * prod(1-alpha_j))`), which requires depth-sorting. Since all our strokes are black-on-white, additive opacity accumulation gives identical results without sorting:
- `pixel_darkness = clamp(sum(alpha_i), 0, 1)`
- `pixel_value = 1.0 - pixel_darkness`

This saves the sorting overhead entirely.

### Efficient rasterization strategy

Naively computing all pixel-Gaussian pairs is `(B, H*W, 32*K)` — about 32M ops per sample at 224x224 with K=20. This is fine on GPU but we can be smarter:

- **Chunked pixel processing**: Process pixels in tiles to manage memory. At batch=64, 224x224, 640 Gaussians, the full distance matrix is ~8GB in float32. Process in spatial chunks of e.g. 1024 pixels.
- **Gaussian bounding**: Each Gaussian only affects pixels within ~3 sigma. We can precompute a bounding box per Gaussian and skip distant pixels. This is a potential optimization but not needed for v1 — the naive approach may be fast enough given the relatively small number of Gaussians (640).

## Integration

### New module: `src/driftsketch/splat_rendering.py`

```python
def splat_render_beziers(
    beziers: torch.Tensor,        # (B, 32, 4, 2) in [-1, 1]
    canvas_size: int = 224,
    stroke_width: float = 1.5,    # sigma_across in pixels
    num_samples: int = 20,        # K samples per curve
) -> torch.Tensor:                # (B, H, W) grayscale
```

### Drop-in replacement

Update call sites to use the new renderer:

1. **`rendering.py`**: Add a `renderer="splat"` option or just replace the implementation. Keep `render_beziers_differentiable` as a thin wrapper that calls splat by default, falls back to pydiffvg if requested.

2. **`perceptual.py` (`CLIPPerceptualLoss.forward`)**: Currently calls `render_batch_beziers`. The new function handles batches natively — no more `max_render` cap or sequential loop.

3. **`train.py`**: Multiple call sites for pixel loss, LPIPS, aesthetic loss. All go through `render_batch_beziers` — just need the underlying impl swapped.

### Signature compatibility

Current `render_batch_beziers(beziers, canvas_size, stroke_width, max_render)` → `(n, H, W)`

New `splat_render_beziers(beziers, canvas_size, stroke_width, num_samples)` → `(B, H, W)`

The `max_render` param goes away since we can render the full batch efficiently. Call sites that used `max_render` to limit compute can drop that constraint.

## Implementation Steps

### Step 1: Core splatting function
- Bernstein basis evaluation (cubic: hardcoded coefficients, no loops)
- Tangent computation from Bezier derivative
- Gaussian parameter computation (position, sigma, rotation)
- Pixel-Gaussian distance computation and alpha accumulation
- Output: `splat_render_beziers()` in `splat_rendering.py`

### Step 2: Visual validation
- Script to render the same Beziers with both pydiffvg and splat, compare side-by-side
- Tune `num_samples` and `stroke_width` to match pydiffvg output visually
- Verify gradients flow correctly: `torch.autograd.gradcheck` on small inputs

### Step 3: Integration
- Wire into `rendering.py` as the default renderer
- Update `perceptual.py` to use batch rendering (remove sequential loop)
- Update `train.py` call sites
- Remove `max_render` bottleneck

### Step 4: Benchmarking
- Compare wall-clock time: pydiffvg vs splat at 128x128, 224x224
- Compare forward and backward pass separately
- Test at batch sizes 16, 32, 64
- Memory profiling

## Parameters to tune

| Parameter | Default | Notes |
|-----------|---------|-------|
| `num_samples` | 20 | Points per Bezier curve. More = smoother, slower. 15-30 range. |
| `stroke_width` | 1.5 | sigma_across in pixels. Controls visual line thickness. |
| `canvas_size` | 224 | 224 for CLIP, 128 for faster training losses |

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Visual quality gap vs pydiffvg | Tune num_samples upward (30+). At some point splats overlap enough to look continuous. |
| Memory at large batch × resolution | Chunked pixel processing. Process spatial tiles instead of full canvas at once. |
| Gradient quality differs from pydiffvg | Gaussian splats provide smooth, well-behaved gradients by construction. Should be equal or better. |
| pydiffvg still needed for SVG export | SVG export in inference.py doesn't use differentiable rendering — it writes SVG XML directly from control points. No dependency. |

## Non-goals

- **CUDA kernels**: Pure PyTorch only. If we need more speed later, we can swap in gsplat or write custom kernels then.
- **Adaptive densification/pruning**: The paper's optimization loop adds/removes curves. We don't need this — our model predicts a fixed set of 32 curves.
- **Colored strokes / fill**: All black-on-white for now. Easy to extend later by adding per-stroke color parameters.
- **Variable stroke width**: All strokes same width. Could add per-stroke or per-point width as a future extension.

## References

- [Bezier Splatting paper](https://arxiv.org/abs/2503.16424) — NeurIPS 2025
- [Bezier Splatting code](https://github.com/xiliu8006/Bezier_splatting) — MIT license
- [GaussianImage](https://github.com/Xinjie-Q/GaussianImage) — ECCV 2024, foundation for the gsplat rasterizer
