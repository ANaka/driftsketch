# LAION Aesthetics Loss for Sketch Generation

**Date:** 2026-02-07
**Status:** Proposal

## Motivation

Use the LAION improved aesthetic predictor as a training signal — either auxiliary alongside CFM velocity loss, or as the sole objective. The aesthetic predictor is a small MLP on top of CLIP ViT-L/14 embeddings, trained on ~441K human-rated images (SAC + LAION-Logos + AVA dataset). Outputs a continuous score on a 1–10 scale.

The idea: if the model learns to generate sketches that score highly on human aesthetic judgments, we might get more visually pleasing outputs — cleaner lines, better composition, more appealing proportions. The "fun" version (pure aesthetic maximization) treats the predictor as a reward function.

## Architecture

### Pipeline

```
Generated Beziers (B, 32, 4, 2)
    ↓
Differentiable rasterization (pydiffvg, already in codebase)
    ↓
RGB image (B, 3, 224, 224)
    ↓
Frozen CLIP ViT-L/14 encoder
    ↓
L2-normalized embedding (B, 768)
    ↓
Frozen aesthetic MLP (768 → 1024 → 128 → 64 → 16 → 1)
    ↓
Aesthetic score (B, 1), range ~1-10
    ↓
Loss = -mean(scores)  (minimize negative = maximize aesthetics)
```

### LAION Aesthetic Predictor V2

Architecture from [christophschuhmann/improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor):

```python
class AestheticMLP(nn.Module):
    def __init__(self, input_size=768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )
```

Key details:
- **No activation functions** — the "linearMSE" variant is fully linear (recommended by authors)
- **Weights:** `sac+logos+ava1-l14-linearMSE.pth` (best variant)
- **Input:** L2-normalized CLIP ViT-L/14 image embeddings (768-dim)
- **Output:** scalar score, ~1-10 range (most natural images: 4-7)
- Both CLIP and the MLP are frozen — no gradients through them

### CLIP Model

The aesthetic predictor requires **CLIP ViT-L/14** specifically:

| Property | Value |
|---|---|
| Model | `ViT-L/14` (OpenAI CLIP) |
| Input | 224 × 224 RGB |
| Embedding dim | 768 |
| Params | ~400M |

Note: the project currently uses `open_clip` ViT-B-32 (512-dim) for image conditioning. The aesthetic predictor needs the larger ViT-L/14. Options:

1. **Load a second CLIP model** — simpler, more memory (~1.5GB extra on GPU)
2. **Use `open_clip` ViT-L/14** for both conditioning and aesthetics — cleaner but requires retraining conditioning pathway
3. **Use original `clip` package** for aesthetics only — avoids `open_clip` version conflicts

Recommendation: **option 1** (separate model). We only need the image encoder for aesthetics, and `open_clip` can load OpenAI's ViT-L/14 weights.

## Integration into Training Loop

### Current Loss Structure

```python
# train.py lines 231-253
loss_velocity = F.mse_loss(v_pred, v_target)
loss_pixel = torch.tensor(0.0)
if args.pixel_loss_weight > 0:
    x1_hat = xt[:K] + (1 - t[:K]) * v_pred[:K]  # one-step denoised estimate
    rendered_pred = render_batch_beziers(x1_hat.view(-1, 32, 4, 2), ...)
    rendered_target = render_batch_beziers(x1[:K].view(-1, 32, 4, 2), ...)
    loss_pixel = F.mse_loss(rendered_pred, rendered_target)

loss = velocity_weight * loss_velocity + pixel_weight * loss_pixel
```

### Proposed Addition

```python
loss_aesthetic = torch.tensor(0.0)
if args.aesthetic_loss_weight > 0:
    # Reuse rendered_pred from pixel loss, or render if pixel loss is off
    if rendered_pred is None:
        K = min(B, args.aesthetic_batch_size)
        x1_hat = xt[:K] + (1 - t[:K]) * v_pred[:K]
        rendered_pred = render_batch_beziers(
            x1_hat.view(-1, 32, 4, 2),
            canvas_size=224,  # CLIP native resolution
            max_render=K,
        )

    # rendered_pred is (K, H, W) grayscale from pydiffvg
    # Convert to (K, 3, 224, 224) RGB for CLIP
    images = prepare_for_clip(rendered_pred, target_size=224)

    with torch.no_grad():
        clip_emb = clip_aesthetic_model.encode_image(images)  # (K, 768)
        clip_emb = clip_emb / clip_emb.norm(dim=-1, keepdim=True)

    # Aesthetic MLP — also frozen, but we need gradients to flow
    # through the rendering, NOT through CLIP/MLP
    scores = aesthetic_mlp(clip_emb.float())  # (K, 1)
    loss_aesthetic = -scores.mean()

loss = (velocity_weight * loss_velocity
        + pixel_weight * loss_pixel
        + aesthetic_weight * loss_aesthetic)
```

### Gradient Flow Problem

Wait — if CLIP and the aesthetic MLP are both frozen with `no_grad`, where do gradients come from?

The aesthetic score depends on the rendered image, which depends on the Bezier control points, which depend on `v_pred`. But CLIP's `encode_image` is not differentiable w.r.t. its input in the standard API (it's a frozen black box). We need gradients through: **rendered image → CLIP → aesthetic score**.

**Solutions:**

#### A. Straight-Through / REINFORCE (simplest)

Treat the aesthetic score as a reward signal. Use REINFORCE or a straight-through estimator:

```python
# Detach the score, use it to weight the velocity loss
with torch.no_grad():
    scores = compute_aesthetic_score(rendered_pred)  # (K, 1)
    # Normalize scores to advantages
    advantages = (scores - scores.mean()) / (scores.std() + 1e-8)

# Weighted velocity loss — upweight samples that scored well
loss_aesthetic = -(advantages * v_pred[:K]).mean()
```

This is essentially reward-weighted regression, similar to how RLHF works.

#### B. Differentiable Through CLIP (expensive but direct)

Make CLIP parameters frozen but keep the forward pass differentiable:

```python
# Don't use torch.no_grad() — let gradients flow through CLIP
clip_emb = clip_aesthetic_model.encode_image(images)  # grad flows through
clip_emb = clip_emb / clip_emb.norm(dim=-1, keepdim=True)
scores = aesthetic_mlp(clip_emb.float())
loss_aesthetic = -scores.mean()
loss_aesthetic.backward()  # Gradients flow: score → CLIP → image → render → v_pred
```

This works because frozen parameters don't get updated, but gradients still propagate through the computation graph. The gradient tells the model "which direction to push the rendered pixels to increase the aesthetic score." This is what CLIPasso, DiffSketcher, and VectorFusion do.

**Cost:** Full backward pass through CLIP ViT-L/14 on every training step (for the aesthetic batch). Significant memory overhead.

**Recommendation:** Start with approach **B** (differentiable through CLIP). It's what the sketch optimization literature uses, and our rendering is already differentiable via pydiffvg. Only fall back to approach A if memory is prohibitive.

#### C. Hybrid: Differentiable Through CLIP, Periodic Application

Apply aesthetic loss every N steps to amortize cost:

```python
if step % aesthetic_every == 0 and args.aesthetic_loss_weight > 0:
    loss_aesthetic = compute_differentiable_aesthetic_loss(...)
```

## Rendering Considerations

### Current DiffVG Setup

`rendering.py` already handles differentiable Bezier rendering. For aesthetic loss we need:

1. **Canvas size 224×224** — CLIP's native input resolution. Currently the pixel loss uses `--pixel-canvas-size 128`. We can render at 224 directly or render at 128 and bilinear-upsample.

2. **Grayscale → RGB** — DiffVG outputs `(H, W)` grayscale (after alpha extraction). CLIP expects `(3, 224, 224)` RGB. Simple expansion: `img.unsqueeze(0).expand(3, -1, -1)`.

3. **CLIP normalization** — differentiable mean/std normalization:
```python
def clip_normalize(images):  # (B, 3, H, W) in [0, 1]
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1)
    return (images - mean.to(images.device)) / std.to(images.device)
```

### Alternative: Pure PyTorch Soft Rasterization

If DiffVG becomes a bottleneck (it doesn't batch, renders sequentially), a pure PyTorch distance-field approach could be faster:

- Compute signed distance from each pixel to each Bezier curve segment
- Apply Gaussian kernel to produce soft strokes
- Fully batched, no C++ dependency

But since DiffVG already works in the codebase and the aesthetic batch size will be small (2-8 samples), this is probably not needed initially.

## Training Modes

### Mode 1: Auxiliary Loss (recommended start)

```bash
driftsketch-train \
    --velocity-loss-weight 1.0 \
    --pixel-loss-weight 0.0 \
    --aesthetic-loss-weight 0.01 \
    --aesthetic-batch-size 4
```

The aesthetic loss provides a gentle signal toward more visually appealing outputs while the velocity loss maintains the CFM objective. Start with a small weight (0.01–0.1) and tune.

### Mode 2: Aesthetic Only ("the fun one")

```bash
driftsketch-train \
    --velocity-loss-weight 0.0 \
    --aesthetic-loss-weight 1.0 \
    --aesthetic-batch-size 8
```

No reconstruction objective — the model learns purely to maximize aesthetic score. Expected behaviors:
- **Mode collapse** — the model finds one pattern the predictor loves and always generates it
- **Adversarial exploitation** — the model discovers textures/patterns that hack the predictor (high-frequency noise, specific color distributions that score well but look bad to humans)
- **Surprising emergent aesthetics** — the model discovers genuinely pleasing patterns we didn't expect

Mitigations for mode collapse:
- **Class conditioning** — force diversity via class labels (model must score well across all categories)
- **Entropy regularization** — penalize low variance in generated outputs
- **Score clipping** — cap the aesthetic reward to prevent runaway optimization
- **Mixed objective** — small velocity loss weight (e.g., 0.1) as regularizer

### Mode 3: Annealed Transition

Start with CFM-only training, gradually increase aesthetic weight:

```python
aesthetic_weight = args.aesthetic_loss_weight * min(1.0, epoch / warmup_epochs)
```

This lets the model first learn basic structure from the data, then refine for aesthetics.

## Implementation Plan

### New files
- `src/driftsketch/aesthetic.py` — `AestheticScorer` class wrapping CLIP ViT-L/14 + aesthetic MLP

### Modified files
- `src/driftsketch/train.py` — add aesthetic loss computation, new CLI args
- `pyproject.toml` — add `clip` or ensure `open_clip` can load ViT-L/14

### Steps

1. **Download aesthetic predictor weights** — script or manual download of `sac+logos+ava1-l14-linearMSE.pth`
2. **Implement `AestheticScorer`** — loads CLIP ViT-L/14 + aesthetic MLP, provides `score(images) -> (B, 1)` and `loss(beziers) -> scalar`
3. **Add CLI args** to train.py — `--aesthetic-loss-weight`, `--aesthetic-batch-size`, `--aesthetic-model-path`, `--aesthetic-every`
4. **Wire into training loop** — compute aesthetic loss on one-step denoised estimates, add to total loss
5. **Log to wandb** — `loss_aesthetic`, `aesthetic_score_mean`, `aesthetic_score_std`
6. **Test Mode 1** (auxiliary) with small weight
7. **Test Mode 2** (pure aesthetic) for fun

### Weight Download

```bash
# From the improved-aesthetic-predictor repo
wget https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth \
    -O weights/sac+logos+ava1-l14-linearMSE.pth
```

Or bundle into the `AestheticScorer` class to auto-download on first use (like how `open_clip` handles model weights).

## Memory Budget

| Component | VRAM (fp32) | VRAM (fp16) |
|---|---|---|
| BezierSketchTransformer (current model) | ~40 MB | ~20 MB |
| CLIP ViT-B-32 (conditioning, existing) | ~340 MB | ~170 MB |
| CLIP ViT-L/14 (aesthetics, new) | ~1.2 GB | ~600 MB |
| Aesthetic MLP | ~4 MB | ~2 MB |
| DiffVG render (per sample) | ~50 MB | ~50 MB |
| **Total additional** | **~1.3 GB** | **~650 MB** |

With fp16 for the aesthetic CLIP and a small batch size (4), this should fit alongside existing training on a 24GB GPU. The aesthetic CLIP can be loaded in fp16 since we don't need precision — just a coarse gradient signal.

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Mode collapse under pure aesthetic loss | High | Use auxiliary mode, add entropy reg |
| Adversarial textures exploit predictor | Medium | Score clipping, visual inspection |
| CLIP ViT-L/14 OOMs on GPU | Low | fp16, reduce aesthetic batch size |
| Aesthetic predictor biased toward photos | High | Fine-tune on sketch ratings (future) |
| Gradients through CLIP are noisy | Medium | Gradient clipping already in place, low LR for aesthetic term |

The biggest concern is that the aesthetic predictor was trained on photographs, not sketches. Sketch aesthetics may be quite different from photo aesthetics. The predictor might simply reward "photo-like" textures rather than clean linework. But that's part of the experiment — we'll see what it optimizes for.

## Prior Art

- **CLIPasso** (Vinker et al., SIGGRAPH 2022) — optimizes Bezier curves via CLIP perceptual loss + DiffVG
- **DiffSketcher** (Xing et al., NeurIPS 2023) — SDS loss through DiffVG for text-guided vector sketches
- **VectorFusion** (Jain et al., CVPR 2023) — text-to-SVG via Score Distillation Sampling
- **SVGDreamer** (Xing et al., CVPR 2024) — uses reward model for vector particle aesthetic weighting

All of these optimize individual SVGs. Our approach is different — we're training a *generative model* with aesthetic loss as a training signal, not optimizing individual outputs. This is closer to RLHF for diffusion models.

## Open Questions

1. **Which CLIP model to load with?** `open_clip` for consistency, or original `clip` package? Need to verify that `open_clip`'s ViT-L/14 weights produce identical embeddings to OpenAI's original.
2. **Render resolution for aesthetics?** 224 (CLIP native) vs. 128 (current pixel loss) + upsample. Rendering at 224 is cleaner but slower per sample.
3. **Should we also try CLIP-based perceptual loss** (not just aesthetic score)? e.g., maximize CLIP similarity between rendered sketch and the conditioning image. This is what CLIPasso does and might be a more direct quality signal.
4. **DPO alternative?** Instead of reward maximization, generate pairs of sketches, rank by aesthetic score, and do DPO-style training. See `docs/plans/flow-dpo-preference-learning.md` for related thinking.
