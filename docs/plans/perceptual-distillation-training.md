# Plan: Perceptual Distillation Training Mode

## Context

DriftSketch currently trains on ControlSketch paired data (15K image+sketch pairs, 15 categories). To scale beyond this, we add a distillation training mode that uses CLIP perceptual losses on arbitrary images — no paired sketch data needed. Any directory of images becomes training data.

**Two-phase workflow:**
1. Bootstrap on ControlSketch with velocity MSE (existing, unchanged)
2. Fine-tune with `--distill` on arbitrary images using CLIP perceptual loss

## Files to Create

### 1. `src/driftsketch/data/images.py` — Image-only dataset

```python
class ImageDataset(Dataset):
    """Load images from any directory (flat or ImageFolder-style)."""
    def __init__(self, root, transform=None, extensions=(".jpg", ".jpeg", ".png")):
        # rglob for all matching files
    def __getitem__(self, idx) -> dict:
        return {"image": self.transform(Image.open(path).convert("RGB"))}
```

### 2. `src/driftsketch/perceptual.py` — CLIP perceptual loss

```python
class CLIPPerceptualLoss(nn.Module):
    """Render beziers → augment → encode with CLIP → cosine similarity loss."""

    def __init__(self, clip_encoder, canvas_size=224, num_augmentations=4):
        # Random crop/rotate/flip augmentation pipeline (critical per CLIPDraw)
        self.aug = T.Compose([
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomRotation(15),
            T.RandomHorizontalFlip(0.5),
        ])

    def forward(self, beziers: (K,32,4,2), target_clip_features: (K,50,768)):
        # 1. render_batch_beziers → (K, H, W) grayscale
        # 2. expand to (K, 3, H, W) RGB
        # 3. augment → (K*num_aug, 3, 224, 224)
        # 4. clip_encoder(augmented) → sketch features (with gradients!)
        # 5. loss = 1 - cosine_sim(sketch_feats.mean(1), target_feats.mean(1))
        return loss, {"loss/clip_cosine": ..., "metric/avg_cosine_sim": ...}
```

Key: augmentation is essential — without it, CLIP-guided optimization finds adversarial solutions.

No LPIPS in distillation mode — it's a same-domain metric, doesn't work for sketch-vs-photo comparison.

## Files to Modify

### 3. `src/driftsketch/clip_encoder.py` — Enable gradient flow

Remove `@torch.no_grad()` from `forward()`. Parameters are already frozen via `requires_grad=False`, so no parameter gradients accumulate. Callers use `torch.no_grad()` explicitly when they want detached features (conditioning path).

### 4. `src/driftsketch/train.py` — Add distillation training loop

New CLI args:
- `--distill` (flag): enable distillation mode
- `--distill-image-dir` (str): directory of images
- `--distill-ode-steps` (int, default 4): ODE steps during training
- `--distill-augmentations` (int, default 4): augmentations per sample
- `--checkpoint` (str): path to bootstrap checkpoint (required with `--distill`)

Distillation training loop (replaces velocity MSE inner loop):
```python
images = batch["image"].to(device)
with torch.no_grad():
    target_clip = clip_encoder(images)     # (B, 50, 768) — no grad

# ODE integration WITH gradients (4 steps, cheap)
x = torch.randn(B, 32, 8, device=device)
dummy_labels = torch.zeros(B, dtype=torch.long, device=device)
dt = 1.0 / args.distill_ode_steps
for step in range(args.distill_ode_steps):
    t_val = torch.full((B,), step * dt, device=device)
    v = model(x, t_val, dummy_labels, clip_features=target_clip)
    x = x + v * dt

# Perceptual loss on rendered subset
K = min(B, args.pixel_batch_size)
beziers = x[:K].view(K, 32, 4, 2)
loss_clip, clip_logs = perceptual_loss_fn(beziers, target_clip[:K])
loss_geo, geo_logs = compute_geometric_losses(beziers, ...)
loss = loss_clip + loss_geo
loss.backward()   # → rendering → ODE → model
```

Update existing conditioning path to wrap CLIP calls in `torch.no_grad()`:
```python
with torch.no_grad():
    clip_features = clip_encoder(images)
```

### 5. `pyproject.toml` — Add torchvision

Add `torchvision>=0.15.0` to `clip` optional deps (needed for augmentation transforms).

## Verification

1. **Gradient flow**: Create random beziers with `requires_grad=True` → render → CLIP encode → loss.backward() → verify beziers.grad is non-zero
2. **Smoke test**: 10 images, 5 epochs distillation, loss decreases
3. **Visual**: Generate from distilled checkpoint, compare to bootstrap-only checkpoint
4. **Augmentation ablation**: Without augmentation, outputs should be adversarial garbage

## Usage

```bash
# Phase 1: Bootstrap (existing)
driftsketch-train --epochs 100 --use-clip --checkpoint-dir checkpoints/bootstrap

# Phase 2: Distill on any images
driftsketch-train --distill \
  --checkpoint checkpoints/bootstrap/model.pt \
  --distill-image-dir ~/images/my_collection \
  --use-clip --epochs 50 --batch-size 16 --lr 1e-5 \
  --pixel-batch-size 4 --distill-ode-steps 4 \
  --checkpoint-dir checkpoints/distilled
```
