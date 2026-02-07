# Training Metrics & Sample Visualization

## Context

The training loop currently logs only MSE loss to wandb. We need richer metrics to diagnose training health and periodic visual output to see what the model is actually generating.

## Changes

### File: `src/driftsketch/train.py`

#### New CLI arg

- `--max-grad-norm` (float, default 1.0): Max gradient norm for clipping.

#### Per-epoch metrics (logged to wandb)

1. **`grad_norm`** — Pre-clip L2 norm of all gradients, computed via `torch.nn.utils.clip_grad_norm_()` with `max_norm=args.max_grad_norm`. Serves dual purpose: monitoring + clipping.
2. **`loss_class_0`, `loss_class_1`** — MSE loss split by class label within the batch.

#### Periodic inference metrics & samples (every 200 epochs)

Run quick inference: 4 samples per class, 20 Euler steps.

3. **`output_variance`** — `var()` of generated points across samples. Low = mode collapse.
4. **`output_mean_abs`** — Mean absolute coordinate value. Should stay near 0-1 range.
5. **`output_std`** — Std of generated coordinates. Tracks output spread.
6. **Sample PNG** — Saved to `outputs/samples_epoch_{epoch:05d}.png`. Also saved on final epoch.

#### Imports

- Reuse `generate()` and `plot_sketches()` from `driftsketch.inference` — no new files.

## Verification

```bash
driftsketch-train --epochs 400 --batch-size 64
# Should produce:
# - wandb logs with grad_norm, loss_class_0, loss_class_1 every epoch
# - wandb logs with output_variance, output_mean_abs, output_std at epoch 200, 400
# - outputs/samples_epoch_00200.png and outputs/samples_epoch_00400.png
```
