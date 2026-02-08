"""Training pipeline for line-segment sketch generation via CLIP perceptual distillation.

Trains a CFM model to generate line-segment sketches from arbitrary images
using CLIP perceptual loss and optional LPIPS. No paired stroke data needed.
"""

import argparse
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader
from torchvision import transforms as T

from driftsketch.data.images import ImageDataset
from driftsketch.dit import BezierSketchDiT
from driftsketch.ema import EMA
from driftsketch.model import BezierSketchTransformer
from driftsketch.rendering import render_batch_lines, render_lines_differentiable


def _plot_line_samples(
    lines: torch.Tensor,
    save_path: str,
    num_cols: int = 4,
) -> None:
    """Plot generated line-segment sketch samples in a grid.

    Args:
        lines: (N, num_strokes, 4) line segments in [-1, 1].
        save_path: path to save the figure.
        num_cols: columns in the grid.
    """
    n = lines.shape[0]
    cols = min(n, num_cols)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for i in range(n):
        ax = axes[i]
        strokes = lines[i].cpu().numpy().reshape(-1, 2, 2)  # (num_strokes, 2, 2)
        for stroke in strokes:
            start, end = stroke[0], stroke[1]
            ax.plot([start[0], end[0]], [start[1], end[1]], color="black", linewidth=0.8)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def train_lines() -> None:
    """Train a line-segment sketch generation model via CLIP perceptual distillation."""
    parser = argparse.ArgumentParser(description="Train DriftSketch line-segment model")
    # Data
    parser.add_argument("--image-dir", type=str, required=True, help="Directory of training images")
    parser.add_argument("--num-workers", type=int, default=8)
    # Line-specific
    parser.add_argument("--num-strokes", type=int, default=128, help="Number of line segments")
    parser.add_argument("--canvas-size", type=int, default=224, help="Canvas size for rendering")
    parser.add_argument("--ode-steps", type=int, default=4, help="ODE integration steps (with grads)")
    parser.add_argument("--stroke-width", type=float, default=2.0, help="Rendering stroke width")
    # Loss weights
    parser.add_argument("--clip-loss-weight", type=float, default=1.0, help="Weight for CLIP perceptual loss")
    parser.add_argument("--lpips-loss-weight", type=float, default=0.5, help="Weight for LPIPS loss")
    parser.add_argument("--num-augmentations", type=int, default=4, help="CLIP augmentations per sample")
    parser.add_argument("--render-batch-size", type=int, default=4, help="Samples to render per batch")
    # Training recipe
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=None, help="Linear warmup epochs (default: 5%% of total)")
    parser.add_argument("--min-lr-factor", type=float, default=0.01, help="Cosine decay min LR fraction")
    parser.add_argument("--use-ema", action="store_true", help="Enable EMA of model weights")
    parser.add_argument("--ema-decay", type=float, default=0.9999, help="EMA decay rate")
    # Architecture
    parser.add_argument("--model-type", type=str, default="decoder", choices=["decoder", "dit"])
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    # CLIP
    parser.add_argument("--clip-model", type=str, default="ViT-B-32")
    parser.add_argument("--clip-pretrained", type=str, default="laion2b_s34b_b79k")
    # Checkpointing
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to load")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=0, help="Save checkpoint every N epochs (0=end only)")
    # Logging
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-sample-upload", action="store_true")
    args = parser.parse_args()

    # --- wandb setup ---
    wandb_run = None
    if not args.no_wandb:
        import wandb

        wandb_run = wandb.init(
            project="driftsketch",
            config={
                "mode": "line_distillation",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "max_grad_norm": args.max_grad_norm,
                "num_strokes": args.num_strokes,
                "canvas_size": args.canvas_size,
                "ode_steps": args.ode_steps,
                "stroke_width": args.stroke_width,
                "clip_loss_weight": args.clip_loss_weight,
                "lpips_loss_weight": args.lpips_loss_weight,
                "num_augmentations": args.num_augmentations,
                "model_type": args.model_type,
                "embed_dim": args.embed_dim,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "dropout": args.dropout,
                "weight_decay": args.weight_decay,
                "use_ema": args.use_ema,
                "ema_decay": args.ema_decay,
                "clip_model": args.clip_model,
                "clip_pretrained": args.clip_pretrained,
            },
        )

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- CLIP encoder ---
    from driftsketch.clip_encoder import FrozenCLIPImageEncoder

    clip_encoder = FrozenCLIPImageEncoder(args.clip_model, args.clip_pretrained).to(device)
    clip_encoder.eval()
    clip_dim = clip_encoder.feature_dim

    # --- CLIP augmentations (CLIPDraw-style, inlined) ---
    clip_aug = T.Compose([
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomRotation(15),
        T.RandomHorizontalFlip(0.5),
        T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    # --- Dataset ---
    dataset = ImageDataset(
        root=args.image_dir,
        transform=clip_encoder.get_transform(),
    )
    print(f"Dataset: {len(dataset)} images from {args.image_dir}")

    from driftsketch.data.controlsketch import controlsketch_collate_fn

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=controlsketch_collate_fn,
    )

    # --- Run name and output directory ---
    if wandb_run:
        run_name = wandb_run.name
    else:
        run_name = "local"
    output_dir = os.path.join("outputs", "training-lines", run_name)
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Checkpoint loading ---
    checkpoint_data = None
    if args.checkpoint:
        checkpoint_data = torch.load(args.checkpoint, map_location=device, weights_only=False)
        print(f"Loaded checkpoint from {args.checkpoint}")

    # --- Model ---
    # coords_per_stroke=4 for lines: (start_x, start_y, end_x, end_y)
    num_classes = 1  # dummy class for unconditional generation
    model_kwargs = dict(
        num_strokes=args.num_strokes,
        coords_per_stroke=4,
        num_classes=num_classes,
        clip_dim=clip_dim,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )
    if args.model_type == "dit":
        model = BezierSketchDiT(**model_kwargs, dropout=args.dropout).to(device)
    else:
        model = BezierSketchTransformer(**model_kwargs).to(device)

    if checkpoint_data is not None:
        model.load_state_dict(checkpoint_data["model_state_dict"])
        print("Loaded model weights from checkpoint")

    # Separate param groups: no weight decay on biases, norms, embeddings
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "bias" in name or "embedding" in name or "embed" in name or "pos_encoding" in name or "null_" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=args.lr)

    # --- LR scheduler: linear warmup + cosine decay ---
    warmup_epochs = args.warmup_epochs if args.warmup_epochs is not None else max(1, args.epochs // 20)
    total_epochs = args.epochs

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return args.min_lr_factor + (1.0 - args.min_lr_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- EMA ---
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None

    # --- LPIPS ---
    lpips_fn = None
    use_lpips = args.lpips_loss_weight > 0
    if use_lpips:
        from driftsketch.losses import LPIPSLoss

        lpips_fn = LPIPSLoss().to(device)

    # --- Training loop ---
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_clip_loss = 0.0
        epoch_lpips_loss = 0.0
        num_batches = 0
        epoch_samples = 0
        epoch_start = time.monotonic()

        for batch in dataloader:
            images = batch["image"].to(device)  # (B, 3, 224, 224)
            B = images.shape[0]

            # Encode target images with frozen CLIP (no gradients — conditioning only)
            with torch.no_grad():
                target_clip = clip_encoder(images)  # (B, L, D)

            # ODE integration WITH gradients (few steps)
            x = torch.randn(B, args.num_strokes, 4, device=device)
            dummy_labels = torch.zeros(B, dtype=torch.long, device=device)
            dt = 1.0 / args.ode_steps
            for step in range(args.ode_steps):
                t_val = torch.full((B,), step * dt, device=device)
                v = model(x, t_val, dummy_labels, clip_features=target_clip)
                x = x + v * dt

            # x is now (B, num_strokes, 4) — generated line segments in [-1, 1]

            # Render a subset for loss computation
            K = min(B, args.render_batch_size)
            lines_out = x[:K]  # (K, num_strokes, 4)

            loss = torch.tensor(0.0, device=device)

            # --- CLIP perceptual loss (inlined) ---
            loss_clip = torch.tensor(0.0, device=device)
            cos_sim_val = 0.0
            if args.clip_loss_weight > 0:
                # 1. Render to grayscale
                rendered = render_batch_lines(
                    lines_out, canvas_size=args.canvas_size,
                    stroke_width=args.stroke_width, max_render=K,
                )  # (K, H, W)

                # 2. Expand to 3-channel
                rendered_rgb = rendered.unsqueeze(1).expand(-1, 3, -1, -1)  # (K, 3, H, W)

                # 3. Augment
                augmented = []
                for i in range(K):
                    img = rendered_rgb[i]  # (3, H, W)
                    for _ in range(args.num_augmentations):
                        augmented.append(clip_aug(img))
                augmented = torch.stack(augmented)  # (K * num_aug, 3, 224, 224)

                # 4. Encode with CLIP (with gradients flowing through rendering)
                sketch_features = clip_encoder(augmented)  # (K * num_aug, L, D)

                # Pool over token dim, average over augmentations
                sketch_pooled = sketch_features.mean(dim=1)  # (K * num_aug, D)
                sketch_pooled = sketch_pooled.view(K, args.num_augmentations, -1).mean(dim=1)  # (K, D)

                # Pool target features
                target_pooled = target_clip[:K].mean(dim=1)  # (K, D)

                # Cosine similarity loss
                cos_sim = F.cosine_similarity(sketch_pooled, target_pooled, dim=-1)  # (K,)
                loss_clip = (1 - cos_sim).mean()
                cos_sim_val = cos_sim.mean().item()

                loss = loss + args.clip_loss_weight * loss_clip

            # --- LPIPS loss ---
            loss_lpips = torch.tensor(0.0, device=device)
            if use_lpips:
                # Render lines if not already rendered for CLIP
                if args.clip_loss_weight <= 0:
                    rendered = render_batch_lines(
                        lines_out, canvas_size=args.canvas_size,
                        stroke_width=args.stroke_width, max_render=K,
                    )  # (K, H, W)

                # Resize target images to canvas_size for comparison
                target_resized = F.interpolate(
                    images[:K], size=(args.canvas_size, args.canvas_size),
                    mode="bilinear", align_corners=False,
                )  # (K, 3, H, W)
                # Convert to grayscale for LPIPS comparison
                target_gray = target_resized.mean(dim=1)  # (K, H, W), in [0, 1] approx

                loss_lpips = lpips_fn(rendered, target_gray)
                loss = loss + args.lpips_loss_weight * loss_lpips

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn_utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            if ema is not None:
                ema.update()

            epoch_loss += loss.item()
            epoch_clip_loss += loss_clip.item()
            epoch_lpips_loss += loss_lpips.item()
            epoch_samples += B
            num_batches += 1
            global_step += 1

            if wandb_run:
                import wandb

                metrics = {
                    "loss/total": loss.item(),
                    "loss/clip_cosine": loss_clip.item(),
                    "loss/lpips": loss_lpips.item(),
                    "metric/avg_cosine_sim": cos_sim_val,
                    "grad_norm": grad_norm.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "output/variance": x.var().item(),
                    "output/mean_abs": x.abs().mean().item(),
                    "output/out_of_range_frac": (x.abs() > 1.0).float().mean().item(),
                }
                wandb.log(metrics, step=global_step)

        scheduler.step()

        epoch_duration = time.monotonic() - epoch_start
        avg_loss = epoch_loss / max(num_batches, 1)
        avg_clip = epoch_clip_loss / max(num_batches, 1)
        avg_lpips = epoch_lpips_loss / max(num_batches, 1)
        samples_per_sec = epoch_samples / max(epoch_duration, 1e-6)
        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.6f}  clip={avg_clip:.6f}"
            f"  lpips={avg_lpips:.6f}  lr={cur_lr:.2e}  {samples_per_sec:.0f} samples/s"
            f"  ({epoch_duration:.1f}s)"
        )

        if wandb_run:
            import wandb

            wandb.log({
                "epoch": epoch,
                "epoch/avg_loss": avg_loss,
                "epoch/avg_clip": avg_clip,
                "epoch/avg_lpips": avg_lpips,
                "timing/epoch_duration_sec": epoch_duration,
                "timing/samples_per_sec": samples_per_sec,
            }, step=global_step)

        # --- Periodic sample generation ---
        is_sample_epoch = epoch % args.sample_every == 0 or epoch == args.epochs
        if is_sample_epoch:
            model.eval()
            if ema is not None:
                ema.apply()

            num_vis = 8
            with torch.no_grad():
                # Use first batch images as conditioning
                vis_batch = next(iter(dataloader))
                vis_images = vis_batch["image"][:num_vis].to(device)
                vis_clip = clip_encoder(vis_images)

                x = torch.randn(num_vis, args.num_strokes, 4, device=device)
                dummy_labels = torch.zeros(num_vis, dtype=torch.long, device=device)
                num_steps = 20
                dt = 1.0 / num_steps
                for step_i in range(num_steps):
                    t_val = torch.full((num_vis,), step_i * dt, device=device)
                    v = model(x, t_val, dummy_labels, clip_features=vis_clip)
                    x = x + v * dt

            sample_path = os.path.join(output_dir, f"samples_epoch_{epoch:04d}.png")
            _plot_line_samples(x, sample_path)
            print(f"  Saved samples to {sample_path}")

            if wandb_run:
                import wandb

                # Log sample stats
                wandb.log({
                    "samples/output_variance": x.var().item(),
                    "samples/output_mean_abs": x.abs().mean().item(),
                    "samples/out_of_range_frac": (x.abs() > 1.0).float().mean().item(),
                }, step=global_step)

                if not args.no_sample_upload:
                    wandb.log(
                        {"samples": wandb.Image(sample_path)},
                        step=global_step,
                    )

            if ema is not None:
                ema.restore()

        # --- Periodic checkpoint saving ---
        if args.save_every > 0 and epoch % args.save_every == 0 and epoch < args.epochs:
            _save_checkpoint(
                model, optimizer, scheduler, ema, epoch, args,
                clip_dim, checkpoint_dir,
                filename=f"model_epoch_{epoch:04d}.pt",
            )

    # --- Save final checkpoint ---
    _save_checkpoint(
        model, optimizer, scheduler, ema, args.epochs, args,
        clip_dim, checkpoint_dir,
        filename="model.pt",
    )

    if wandb_run:
        import wandb

        wandb.finish()


def _save_checkpoint(
    model, optimizer, scheduler, ema, epoch, args,
    clip_dim, checkpoint_dir, filename="model.pt",
) -> None:
    """Save a training checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    ckpt_data = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "config": {
            "primitive_type": "line",
            "model_type": args.model_type,
            "num_strokes": args.num_strokes,
            "coords_per_stroke": 4,
            "embed_dim": args.embed_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "num_classes": 1,
            "clip_dim": clip_dim,
            "canvas_size": args.canvas_size,
            "stroke_width": args.stroke_width,
        },
    }
    if ema is not None:
        ckpt_data["ema_state_dict"] = ema.state_dict()
    torch.save(ckpt_data, checkpoint_path)
    print(f"  Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    train_lines()
