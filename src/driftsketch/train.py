"""Training pipeline for Conditional Flow Matching Bezier sketch generation."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader

from driftsketch.data.controlsketch import ControlSketchDataset
from driftsketch.model import BezierSketchTransformer


def _eval_cubic_bezier(control_points: np.ndarray, num_samples: int = 20) -> np.ndarray:
    """Evaluate a cubic Bezier curve at evenly spaced t values.

    Args:
        control_points: (4, 2) array of control points [P0, P1, P2, P3].
        num_samples: number of points to sample along the curve.

    Returns:
        (num_samples, 2) array of points on the curve.
    """
    ts = np.linspace(0, 1, num_samples, dtype=np.float32)
    omt = 1.0 - ts
    p0, p1, p2, p3 = control_points
    pts = (
        omt[:, None] ** 3 * p0
        + 3 * omt[:, None] ** 2 * ts[:, None] * p1
        + 3 * omt[:, None] * ts[:, None] ** 2 * p2
        + ts[:, None] ** 3 * p3
    )
    return pts


def _plot_bezier_samples(
    beziers: torch.Tensor,
    labels: torch.Tensor,
    label_names: dict[int, str],
    save_path: str,
) -> None:
    """Plot generated Bezier sketch samples in a grid.

    Args:
        beziers: (N, 32, 4, 2) control points in [-1, 1].
        labels: (N,) integer class labels.
        save_path: path to save the figure.
    """
    n = beziers.shape[0]
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for i in range(n):
        ax = axes[i]
        strokes = beziers[i].cpu().numpy()  # (32, 4, 2)
        for s in range(strokes.shape[0]):
            pts = _eval_cubic_bezier(strokes[s])
            ax.plot(pts[:, 0], pts[:, 1], linewidth=0.8, color="black")
        cls_name = label_names.get(labels[i].item(), str(labels[i].item()))
        ax.set_title(cls_name, fontsize=8)
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


def train() -> None:
    """Train the CFM Bezier sketch generation model."""
    parser = argparse.ArgumentParser(description="Train DriftSketch CFM model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--data-dir", type=str, default=None, help="ControlSketch data root")
    parser.add_argument("--categories", nargs="+", default=None, help="Category names to use")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--num-classes", type=int, default=None, help="Number of classes (auto from dataset if omitted)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--use-clip", action="store_true", help="Enable CLIP image conditioning")
    parser.add_argument("--p-uncond", type=float, default=0.1, help="Probability of dropping conditioning for CFG")
    parser.add_argument("--clip-model", type=str, default="ViT-B-32", help="CLIP model name")
    parser.add_argument("--clip-pretrained", type=str, default="laion2b_s34b_b79k", help="CLIP pretrained weights")
    parser.add_argument("--velocity-loss-weight", type=float, default=1.0, help="Weight for velocity MSE loss")
    parser.add_argument("--smoothness-loss-weight", type=float, default=0.0, help="Weight for stroke smoothness loss")
    parser.add_argument("--degenerate-loss-weight", type=float, default=0.0, help="Weight for degenerate stroke loss")
    parser.add_argument("--coverage-loss-weight", type=float, default=0.0, help="Weight for coverage uniformity loss")
    parser.add_argument("--pixel-loss-weight", type=float, default=0.0, help="Weight for pixel rendering loss")
    parser.add_argument("--lpips-loss-weight", type=float, default=0.0, help="Weight for LPIPS perceptual loss")
    parser.add_argument("--pixel-batch-size", type=int, default=4, help="How many samples to render per batch")
    parser.add_argument("--pixel-canvas-size", type=int, default=128, help="Canvas size for differentiable rendering")
    args = parser.parse_args()

    # --- wandb setup ---
    wandb_run = None
    if not args.no_wandb:
        import wandb

        wandb_run = wandb.init(
            project="driftsketch",
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "max_grad_norm": args.max_grad_norm,
                "dataset": "controlsketch",
                "split": args.split,
            },
        )

    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- CLIP encoder (must be created before dataset for transform) ---
    if args.use_clip:
        from driftsketch.clip_encoder import FrozenCLIPImageEncoder

        clip_encoder = FrozenCLIPImageEncoder(args.clip_model, args.clip_pretrained).to(device)
        clip_encoder.eval()
        clip_dim = clip_encoder.feature_dim
    else:
        clip_encoder = None
        clip_dim = 0

    # --- Dataset ---
    dataset = ControlSketchDataset(
        split=args.split,
        categories=args.categories,
        data_dir=args.data_dir,
        return_images=args.use_clip,
        image_transform=clip_encoder.get_transform() if clip_encoder else None,
    )
    num_classes = args.num_classes if args.num_classes is not None else dataset.num_classes
    label_names = dataset.label_to_category
    print(f"Dataset: {len(dataset)} samples, {num_classes} classes")
    print(f"Categories: {dataset.categories}")

    if wandb_run:
        import wandb

        wandb_config_update = {"num_classes": num_classes, "categories": dataset.categories}
        if args.use_clip:
            wandb_config_update.update({
                "use_clip": True,
                "clip_model": args.clip_model,
                "clip_pretrained": args.clip_pretrained,
                "clip_dim": clip_dim,
                "p_uncond": args.p_uncond,
            })
        wandb_config_update.update({
            "velocity_loss_weight": args.velocity_loss_weight,
            "smoothness_loss_weight": args.smoothness_loss_weight,
            "degenerate_loss_weight": args.degenerate_loss_weight,
            "coverage_loss_weight": args.coverage_loss_weight,
            "pixel_loss_weight": args.pixel_loss_weight,
            "lpips_loss_weight": args.lpips_loss_weight,
            "pixel_batch_size": args.pixel_batch_size,
            "pixel_canvas_size": args.pixel_canvas_size,
        })
        wandb.config.update(wandb_config_update)

    collate_fn = None
    if args.use_clip:
        from driftsketch.data.controlsketch import controlsketch_collate_fn

        collate_fn = controlsketch_collate_fn

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # --- Output directory ---
    if wandb_run:
        output_dir = os.path.join("outputs", "training", wandb_run.id)
    else:
        output_dir = os.path.join("outputs", "training", "local")
    os.makedirs(output_dir, exist_ok=True)

    # --- Model ---
    model = BezierSketchTransformer(num_classes=num_classes, clip_dim=clip_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # --- Auxiliary losses setup ---
    use_geo = args.smoothness_loss_weight > 0 or args.degenerate_loss_weight > 0 or args.coverage_loss_weight > 0
    use_pixel = args.pixel_loss_weight > 0
    use_lpips = args.lpips_loss_weight > 0

    if use_geo:
        from driftsketch.losses import compute_geometric_losses

    if use_pixel or use_lpips:
        from driftsketch.rendering import render_batch_beziers

    lpips_fn = None
    if use_lpips:
        from driftsketch.losses import LPIPSLoss

        lpips_fn = LPIPSLoss().to(device)

    # Pick a few classes to visualize during training
    viz_classes = list(range(min(6, num_classes)))
    num_vis_per_class = 2

    # --- Training loop ---
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_vel = 0.0
        epoch_pix = 0.0
        num_batches = 0

        for batch in dataloader:
            beziers = batch["beziers"]  # (B, 32, 4, 2)
            B = beziers.shape[0]
            x1 = beziers.view(B, 32, 8).to(device)  # flatten to (B, 32, 8)
            labels = batch["label"].to(device)

            x0 = torch.randn_like(x1)
            t = torch.rand(B, 1, 1, device=device)
            xt = (1 - t) * x0 + t * x1
            v_target = x1 - x0

            clip_features = None
            cfg_mask = None
            if clip_encoder is not None:
                images = batch["image"].to(device)  # (B, 3, 224, 224)
                clip_features = clip_encoder(images)  # (B, 50, 768)
                cfg_mask = torch.rand(B, device=device) < args.p_uncond  # (B,) bool

            v_pred = model(xt, t.squeeze(-1).squeeze(-1), labels, clip_features=clip_features, cfg_mask=cfg_mask)

            loss_velocity = F.mse_loss(v_pred, v_target)
            loss = args.velocity_loss_weight * loss_velocity

            # One-step denoised estimate (shared by all aux losses)
            needs_x1_hat = use_geo or use_pixel or use_lpips
            if needs_x1_hat:
                x1_hat = xt + (1.0 - t) * v_pred  # (B, 32, 8)
                x1_hat_beziers = x1_hat.view(B, 32, 4, 2)

            # Geometric losses (full batch, cheap)
            geo_logs: dict[str, float] = {}
            if use_geo:
                loss_geo, geo_logs = compute_geometric_losses(
                    x1_hat_beziers,
                    w_smoothness=args.smoothness_loss_weight,
                    w_degenerate=args.degenerate_loss_weight,
                    w_coverage=args.coverage_loss_weight,
                )
                loss = loss + loss_geo

            # Pixel + LPIPS losses (subset of batch, expensive)
            loss_pixel = torch.tensor(0.0, device=device)
            loss_lpips = torch.tensor(0.0, device=device)
            if use_pixel or use_lpips:
                K = min(B, args.pixel_batch_size)
                rendered_pred = render_batch_beziers(
                    x1_hat_beziers[:K],
                    canvas_size=args.pixel_canvas_size,
                    max_render=K,
                )
                with torch.no_grad():
                    rendered_target = render_batch_beziers(
                        x1[:K].view(-1, 32, 4, 2),
                        canvas_size=args.pixel_canvas_size,
                        max_render=K,
                    )
                if use_pixel:
                    loss_pixel = F.mse_loss(rendered_pred, rendered_target)
                    loss = loss + args.pixel_loss_weight * loss_pixel
                if use_lpips:
                    loss_lpips = lpips_fn(rendered_pred, rendered_target)
                    loss = loss + args.lpips_loss_weight * loss_lpips

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn_utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            if wandb_run:
                import wandb

                metrics = {
                    "loss/total": loss.item(),
                    "loss/velocity": loss_velocity.item(),
                    "grad_norm": grad_norm.item(),
                }
                metrics.update(geo_logs)
                if use_pixel:
                    metrics["loss/pixel"] = loss_pixel.item()
                if use_lpips:
                    metrics["loss/lpips"] = loss_lpips.item()
                wandb.log(metrics, step=global_step)

            epoch_loss += loss.item()
            epoch_vel += loss_velocity.item()
            epoch_pix += loss_pixel.item()
            num_batches += 1
            global_step += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_vel = epoch_vel / max(num_batches, 1)
        avg_pix = epoch_pix / max(num_batches, 1)
        print(f"Epoch {epoch}/{args.epochs}  avg_loss={avg_loss:.6f}  avg_vel={avg_vel:.6f}  avg_pix={avg_pix:.6f}")

        if wandb_run:
            import wandb

            wandb.log({"epoch_avg_loss": avg_loss}, step=global_step)

        # --- Periodic sample generation ---
        is_sample_epoch = epoch % args.sample_every == 0 or epoch == args.epochs
        if is_sample_epoch:
            model.eval()
            num_vis = num_vis_per_class * len(viz_classes)
            vis_labels = torch.tensor(
                [c for c in viz_classes for _ in range(num_vis_per_class)],
                dtype=torch.long,
                device=device,
            )

            with torch.no_grad():
                x = torch.randn(num_vis, 32, 8, device=device)
                num_steps = 20
                dt = 1.0 / num_steps
                for step_i in range(num_steps):
                    t_val = torch.full((num_vis,), step_i * dt, device=device)
                    v = model(x, t_val, vis_labels)
                    x = x + v * dt

                # Reshape to (num_vis, 32, 4, 2) for plotting
                generated_beziers = x.view(num_vis, 32, 4, 2)

            if wandb_run:
                import wandb

                wandb.log(
                    {
                        "output_variance": x.var().item(),
                        "output_mean_abs": x.abs().mean().item(),
                        "output_std": x.std().item(),
                    },
                    step=global_step,
                )

            sample_path = os.path.join(output_dir, f"samples_epoch_{epoch:04d}.png")
            _plot_bezier_samples(generated_beziers, vis_labels, label_names, sample_path)
            print(f"  Saved samples to {sample_path}")

            if wandb_run:
                import wandb

                wandb.log(
                    {"samples": wandb.Image(sample_path)},
                    step=global_step,
                )

    # --- Save checkpoint ---
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, "model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": args.epochs,
            "config": {
                "num_strokes": 32,
                "coords_per_stroke": 8,
                "embed_dim": model.embed_dim,
                "num_classes": num_classes,
                "categories": dataset.categories,
                "clip_dim": clip_dim,
            },
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")

    if wandb_run:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    train()
