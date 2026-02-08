"""Training pipeline for Conditional Flow Matching Bezier sketch generation."""

import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.utils.data import DataLoader

from driftsketch.data.controlsketch import ControlSketchDataset
from driftsketch.dit import BezierSketchDiT
from driftsketch.ema import EMA
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
    parser.add_argument("--aesthetic-loss-weight", type=float, default=0.0, help="Weight for aesthetic loss")
    parser.add_argument("--aesthetic-batch-size", type=int, default=4, help="How many samples to score per batch")
    parser.add_argument("--aesthetic-every", type=int, default=1, help="Apply aesthetic loss every N steps")
    parser.add_argument("--aesthetic-model-path", type=str, default=None, help="Path to aesthetic MLP weights (auto-downloads if None)")
    parser.add_argument("--density-loss-weight", type=float, default=0.0, help="Penalize local line overlap (>0)")
    parser.add_argument("--curvature-loss-weight", type=float, default=0.0, help="Penalize sharp turns (>0)")
    parser.add_argument("--length-loss-weight", type=float, default=0.0, help="Penalize total arc length (>0), reward (<0)")
    parser.add_argument("--uniformity-loss-weight", type=float, default=0.0, help="Penalize spatial spread (>0), reward (<0)")
    parser.add_argument("--density-grid-size", type=int, default=32, help="Grid resolution for density loss")
    parser.add_argument("--density-threshold", type=float, default=3.0, help="Density threshold before penalty")
    # Training recipe
    parser.add_argument("--weight-decay", type=float, default=0.05, help="AdamW weight decay")
    parser.add_argument("--warmup-epochs", type=int, default=None, help="Linear warmup epochs (default: 5%% of total)")
    parser.add_argument("--min-lr-factor", type=float, default=0.01, help="Cosine decay min LR as fraction of base")
    parser.add_argument("--use-ema", action="store_true", help="Enable EMA of model weights")
    parser.add_argument("--ema-decay", type=float, default=0.9999, help="EMA decay rate")
    parser.add_argument("--save-every", type=int, default=0, help="Save checkpoint every N epochs (0=end only)")
    # Architecture
    parser.add_argument("--model-type", type=str, default="decoder", choices=["decoder", "dit"], help="Model architecture")
    parser.add_argument("--embed-dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--num-layers", type=int, default=8, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    # Distillation mode
    parser.add_argument("--distill", action="store_true", help="Enable distillation training mode")
    parser.add_argument("--distill-image-dir", type=str, default=None, help="Directory of images for distillation")
    parser.add_argument("--distill-ode-steps", type=int, default=4, help="ODE steps during distillation training")
    parser.add_argument("--distill-augmentations", type=int, default=4, help="CLIP augmentations per sample")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to bootstrap checkpoint to load")
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
                "dataset": "distill" if args.distill else "controlsketch",
                "split": args.split,
                "model_type": args.model_type,
                "embed_dim": args.embed_dim,
                "num_layers": args.num_layers,
                "num_heads": args.num_heads,
                "dropout": args.dropout,
                "weight_decay": args.weight_decay,
                "use_ema": args.use_ema,
                "ema_decay": args.ema_decay,
            },
        )

    # --- Validate distillation args ---
    if args.distill:
        if not args.use_clip:
            parser.error("--distill requires --use-clip")
        if not args.distill_image_dir:
            parser.error("--distill requires --distill-image-dir")
        if not args.checkpoint:
            parser.error("--distill requires --checkpoint")

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
    if args.distill:
        from driftsketch.data.images import ImageDataset

        dataset = ImageDataset(
            root=args.distill_image_dir,
            transform=clip_encoder.get_transform(),
        )
        # num_classes and label_names come from checkpoint (loaded below)
        num_classes = None
        label_names = {}
        categories = []
        print(f"Distillation dataset: {len(dataset)} images from {args.distill_image_dir}")
    else:
        dataset = ControlSketchDataset(
            split=args.split,
            categories=args.categories,
            data_dir=args.data_dir,
            return_images=args.use_clip,
            image_transform=clip_encoder.get_transform() if clip_encoder else None,
        )
        num_classes = args.num_classes if args.num_classes is not None else dataset.num_classes
        label_names = dataset.label_to_category
        categories = dataset.categories
        print(f"Dataset: {len(dataset)} samples, {num_classes} classes")
        print(f"Categories: {categories}")

    if wandb_run:
        import wandb

        wandb_config_update = {"num_classes": num_classes, "categories": categories}
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
            "aesthetic_loss_weight": args.aesthetic_loss_weight,
            "aesthetic_batch_size": args.aesthetic_batch_size,
            "aesthetic_every": args.aesthetic_every,
            "density_loss_weight": args.density_loss_weight,
            "curvature_loss_weight": args.curvature_loss_weight,
            "length_loss_weight": args.length_loss_weight,
            "uniformity_loss_weight": args.uniformity_loss_weight,
            "density_grid_size": args.density_grid_size,
            "density_threshold": args.density_threshold,
        })
        if args.distill:
            wandb_config_update.update({
                "distill": True,
                "distill_image_dir": args.distill_image_dir,
                "distill_ode_steps": args.distill_ode_steps,
                "distill_augmentations": args.distill_augmentations,
                "bootstrap_checkpoint": args.checkpoint,
            })
        wandb.config.update(wandb_config_update)

    collate_fn = None
    if args.distill or args.use_clip:
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

    # --- Run name and output directory ---
    if wandb_run:
        run_name = wandb_run.name
    else:
        run_name = "local"
    output_dir = os.path.join("outputs", "training", run_name)
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Checkpoint loading ---
    checkpoint_data = None
    if args.checkpoint:
        checkpoint_data = torch.load(args.checkpoint, map_location=device, weights_only=False)
        ckpt_config = checkpoint_data.get("config", {})
        if args.distill:
            num_classes = ckpt_config.get("num_classes", num_classes)
            categories = ckpt_config.get("categories", categories)
            label_names = {i: cat for i, cat in enumerate(categories)}
            print(f"Loaded checkpoint config: {num_classes} classes, categories={categories}")

    # --- Model ---
    model_kwargs = dict(
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
        print(f"Loaded model weights from {args.checkpoint}")

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

    if checkpoint_data is not None and not args.distill:
        # Restore optimizer state for continued training (not distillation — fresh optimizer for new objective)
        if "optimizer_state_dict" in checkpoint_data:
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

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

    # --- Auxiliary losses setup ---
    use_geo = args.smoothness_loss_weight > 0 or args.degenerate_loss_weight > 0 or args.coverage_loss_weight > 0
    use_pixel = args.pixel_loss_weight > 0
    use_lpips = args.lpips_loss_weight > 0

    use_aesthetic = args.aesthetic_loss_weight > 0
    use_plotter = any(w != 0 for w in [
        args.density_loss_weight, args.curvature_loss_weight,
        args.length_loss_weight, args.uniformity_loss_weight,
    ])

    if use_geo:
        from driftsketch.losses import compute_geometric_losses

    if use_pixel or use_lpips or use_aesthetic:
        from driftsketch.rendering import render_batch_beziers

    lpips_fn = None
    if use_lpips:
        from driftsketch.losses import LPIPSLoss

        lpips_fn = LPIPSLoss().to(device)

    if use_plotter:
        from driftsketch.plotter_losses import compute_plotter_losses

    aesthetic_scorer = None
    if use_aesthetic:
        from driftsketch.aesthetic import AestheticScorer

        aesthetic_scorer = AestheticScorer(model_path=args.aesthetic_model_path, device=device)

    # --- Distillation loss setup ---
    perceptual_loss_fn = None
    if args.distill:
        from driftsketch.perceptual import CLIPPerceptualLoss
        from driftsketch.rendering import render_batch_beziers  # noqa: F811

        perceptual_loss_fn = CLIPPerceptualLoss(
            clip_encoder,
            canvas_size=args.pixel_canvas_size,
            num_augmentations=args.distill_augmentations,
        )

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
            if args.distill:
                # --- Distillation training loop ---
                images = batch["image"].to(device)  # (B, 3, 224, 224)
                B = images.shape[0]

                # Encode target images (no gradients — conditioning only)
                with torch.no_grad():
                    target_clip = clip_encoder(images)  # (B, 50, 768)

                # ODE integration WITH gradients (few steps, cheap)
                x = torch.randn(B, 32, 8, device=device)
                dummy_labels = torch.zeros(B, dtype=torch.long, device=device)
                dt = 1.0 / args.distill_ode_steps
                for step in range(args.distill_ode_steps):
                    t_val = torch.full((B,), step * dt, device=device)
                    v = model(x, t_val, dummy_labels, clip_features=target_clip)
                    x = x + v * dt

                # Perceptual loss on rendered subset
                K = min(B, args.pixel_batch_size)
                beziers_out = x[:K].view(K, 32, 4, 2)
                loss_clip, clip_logs = perceptual_loss_fn(beziers_out, target_clip[:K])
                loss = loss_clip

                # Geometric losses on generated beziers (optional, cheap)
                geo_logs: dict[str, float] = {}
                if use_geo:
                    loss_geo, geo_logs = compute_geometric_losses(
                        beziers_out,
                        w_smoothness=args.smoothness_loss_weight,
                        w_degenerate=args.degenerate_loss_weight,
                        w_coverage=args.coverage_loss_weight,
                    )
                    loss = loss + loss_geo

                # Plotter regularization losses (optional)
                plotter_logs: dict[str, float] = {}
                if use_plotter:
                    loss_plotter, plotter_logs = compute_plotter_losses(
                        beziers_out,
                        density_weight=args.density_loss_weight,
                        curvature_weight=args.curvature_loss_weight,
                        length_weight=args.length_loss_weight,
                        uniformity_weight=args.uniformity_loss_weight,
                        density_grid_size=args.density_grid_size,
                        density_threshold=args.density_threshold,
                    )
                    loss = loss + loss_plotter

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn_utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                if ema is not None:
                    ema.update()

                loss_velocity = torch.tensor(0.0)
                loss_pixel = torch.tensor(0.0)

                if wandb_run:
                    import wandb

                    metrics = {
                        "loss/total": loss.item(),
                        "grad_norm": grad_norm.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                    metrics.update(clip_logs)
                    metrics.update(geo_logs)
                    metrics.update(plotter_logs)
                    wandb.log(metrics, step=global_step)
            else:
                # --- Standard velocity MSE training loop ---
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
                    with torch.no_grad():
                        clip_features = clip_encoder(images)  # (B, 50, 768)
                    cfg_mask = torch.rand(B, device=device) < args.p_uncond  # (B,) bool

                v_pred = model(xt, t.squeeze(-1).squeeze(-1), labels, clip_features=clip_features, cfg_mask=cfg_mask)

                loss_velocity = F.mse_loss(v_pred, v_target)
                loss = args.velocity_loss_weight * loss_velocity

                # One-step denoised estimate (shared by all aux losses)
                needs_x1_hat = use_geo or use_pixel or use_lpips or use_aesthetic or use_plotter
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

                # Aesthetic loss (subset of batch, expensive)
                loss_aesthetic = torch.tensor(0.0, device=device)
                aes_scores = None
                if use_aesthetic and global_step % args.aesthetic_every == 0:
                    K_aes = min(B, args.aesthetic_batch_size)
                    rendered_aes = render_batch_beziers(
                        x1_hat_beziers[:K_aes],
                        canvas_size=224,
                        max_render=K_aes,
                    )
                    loss_aesthetic, aes_scores = aesthetic_scorer.compute_loss(rendered_aes)
                    loss = loss + args.aesthetic_loss_weight * loss_aesthetic

                # Plotter regularization losses (full batch, moderate cost)
                plotter_logs: dict[str, float] = {}
                if use_plotter:
                    loss_plotter, plotter_logs = compute_plotter_losses(
                        x1_hat_beziers,
                        density_weight=args.density_loss_weight,
                        curvature_weight=args.curvature_loss_weight,
                        length_weight=args.length_loss_weight,
                        uniformity_weight=args.uniformity_loss_weight,
                        density_grid_size=args.density_grid_size,
                        density_threshold=args.density_threshold,
                    )
                    loss = loss + loss_plotter

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn_utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                if ema is not None:
                    ema.update()

                if wandb_run:
                    import wandb

                    metrics = {
                        "loss/total": loss.item(),
                        "loss/velocity": loss_velocity.item(),
                        "grad_norm": grad_norm.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                    metrics.update(geo_logs)
                    metrics.update(plotter_logs)
                    if use_pixel:
                        metrics["loss/pixel"] = loss_pixel.item()
                    if use_lpips:
                        metrics["loss/lpips"] = loss_lpips.item()
                    if use_aesthetic:
                        metrics["loss/aesthetic"] = loss_aesthetic.item()
                        if aes_scores is not None:
                            metrics["aesthetic/score_mean"] = aes_scores.mean().item()
                            metrics["aesthetic/score_std"] = aes_scores.std().item()
                    wandb.log(metrics, step=global_step)

            epoch_loss += loss.item()
            epoch_vel += loss_velocity.item()
            epoch_pix += loss_pixel.item()
            num_batches += 1
            global_step += 1

        scheduler.step()

        avg_loss = epoch_loss / max(num_batches, 1)
        avg_vel = epoch_vel / max(num_batches, 1)
        avg_pix = epoch_pix / max(num_batches, 1)
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{args.epochs}  avg_loss={avg_loss:.6f}  avg_vel={avg_vel:.6f}  avg_pix={avg_pix:.6f}  lr={cur_lr:.2e}")

        if wandb_run:
            import wandb

            wandb.log({"epoch_avg_loss": avg_loss}, step=global_step)

        # --- Periodic sample generation ---
        is_sample_epoch = epoch % args.sample_every == 0 or epoch == args.epochs
        if is_sample_epoch:
            model.eval()
            if ema is not None:
                ema.apply()
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

            if ema is not None:
                ema.restore()

        # --- Periodic checkpoint saving ---
        if args.save_every > 0 and epoch % args.save_every == 0 and epoch < args.epochs:
            os.makedirs(checkpoint_dir, exist_ok=True)
            periodic_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch:04d}.pt")
            ckpt_data = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "config": {
                    "model_type": args.model_type,
                    "num_strokes": 32,
                    "coords_per_stroke": 8,
                    "embed_dim": args.embed_dim,
                    "num_layers": args.num_layers,
                    "num_heads": args.num_heads,
                    "dropout": args.dropout,
                    "num_classes": num_classes,
                    "categories": categories,
                    "clip_dim": clip_dim,
                },
            }
            if ema is not None:
                ckpt_data["ema_state_dict"] = ema.state_dict()
            torch.save(ckpt_data, periodic_path)
            print(f"  Saved periodic checkpoint to {periodic_path}")

    # --- Save final checkpoint ---
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "model.pt")
    ckpt_data = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": args.epochs,
        "config": {
            "model_type": args.model_type,
            "num_strokes": 32,
            "coords_per_stroke": 8,
            "embed_dim": args.embed_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "num_classes": num_classes,
            "categories": categories,
            "clip_dim": clip_dim,
        },
    }
    if ema is not None:
        ckpt_data["ema_state_dict"] = ema.state_dict()
    torch.save(ckpt_data, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    if wandb_run:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    train()
