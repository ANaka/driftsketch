"""Inference pipeline: Euler ODE solver, visualisation, and SVG export for Bezier CFM sketches."""

from __future__ import annotations

import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from driftsketch.dit import BezierSketchDiT
from driftsketch.model import BezierSketchTransformer


def generate_beziers(
    model: BezierSketchTransformer | BezierSketchDiT,
    class_label: int,
    num_samples: int,
    num_steps: int = 50,
    device: str = "cpu",
    clip_features: Tensor | None = None,
    cfg_scale: float = 1.0,
) -> Tensor:
    """Generate Bezier stroke sketches via Euler ODE integration.

    Args:
        model: A ``BezierSketchTransformer`` already on *device* and in eval mode.
        class_label: Class index to condition on.
        num_samples: Number of sketches to produce.
        num_steps: Number of Euler integration steps from t=0 to t=1.
        device: Device string matching the model placement.
        clip_features: Optional CLIP image features of shape ``(B, L, D)``.
        cfg_scale: Classifier-free guidance scale (1.0 = no guidance).

    Returns:
        Tensor of shape ``(num_samples, 32, 4, 2)`` — cubic Bezier control points.
    """
    num_strokes = model.pos_encoding.shape[1]
    coords_per_stroke = model.stroke_proj.in_features
    x = torch.randn(num_samples, num_strokes, coords_per_stroke, device=device)
    dt = 1.0 / num_steps
    labels = torch.full((num_samples,), class_label, dtype=torch.long, device=device)

    for i in range(num_steps):
        t_val = i * dt
        t = torch.full((num_samples,), t_val, device=device)
        with torch.no_grad():
            if clip_features is not None and cfg_scale != 1.0:
                # Conditional prediction
                v_cond = model(x, t, labels, clip_features=clip_features)
                # Unconditional prediction (all masked)
                cfg_mask = torch.ones(num_samples, dtype=torch.bool, device=device)
                v_uncond = model(x, t, labels, clip_features=clip_features, cfg_mask=cfg_mask)
                # CFG interpolation
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            elif clip_features is not None:
                v = model(x, t, labels, clip_features=clip_features)
            else:
                v = model(x, t, labels)
        x = x + v * dt

    return x.view(num_samples, num_strokes, 4, 2)


def plot_bezier_sketches(samples: Tensor, path: str, title: str | None = None) -> None:
    """Render generated Bezier stroke sketches in a grid and save to *path*.

    Args:
        samples: ``(N, 32, 4, 2)`` tensor of cubic Bezier control points.
        path: Destination file path for the saved figure.
        title: Optional title for the figure.
    """
    n = samples.shape[0]
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    # Evaluate cubic Bezier curves
    t_vals = torch.linspace(0, 1, 20).unsqueeze(0)  # (1, 20)

    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.set_aspect("equal")
        ax.axis("off")

        if idx >= n:
            continue

        bezier = samples[idx].cpu()  # (32, 4, 2)
        for stroke in bezier:
            p0, p1, p2, p3 = stroke[0], stroke[1], stroke[2], stroke[3]
            # B(t) = (1-t)^3*P0 + 3(1-t)^2*t*P1 + 3(1-t)*t^2*P2 + t^3*P3
            t = t_vals  # (1, 20)
            curve = (
                (1 - t).pow(3) * p0.unsqueeze(-1)
                + 3 * (1 - t).pow(2) * t * p1.unsqueeze(-1)
                + 3 * (1 - t) * t.pow(2) * p2.unsqueeze(-1)
                + t.pow(3) * p3.unsqueeze(-1)
            )  # (2, 20)
            xs = curve[0].numpy()
            ys = curve[1].numpy()
            ax.plot(xs, ys, color="black", linewidth=0.8)
        ax.invert_yaxis()

    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def beziers_to_pydiffvg_shapes(
    beziers: Tensor, canvas_size: int = 256
) -> tuple[list, list]:
    """Convert Bezier control points to pydiffvg shapes for differentiable rendering.

    Args:
        beziers: ``(32, 4, 2)`` single sample, values in [-1, 1].
        canvas_size: Output canvas dimension in pixels.

    Returns:
        Tuple of (shapes, shape_groups) for use with pydiffvg.
    """
    try:
        import pydiffvg
    except ImportError:
        raise RuntimeError(
            "pydiffvg is required for SVG export. Install it from "
            "https://github.com/BachiLi/diffvg"
        )

    # Map from [-1, 1] to [0, canvas_size]
    beziers_canvas = (beziers + 1) / 2 * canvas_size

    shapes = []
    shape_groups = []
    for i, stroke in enumerate(beziers_canvas):
        # stroke: (4, 2) — P0, P1, P2, P3
        points = stroke.contiguous()  # (4, 2)
        path = pydiffvg.Path(
            num_control_points=torch.tensor([2]),
            points=points,
            stroke_width=torch.tensor(2.0),
            is_closed=False,
        )
        shapes.append(path)
        group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([i]),
            fill_color=None,
            stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        )
        shape_groups.append(group)

    return shapes, shape_groups


def export_svg(beziers: Tensor, filename: str, canvas_size: int = 256) -> None:
    """Export a single sample's Bezier strokes as an SVG file.

    Args:
        beziers: ``(32, 4, 2)`` single sample, values in [-1, 1].
        filename: Output SVG file path.
        canvas_size: Output canvas dimension in pixels.
    """
    import pydiffvg

    shapes, groups = beziers_to_pydiffvg_shapes(beziers, canvas_size)
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    pydiffvg.save_svg(filename, canvas_size, canvas_size, shapes, groups)


def main() -> None:
    """CLI entry-point for generating Bezier sketches from a trained checkpoint."""
    parser = argparse.ArgumentParser(
        description="Generate Bezier sketches with a trained CFM model"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--class-label",
        type=int,
        required=True,
        help="Class index to generate",
    )
    parser.add_argument(
        "--num-samples", type=int, default=8, help="Number of samples to generate"
    )
    parser.add_argument(
        "--num-steps", type=int, default=50, help="Euler integration steps"
    )
    parser.add_argument(
        "--output", default="outputs/generated.png", help="Output image path"
    )
    parser.add_argument(
        "--export-svg",
        default=None,
        help="If provided, export first sample as SVG to this path",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of classes (for model construction)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image for CLIP conditioning",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="ViT-B-32",
        help="Open CLIP model architecture name",
    )
    parser.add_argument(
        "--clip-pretrained",
        type=str,
        default="laion2b_s34b_b79k",
        help="Open CLIP pretrained weights tag",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # CLIP conditioning (lazy imports, only when --image is provided)
    if args.image:
        from PIL import Image

        from driftsketch.clip_encoder import FrozenCLIPImageEncoder

        clip_encoder = FrozenCLIPImageEncoder(args.clip_model, args.clip_pretrained).to(device)
        clip_encoder.eval()
        clip_dim = clip_encoder.feature_dim

        # Load and preprocess image
        transform = clip_encoder.get_transform()
        img = Image.open(args.image).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)
        clip_features = clip_encoder(img_tensor)  # (1, 50, 768)
        # Expand to num_samples
        clip_features = clip_features.expand(args.num_samples, -1, -1)  # (N, 50, 768)
    else:
        clip_dim = 0
        clip_features = None

    # Load checkpoint — support old format (raw state_dict) and new format with model type
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        config = ckpt.get("config", {})
        num_classes = config.get("num_classes", args.num_classes)
        ckpt_clip_dim = config.get("clip_dim", 0)
        final_clip_dim = clip_dim if args.image else ckpt_clip_dim
        model_type = config.get("model_type", "decoder")

        model_kwargs = dict(
            num_classes=num_classes,
            clip_dim=final_clip_dim,
            embed_dim=config.get("embed_dim", 256),
            num_heads=config.get("num_heads", 8),
            num_layers=config.get("num_layers", 8),
        )
        if model_type == "dit":
            model = BezierSketchDiT(**model_kwargs, dropout=config.get("dropout", 0.0))
        else:
            model = BezierSketchTransformer(**model_kwargs)

        # Prefer EMA weights if available
        if "ema_state_dict" in ckpt:
            ema_sd = ckpt["ema_state_dict"]
            model_sd = ckpt["model_state_dict"]
            # EMA state_dict has same keys as model params — load into model
            for k in ema_sd:
                if k in model_sd:
                    model_sd[k] = ema_sd[k]
            model.load_state_dict(model_sd)
        else:
            model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded {model_type} model from {args.checkpoint}")
    else:
        final_clip_dim = clip_dim if args.image else 0
        model = BezierSketchTransformer(num_classes=args.num_classes, clip_dim=final_clip_dim)
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()

    samples = generate_beziers(
        model,
        class_label=args.class_label,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        device=device,
        clip_features=clip_features,
        cfg_scale=args.cfg_scale if args.image else 1.0,
    )

    plot_bezier_sketches(samples, args.output)
    print(f"Saved {args.num_samples} samples to {args.output}")

    if args.export_svg:
        export_svg(samples[0], args.export_svg)
        print(f"Exported SVG to {args.export_svg}")


if __name__ == "__main__":
    main()
