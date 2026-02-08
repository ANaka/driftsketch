#!/usr/bin/env python3
"""Visual validation of splat renderer vs pydiffvg.

Renders the same Bezier strokes with both renderers and compares:
- Side-by-side PNG images
- Pixel-wise MSE and max absolute difference
- Gradient correctness via torch.autograd.gradcheck

Usage:
    python scripts/validate_splat.py [--data-dir data/raw/controlsketch] [--output-dir outputs/splat_validation]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from driftsketch.splat_rendering import splat_render_beziers


def generate_random_beziers(num_samples: int = 4, seed: int = 42) -> torch.Tensor:
    """Generate random cubic Bezier control points in [-1, 1].

    Returns: (num_samples, 32, 4, 2)
    """
    rng = torch.Generator().manual_seed(seed)
    return torch.rand(num_samples, 32, 4, 2, generator=rng) * 2 - 1


def load_controlsketch_beziers(data_dir: str, num_samples: int = 4) -> torch.Tensor | None:
    """Try to load real ControlSketch beziers. Returns None if unavailable."""
    try:
        from driftsketch.data.controlsketch import ControlSketchDataset

        ds = ControlSketchDataset(split="validation", data_dir=data_dir)
        if len(ds) == 0:
            return None

        rng = np.random.default_rng(42)
        indices = rng.choice(len(ds), size=min(num_samples, len(ds)), replace=False)
        beziers = torch.stack([ds[int(i)]["beziers"] for i in indices])
        return beziers
    except (FileNotFoundError, Exception) as e:
        print(f"Could not load ControlSketch data: {e}")
        return None


def render_with_pydiffvg(beziers: torch.Tensor, canvas_size: int = 128, stroke_width: float = 2.0) -> torch.Tensor:
    """Render using the existing pydiffvg renderer. Returns (B, H, W)."""
    from driftsketch.rendering import render_beziers_differentiable

    images = []
    for i in range(beziers.shape[0]):
        img = render_beziers_differentiable(beziers[i], canvas_size=canvas_size, stroke_width=stroke_width)
        images.append(img)
    return torch.stack(images)


def run_visual_comparison(
    beziers: torch.Tensor,
    output_dir: Path,
    label: str,
    canvas_size: int = 128,
    stroke_width_splat: float = 1.5,
    stroke_width_pydiffvg: float = 2.0,
) -> None:
    """Render with both renderers and save side-by-side comparison."""
    num_samples = beziers.shape[0]

    # Render with splat
    splat_imgs = splat_render_beziers(beziers, canvas_size=canvas_size, stroke_width=stroke_width_splat)

    # Try pydiffvg
    pydiffvg_available = True
    try:
        pydiffvg_imgs = render_with_pydiffvg(beziers, canvas_size=canvas_size, stroke_width=stroke_width_pydiffvg)
    except Exception as e:
        print(f"pydiffvg not available ({e}), showing splat only")
        pydiffvg_available = False

    if pydiffvg_available:
        # Side-by-side comparison
        fig, axes = plt.subplots(num_samples, 3, figsize=(9, num_samples * 3))
        if num_samples == 1:
            axes = axes[None, :]

        for i in range(num_samples):
            splat_np = splat_imgs[i].detach().cpu().numpy()
            pydiffvg_np = pydiffvg_imgs[i].detach().cpu().numpy()
            diff = np.abs(splat_np - pydiffvg_np)

            axes[i, 0].imshow(pydiffvg_np, cmap="gray", vmin=0, vmax=1)
            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])
            if i == 0:
                axes[i, 0].set_title("pydiffvg", fontsize=10)

            axes[i, 1].imshow(splat_np, cmap="gray", vmin=0, vmax=1)
            axes[i, 1].set_xticks([])
            axes[i, 1].set_yticks([])
            if i == 0:
                axes[i, 1].set_title("splat", fontsize=10)

            axes[i, 2].imshow(diff, cmap="hot", vmin=0, vmax=1)
            axes[i, 2].set_xticks([])
            axes[i, 2].set_yticks([])
            if i == 0:
                axes[i, 2].set_title("|difference|", fontsize=10)

        # Compute metrics
        mse = ((splat_imgs.detach() - pydiffvg_imgs.detach()) ** 2).mean().item()
        max_diff = (splat_imgs.detach() - pydiffvg_imgs.detach()).abs().max().item()

        fig.suptitle(
            f"{label} — MSE: {mse:.6f}, Max |diff|: {max_diff:.4f}",
            fontsize=12,
        )
        print(f"  [{label}] MSE: {mse:.6f}, Max |diff|: {max_diff:.4f}")
    else:
        # Splat only
        fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 3, 3))
        if num_samples == 1:
            axes = [axes]
        for i in range(num_samples):
            axes[i].imshow(splat_imgs[i].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_title(f"splat #{i}", fontsize=10)
        fig.suptitle(f"{label} — splat renderer (pydiffvg unavailable)", fontsize=12)
        print(f"  [{label}] splat render complete (pydiffvg unavailable for comparison)")

    plt.tight_layout()
    out_path = output_dir / f"comparison_{label}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def run_gradcheck(device: str = "cpu") -> None:
    """Run torch.autograd.gradcheck on splat_render_beziers with small inputs."""
    print("\n=== Gradient Check ===")

    # Small input: 1 sample, 2 curves, 32x32 canvas, float64 for numerical precision
    beziers = torch.randn(1, 2, 4, 2, dtype=torch.float64, device=device, requires_grad=True) * 0.5

    def func(b: torch.Tensor) -> torch.Tensor:
        # Pad to 32 curves (gradcheck needs consistent shapes)
        padded = torch.zeros(1, 32, 4, 2, dtype=b.dtype, device=b.device)
        padded[:, :2] = b
        return splat_render_beziers(padded, canvas_size=32, stroke_width=1.5, num_samples=10)

    try:
        result = torch.autograd.gradcheck(func, (beziers,), eps=1e-4, atol=1e-3, rtol=1e-3)
        print(f"  gradcheck passed: {result}")
    except Exception as e:
        print(f"  gradcheck FAILED: {e}")
        # Try with relaxed tolerances
        print("  Retrying with relaxed tolerances (atol=1e-2, rtol=1e-2)...")
        try:
            result = torch.autograd.gradcheck(func, (beziers,), eps=1e-4, atol=1e-2, rtol=1e-2)
            print(f"  gradcheck passed (relaxed): {result}")
        except Exception as e2:
            print(f"  gradcheck FAILED (relaxed): {e2}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate splat renderer vs pydiffvg")
    parser.add_argument("--data-dir", type=str, default="data/raw/controlsketch", help="ControlSketch data directory")
    parser.add_argument("--output-dir", type=str, default="outputs/splat_validation", help="Output directory for images")
    parser.add_argument("--canvas-size", type=int, default=128, help="Canvas size for rendering")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of samples to compare")
    parser.add_argument("--skip-gradcheck", action="store_true", help="Skip gradient check")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Splat Renderer Visual Validation ===\n")

    # 1. Try real data
    print("Loading ControlSketch data...")
    real_beziers = load_controlsketch_beziers(args.data_dir, num_samples=args.num_samples)
    if real_beziers is not None:
        print(f"  Loaded {real_beziers.shape[0]} real samples")
        run_visual_comparison(real_beziers, output_dir, "controlsketch", canvas_size=args.canvas_size)
    else:
        print("  No real data available, skipping")

    # 2. Random beziers
    print("\nGenerating random Bezier curves...")
    random_beziers = generate_random_beziers(num_samples=args.num_samples)
    print(f"  Generated {random_beziers.shape[0]} random samples")
    run_visual_comparison(random_beziers, output_dir, "random", canvas_size=args.canvas_size)

    # 3. Gradient check
    if not args.skip_gradcheck:
        run_gradcheck()

    print("\nDone.")


if __name__ == "__main__":
    main()
