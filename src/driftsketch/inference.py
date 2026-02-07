"""Inference pipeline: Euler ODE solver and visualisation for CFM sketches."""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import torch
from torch import Tensor

from driftsketch.model import VectorSketchTransformer


def generate(
    model: VectorSketchTransformer,
    class_label: int,
    num_samples: int,
    num_steps: int = 50,
    device: str = "cpu",
) -> Tensor:
    """Generate sketches via Euler ODE integration of the learned vector field.

    Args:
        model: A ``VectorSketchTransformer`` already on *device* and in eval mode.
        class_label: Class index to condition on (0 = circle, 1 = square).
        num_samples: Number of sketches to produce.
        num_steps: Number of Euler integration steps from t=0 to t=1.
        device: Device string matching the model placement.

    Returns:
        Tensor of shape ``(num_samples, num_points, 2)``.
    """
    num_points = model.pos_encoding.shape[1]
    x = torch.randn(num_samples, num_points, 2, device=device)
    dt = 1.0 / num_steps
    labels = torch.full((num_samples,), class_label, dtype=torch.long, device=device)

    for i in range(num_steps):
        t_val = i * dt
        t = torch.full((num_samples,), t_val, device=device)
        with torch.no_grad():
            v = model(x, t, labels)
        x = x + v * dt

    return x


def plot_sketches(circles: Tensor, squares: Tensor, path: str) -> None:
    """Render generated circle and square sketches side-by-side and save to *path*.

    Args:
        circles: ``(N, num_points, 2)`` tensor of circle sketches.
        squares: ``(N, num_points, 2)`` tensor of square sketches.
        path: Destination file path for the saved figure.
    """
    n = circles.shape[0]
    fig, axes = plt.subplots(n, 2, figsize=(4, 2 * n))

    # Ensure axes is always 2-D even when n == 1.
    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        for j, (sketch, title) in enumerate(
            [(circles[i], "circle"), (squares[i], "square")]
        ):
            ax = axes[i, j]
            pts = sketch.cpu().numpy()
            ax.scatter(pts[:, 0], pts[:, 1], s=2)
            ax.set_aspect("equal")
            ax.axis("off")
            if i == 0:
                ax.set_title(title)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """CLI entry-point for generating sketches from a trained checkpoint."""
    parser = argparse.ArgumentParser(description="Generate sketches with a trained CFM model")
    parser.add_argument("--checkpoint", default="checkpoints/model.pt", help="Path to model checkpoint")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of samples per class")
    parser.add_argument("--output", default="outputs/generated.png", help="Output image path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VectorSketchTransformer()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    circles = generate(model, class_label=0, num_samples=args.num_samples, device=device)
    squares = generate(model, class_label=1, num_samples=args.num_samples, device=device)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plot_sketches(circles, squares, args.output)
    print(f"Saved {args.num_samples} samples per class to {args.output}")


if __name__ == "__main__":
    main()
