#!/usr/bin/env python3
"""Visualize the ControlSketch dataset — shows source images alongside vector sketches."""

import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from driftsketch.data.controlsketch import ControlSketchDataset


def main() -> None:
    ds = ControlSketchDataset(split="validation", return_images=True)
    stats = ds.stats()

    print("=== ControlSketch Dataset Statistics ===")
    print(f"Split:         {stats['split']}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Num classes:   {stats['num_classes']}")
    print(f"Num points:    {stats['num_points']}")
    print()
    for cat, count in stats["samples_per_category"].items():
        print(f"  {cat:15s}: {count}")

    # Visualize: 3 columns (image, bezier sketch, polyline), one row per category (first 10)
    categories = stats["categories"][:10]
    n_rows = len(categories)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, n_rows * 3))

    rng = np.random.default_rng(42)

    for row, cat in enumerate(categories):
        # Get a random sample from this category
        label = ds.category_to_label[cat]
        cat_indices = [i for i, (_, c) in enumerate(ds.samples) if c == cat]
        idx = rng.choice(cat_indices)
        sample = ds[idx]

        # Column 0: Source image
        ax = axes[row, 0]
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(sample["image_bytes"]))
            ax.imshow(img)
        except ImportError:
            ax.text(0.5, 0.5, "(PIL needed)", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title("Source Image", fontsize=10)
        ax.set_ylabel(cat, fontsize=9, rotation=0, labelpad=55, va="center")

        # Column 1: Bezier curves
        ax = axes[row, 1]
        beziers = sample["beziers"].numpy()  # (32, 4, 2)
        for stroke in beziers:
            ts = np.linspace(0, 1, 20)
            p0, p1, p2, p3 = stroke
            omt = 1 - ts
            curve = (
                omt[:, None] ** 3 * p0
                + 3 * omt[:, None] ** 2 * ts[:, None] * p1
                + 3 * omt[:, None] * ts[:, None] ** 2 * p2
                + ts[:, None] ** 3 * p3
            )
            ax.plot(curve[:, 0], curve[:, 1], "b-", linewidth=0.8)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title("Bezier Sketch", fontsize=10)

        # Column 2: Resampled polyline
        ax = axes[row, 2]
        pts = sample["points"].numpy()  # (num_points, 2)
        ax.plot(pts[:, 0], pts[:, 1], "r-", linewidth=0.8)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title("Polyline (64 pts)", fontsize=10)

    fig.suptitle("ControlSketch Dataset Preview", fontsize=14)
    plt.tight_layout()

    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "controlsketch_preview.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to {out_path}")


if __name__ == "__main__":
    main()
