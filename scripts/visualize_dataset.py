#!/usr/bin/env python3
"""Visualize the processed QuickDraw dataset and print statistics."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from driftsketch.data.dataset import QuickDrawDataset


def main() -> None:
    ds = QuickDrawDataset()
    stats = ds.stats()

    print("=== Dataset Statistics ===")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Num classes:   {stats['num_classes']}")
    print(f"Num points:    {stats['num_points']}")
    print(f"Point range:   {stats['point_range']}")
    print()
    for cat, count in stats["samples_per_category"].items():
        print(f"  {cat:12s}: {count}")

    # Check for degenerate samples
    flat = ds.points.reshape(len(ds), -1)
    degenerate = (np.abs(flat).max(axis=1) < 1e-6).sum()
    nan_count = np.isnan(ds.points).any(axis=(1, 2)).sum()
    print(f"\nDegenerate samples (all ~0): {degenerate}")
    print(f"Samples with NaN:           {nan_count}")

    # Visualize grid: one row per category, 5 samples each
    categories = stats["categories"]
    n_cols = 5
    n_rows = len(categories)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))

    for row, cat in enumerate(categories):
        label = ds.category_to_label[cat]
        indices = np.where(ds.labels == label)[0]
        sample_idx = np.random.default_rng(42).choice(indices, size=n_cols, replace=False)

        for col, idx in enumerate(sample_idx):
            ax = axes[row, col]
            pts = ds.points[idx]
            ax.plot(pts[:, 0], pts[:, 1], "b-", linewidth=0.8)
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_aspect("equal")
            ax.invert_yaxis()  # QuickDraw has y increasing downward
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(cat, fontsize=10, rotation=0, labelpad=50, va="center")

    fig.suptitle("QuickDraw Dataset Preview (5 samples per category)", fontsize=14)
    plt.tight_layout()

    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "dataset_preview.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to {out_path}")


if __name__ == "__main__":
    main()
