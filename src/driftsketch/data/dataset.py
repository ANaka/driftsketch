"""PyTorch Dataset for QuickDraw sketches processed into fixed-length point sequences."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .processing import strokes_to_points

_DEFAULT_RAW_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "raw" / "quickdraw"
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "processed"


class QuickDrawDataset(Dataset):
    """QuickDraw sketches as (points, label) tensors.

    On first access for a given configuration, processes raw .ndjson files and
    caches the results as .npy files. Subsequent loads use the cache directly.
    """

    def __init__(
        self,
        categories: list[str] | None = None,
        num_points: int = 64,
        max_per_category: int = 5000,
        raw_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ):
        self.raw_dir = Path(raw_dir) if raw_dir else _DEFAULT_RAW_DIR
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self.num_points = num_points
        self.max_per_category = max_per_category

        # Discover categories from raw dir if not specified
        if categories is None:
            categories = sorted(
                p.stem for p in self.raw_dir.glob("*.ndjson")
            )
        self.categories = categories
        self.category_to_label = {cat: i for i, cat in enumerate(categories)}
        self.label_to_category = {i: cat for cat, i in self.category_to_label.items()}
        self.num_classes = len(categories)

        # Load or process data
        self.points, self.labels = self._load_or_process()

    def _cache_path(self, category: str, suffix: str) -> Path:
        return self.cache_dir / f"{category}_n{self.num_points}_max{self.max_per_category}_{suffix}.npy"

    def _load_or_process(self) -> tuple[np.ndarray, np.ndarray]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        all_points = []
        all_labels = []

        for cat in self.categories:
            pts_path = self._cache_path(cat, "points")
            lbl_path = self._cache_path(cat, "labels")

            if pts_path.exists() and lbl_path.exists():
                pts = np.load(pts_path)
                lbl = np.load(lbl_path)
            else:
                pts, lbl = self._process_category(cat)
                np.save(pts_path, pts)
                np.save(lbl_path, lbl)

            all_points.append(pts)
            all_labels.append(lbl)

        return np.concatenate(all_points), np.concatenate(all_labels)

    def _process_category(self, category: str) -> tuple[np.ndarray, np.ndarray]:
        raw_path = self.raw_dir / f"{category}.ndjson"
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data not found: {raw_path}")

        label = self.category_to_label[category]
        points_list = []

        print(f"Processing {category}...")
        with open(raw_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= self.max_per_category:
                    break
                record = json.loads(line)
                pts = strokes_to_points(record["drawing"], self.num_points)
                # Skip degenerate samples
                if np.abs(pts).max() < 1e-6:
                    continue
                points_list.append(pts)

        pts_array = np.stack(points_list)
        lbl_array = np.full(len(points_list), label, dtype=np.int64)
        print(f"  {category}: {len(points_list)} valid samples")
        return pts_array, lbl_array

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        pts = torch.from_numpy(self.points[idx])   # (num_points, 2)
        lbl = torch.tensor(self.labels[idx])        # scalar
        return pts, lbl

    def stats(self) -> dict:
        """Return basic dataset statistics."""
        unique, counts = np.unique(self.labels, return_counts=True)
        per_cat = {self.label_to_category[int(u)]: int(c) for u, c in zip(unique, counts)}
        return {
            "total_samples": len(self.labels),
            "num_classes": self.num_classes,
            "categories": self.categories,
            "samples_per_category": per_cat,
            "point_range": [float(self.points.min()), float(self.points.max())],
            "num_points": self.num_points,
        }
