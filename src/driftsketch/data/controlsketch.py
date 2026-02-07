"""PyTorch Dataset for ControlSketch (SwiftSketch) — paired image + SVG Bezier sketches."""

import io
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "raw" / "controlsketch"

# Each SVG stroke is: M x0 y0 C x1 y1 x2 y2 x3 y3
# = 4 control points per stroke, 32 strokes per sketch
STROKES_PER_SKETCH = 32
POINTS_PER_STROKE = 4  # start, cp1, cp2, end
SVG_VIEWPORT = 512.0


def parse_svg_beziers(svg_str: str) -> np.ndarray:
    """Parse SVG cubic Bezier paths into control point array.

    Returns array of shape (32, 4, 2) — 32 strokes, 4 control points each, (x, y).
    Normalized to [-1, 1] from the 512x512 SVG viewport.
    """
    paths = re.findall(r'<path d="([^"]+)"', svg_str)
    result = np.zeros((STROKES_PER_SKETCH, POINTS_PER_STROKE, 2), dtype=np.float32)

    for i, path_d in enumerate(paths[:STROKES_PER_SKETCH]):
        nums = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+', path_d)]
        if len(nums) == 8:
            # M x0 y0 C x1 y1 x2 y2 x3 y3
            result[i] = np.array(nums, dtype=np.float32).reshape(4, 2)

    # Normalize from [0, 512] to [-1, 1]
    result = (result / SVG_VIEWPORT) * 2.0 - 1.0
    return result


def beziers_to_points(beziers: np.ndarray, num_points: int = 64) -> np.ndarray:
    """Convert Bezier control points (32, 4, 2) to a resampled polyline (num_points, 2).

    Evaluates each cubic Bezier at multiple t values, concatenates, and resamples
    to a fixed number of evenly-spaced points via arc-length parameterization.
    """
    SAMPLES_PER_BEZIER = 10
    all_pts = []

    for i in range(beziers.shape[0]):
        p0, p1, p2, p3 = beziers[i]
        ts = np.linspace(0, 1, SAMPLES_PER_BEZIER, dtype=np.float32)
        # Cubic Bezier: B(t) = (1-t)^3*P0 + 3(1-t)^2*t*P1 + 3(1-t)*t^2*P2 + t^3*P3
        omt = 1.0 - ts
        pts = (
            omt[:, None] ** 3 * p0
            + 3 * omt[:, None] ** 2 * ts[:, None] * p1
            + 3 * omt[:, None] * ts[:, None] ** 2 * p2
            + ts[:, None] ** 3 * p3
        )
        all_pts.append(pts)

    polyline = np.concatenate(all_pts, axis=0)

    # Arc-length resample to num_points
    diffs = np.diff(polyline, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_len = cum_len[-1]

    if total_len < 1e-8:
        return np.zeros((num_points, 2), dtype=np.float32)

    target_lengths = np.linspace(0, total_len, num_points)
    resampled = np.empty((num_points, 2), dtype=np.float32)
    resampled[:, 0] = np.interp(target_lengths, cum_len, polyline[:, 0])
    resampled[:, 1] = np.interp(target_lengths, cum_len, polyline[:, 1])
    return resampled


class ControlSketchDataset(Dataset):
    """ControlSketch dataset — paired images + vector sketches.

    Each sample provides:
    - beziers: (32, 4, 2) cubic Bezier control points, normalized to [-1, 1]
    - points: (num_points, 2) resampled polyline, normalized to [-1, 1]
    - label: integer class label
    - And optionally: image bytes, caption, attention map, mask

    For unconditional/class-conditional generation, use `points` and `label`.
    For image-conditioned generation, also use `image` and `attn_map`.
    """

    def __init__(
        self,
        split: str = "train",
        categories: list[str] | None = None,
        num_points: int = 64,
        data_dir: str | Path | None = None,
        return_images: bool = False,
    ):
        self.data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        self.split_dir = self.data_dir / split
        self.num_points = num_points
        self.return_images = return_images

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        # Discover or filter categories
        all_cats = sorted(
            d.name for d in self.split_dir.iterdir() if d.is_dir()
        )
        if categories is not None:
            all_cats = [c for c in all_cats if c in categories]
        self.categories = all_cats
        self.category_to_label = {cat: i for i, cat in enumerate(all_cats)}
        self.label_to_category = {i: cat for cat, i in self.category_to_label.items()}
        self.num_classes = len(all_cats)

        # Build index of (npz_path, category) pairs
        self.samples: list[tuple[Path, str]] = []
        for cat in self.categories:
            cat_dir = self.split_dir / cat
            for npz_file in sorted(cat_dir.glob("*.npz")):
                self.samples.append((npz_file, cat))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        npz_path, cat = self.samples[idx]
        data = np.load(npz_path, allow_pickle=True)

        svg_str = str(data["svg_32s"])
        beziers = parse_svg_beziers(svg_str)
        points = beziers_to_points(beziers, self.num_points)
        label = self.category_to_label[cat]

        result = {
            "beziers": torch.from_numpy(beziers),           # (32, 4, 2)
            "points": torch.from_numpy(points),             # (num_points, 2)
            "label": torch.tensor(label, dtype=torch.long), # scalar
            "caption": str(data["caption"]),
        }

        if self.return_images:
            img_bytes = bytes(data["image"])
            result["image_bytes"] = img_bytes
            result["attn_map"] = torch.from_numpy(data["attn_map"].copy())  # (512, 512)
            result["mask"] = torch.from_numpy(data["mask"].copy())          # (512, 512)

        return result

    def stats(self) -> dict:
        per_cat = {}
        for _, cat in self.samples:
            per_cat[cat] = per_cat.get(cat, 0) + 1
        return {
            "split": self.split_dir.name,
            "total_samples": len(self.samples),
            "num_classes": self.num_classes,
            "categories": self.categories,
            "samples_per_category": per_cat,
            "num_points": self.num_points,
        }
