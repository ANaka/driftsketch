"""Convert QuickDraw stroke data to fixed-length point sequences."""

import numpy as np


def strokes_to_points(strokes: list[list[list[int]]], num_points: int = 64) -> np.ndarray:
    """Convert QuickDraw stroke format to a fixed-length (num_points, 2) array.

    Each stroke is [x_coords, y_coords]. We concatenate all strokes into one
    polyline, resample to exactly `num_points` via arc-length parameterization,
    and normalize to [-1, 1].
    """
    # Concatenate all strokes into a single polyline
    xs, ys = [], []
    for stroke in strokes:
        xs.extend(stroke[0])
        ys.extend(stroke[1])

    if len(xs) < 2:
        # Degenerate drawing — return zeros
        return np.zeros((num_points, 2), dtype=np.float32)

    pts = np.column_stack([xs, ys]).astype(np.float64)

    # Compute cumulative arc length
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_len = cum_len[-1]

    if total_len < 1e-8:
        # All points are identical
        return np.zeros((num_points, 2), dtype=np.float32)

    # Resample at evenly spaced arc-length positions
    target_lengths = np.linspace(0, total_len, num_points)
    resampled = np.empty((num_points, 2), dtype=np.float64)
    resampled[:, 0] = np.interp(target_lengths, cum_len, pts[:, 0])
    resampled[:, 1] = np.interp(target_lengths, cum_len, pts[:, 1])

    # Normalize to [-1, 1]
    center = (resampled.max(axis=0) + resampled.min(axis=0)) / 2.0
    resampled -= center
    scale = np.abs(resampled).max()
    if scale > 1e-8:
        resampled /= scale

    return resampled.astype(np.float32)
