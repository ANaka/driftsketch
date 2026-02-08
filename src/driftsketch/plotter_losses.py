"""Pen-plotter regularization losses for Bezier sketch generation.

All losses operate on (B, 32, 4, 2) cubic Bezier control points and return
scalar tensors averaged over the batch. Designed for training-time use on
the one-step denoised estimate x1_hat.

Loss categories:
1. Line density — penalizes local stroke overlap (pen clogging)
2. Curvature — penalizes sharp turns (servo vibration)
3. Total length — controls total arc length (drawing time)
4. Spatial uniformity — controls spatial coverage (composition style)
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Bezier geometry helpers (batched, differentiable)
# ---------------------------------------------------------------------------


def eval_cubic_bezier(cp: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate cubic Bezier curves at parameter values t.

    Args:
        cp: (B, 32, 4, 2) control points.
        t: (T,) parameter values in [0, 1].

    Returns:
        (B, 32, T, 2) evaluated points.
    """
    # t: (T,) -> (1, 1, T, 1) for broadcasting
    t = t.view(1, 1, -1, 1)
    s = 1.0 - t

    p0 = cp[:, :, 0:1, :]  # (B, 32, 1, 2)
    p1 = cp[:, :, 1:2, :]
    p2 = cp[:, :, 2:3, :]
    p3 = cp[:, :, 3:4, :]

    return s**3 * p0 + 3 * s**2 * t * p1 + 3 * s * t**2 * p2 + t**3 * p3


def bezier_derivative(cp: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """First derivative B'(t) of cubic Bezier curves.

    Args:
        cp: (B, 32, 4, 2) control points.
        t: (T,) parameter values in [0, 1].

    Returns:
        (B, 32, T, 2) first derivatives.
    """
    t = t.view(1, 1, -1, 1)
    s = 1.0 - t

    p0 = cp[:, :, 0:1, :]
    p1 = cp[:, :, 1:2, :]
    p2 = cp[:, :, 2:3, :]
    p3 = cp[:, :, 3:4, :]

    return 3 * s**2 * (p1 - p0) + 6 * s * t * (p2 - p1) + 3 * t**2 * (p3 - p2)


def bezier_second_derivative(cp: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Second derivative B''(t) of cubic Bezier curves.

    Args:
        cp: (B, 32, 4, 2) control points.
        t: (T,) parameter values in [0, 1].

    Returns:
        (B, 32, T, 2) second derivatives.
    """
    t = t.view(1, 1, -1, 1)
    s = 1.0 - t

    p0 = cp[:, :, 0:1, :]
    p1 = cp[:, :, 1:2, :]
    p2 = cp[:, :, 2:3, :]
    p3 = cp[:, :, 3:4, :]

    return 6 * s * (p2 - 2 * p1 + p0) + 6 * t * (p3 - 2 * p2 + p1)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def line_density_loss(
    beziers: torch.Tensor,
    grid_size: int = 32,
    sigma: float = 1.5,
    threshold: float = 3.0,
    samples_per_stroke: int = 20,
) -> torch.Tensor:
    """Penalize high local stroke density (overlapping lines cause pen clogging).

    Uses differentiable bilinear splatting onto a 2D grid, Gaussian blur,
    then a soft threshold penalty.

    Args:
        beziers: (B, 32, 4, 2) control points in [-1, 1].
        grid_size: Resolution of the density grid.
        sigma: Gaussian blur sigma.
        threshold: Density above which penalty kicks in.
        samples_per_stroke: Number of parameter samples per stroke.

    Returns:
        Scalar loss averaged over the batch.
    """
    B = beziers.shape[0]
    device = beziers.device

    # Sample points along all strokes
    t = torch.linspace(0, 1, samples_per_stroke, device=device)
    pts = eval_cubic_bezier(beziers, t)  # (B, 32, T, 2)
    pts = pts.reshape(B, -1, 2)  # (B, 32*T, 2)

    # Map from [-1, 1] to [0, grid_size - 1]
    pts_grid = (pts + 1.0) * 0.5 * (grid_size - 1)
    pts_grid = pts_grid.clamp(0, grid_size - 1)

    # Bilinear splatting
    x = pts_grid[..., 0]  # (B, N)
    y = pts_grid[..., 1]  # (B, N)
    x0 = x.floor().long()
    y0 = y.floor().long()
    x1 = (x0 + 1).clamp(max=grid_size - 1)
    y1 = (y0 + 1).clamp(max=grid_size - 1)
    x0 = x0.clamp(0, grid_size - 1)
    y0 = y0.clamp(0, grid_size - 1)

    fx = x - x0.float()  # fractional part
    fy = y - y0.float()

    # Weights for 4 neighboring cells (gradients flow through these)
    w00 = (1 - fx) * (1 - fy)  # (B, N)
    w10 = fx * (1 - fy)
    w01 = (1 - fx) * fy
    w11 = fx * fy

    # Differentiable splatting via scatter_add on flattened grid
    # Flatten grid indices: flat_idx = b * H * W + row * W + col
    G2 = grid_size * grid_size
    batch_offset = torch.arange(B, device=device).unsqueeze(1) * G2  # (B, 1)

    idx00 = batch_offset + y0 * grid_size + x0  # (B, N)
    idx01 = batch_offset + y1 * grid_size + x0
    idx10 = batch_offset + y0 * grid_size + x1
    idx11 = batch_offset + y1 * grid_size + x1

    # Concatenate all 4 contributions
    all_idx = torch.cat([idx00, idx01, idx10, idx11], dim=1).reshape(-1)  # (B*4*N,)
    all_w = torch.cat([w00, w01, w10, w11], dim=1).reshape(-1)  # (B*4*N,)

    density = torch.zeros(B * G2, device=device)
    density.scatter_add_(0, all_idx, all_w)
    density = density.view(B, grid_size, grid_size)

    # Gaussian blur
    ks = int(4 * sigma + 1) | 1  # ensure odd kernel size
    ax = torch.arange(ks, device=device, dtype=torch.float32) - ks // 2
    kernel_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)  # (ks, ks)
    kernel_2d = kernel_2d.view(1, 1, ks, ks)

    density = density.unsqueeze(1)  # (B, 1, H, W)
    density = F.conv2d(density, kernel_2d, padding=ks // 2)
    density = density.squeeze(1)  # (B, H, W)

    # Soft threshold penalty
    excess = F.relu(density - threshold)
    return (excess**2).mean()


def curvature_loss(
    beziers: torch.Tensor,
    num_samples: int = 20,
) -> torch.Tensor:
    """Penalize high curvature (sharp turns stress plotter servos).

    Computes curvature kappa(t) = |B'(t) x B''(t)| / |B'(t)|^3 and penalizes
    the maximum curvature per stroke.

    Args:
        beziers: (B, 32, 4, 2) control points.
        num_samples: Number of parameter samples per stroke.

    Returns:
        Scalar loss averaged over the batch.
    """
    t = torch.linspace(0, 1, num_samples, device=beziers.device)
    d1 = bezier_derivative(beziers, t)  # (B, 32, T, 2)
    d2 = bezier_second_derivative(beziers, t)  # (B, 32, T, 2)

    # 2D cross product magnitude
    cross = (d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]).abs()  # (B, 32, T)

    # Speed cubed
    speed = d1.norm(dim=-1)  # (B, 32, T)
    speed_cubed = speed**3

    # Curvature with numerical stability
    kappa = cross / (speed_cubed + 1e-6)  # (B, 32, T)

    # Max curvature per stroke, squared
    max_kappa = kappa.amax(dim=-1)  # (B, 32)
    return (max_kappa**2).mean()


def total_length_loss(
    beziers: torch.Tensor,
    num_samples: int = 50,
) -> torch.Tensor:
    """Compute total arc length of all strokes (controls drawing time).

    Uses trapezoidal integration of |B'(t)| over [0, 1].

    Args:
        beziers: (B, 32, 4, 2) control points.
        num_samples: Number of integration samples.

    Returns:
        Scalar loss: mean over batch of sum of arc lengths.
    """
    t = torch.linspace(0, 1, num_samples, device=beziers.device)
    d1 = bezier_derivative(beziers, t)  # (B, 32, T, 2)
    speed = d1.norm(dim=-1)  # (B, 32, T)

    # Trapezoidal integration: dt * (0.5*f[0] + f[1] + ... + f[-2] + 0.5*f[-1])
    dt = 1.0 / (num_samples - 1)
    weights = torch.ones(num_samples, device=beziers.device)
    weights[0] = 0.5
    weights[-1] = 0.5
    arc_lengths = (speed * weights.view(1, 1, -1)).sum(dim=-1) * dt  # (B, 32)

    # Sum over strokes, mean over batch
    return arc_lengths.sum(dim=-1).mean()


def spatial_uniformity_loss(
    beziers: torch.Tensor,
    num_samples: int = 20,
) -> torch.Tensor:
    """Measure spatial variance of stroke points (controls coverage/spread).

    Args:
        beziers: (B, 32, 4, 2) control points.
        num_samples: Number of parameter samples per stroke.

    Returns:
        Scalar loss: mean variance of point positions over the batch.
    """
    t = torch.linspace(0, 1, num_samples, device=beziers.device)
    pts = eval_cubic_bezier(beziers, t)  # (B, 32, T, 2)
    pts = pts.reshape(pts.shape[0], -1, 2)  # (B, 32*T, 2)

    # Variance across all points per batch item
    var = pts.var(dim=1).sum(dim=-1)  # (B,) — sum of x and y variance
    return var.mean()


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------


def compute_plotter_losses(
    beziers: torch.Tensor,
    density_weight: float = 0.0,
    curvature_weight: float = 0.0,
    length_weight: float = 0.0,
    uniformity_weight: float = 0.0,
    density_grid_size: int = 32,
    density_threshold: float = 3.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute all active plotter regularization losses.

    Only computes losses whose weight != 0. Returns individual unweighted
    loss values in the metrics dict for logging.

    Args:
        beziers: (B, 32, 4, 2) control points.
        density_weight: Weight for line density loss.
        curvature_weight: Weight for curvature loss.
        length_weight: Weight for total length loss.
        uniformity_weight: Weight for spatial uniformity loss.
        density_grid_size: Grid resolution for density computation.
        density_threshold: Density threshold before penalty.

    Returns:
        (weighted_total_loss, metrics_dict)
    """
    total = torch.tensor(0.0, device=beziers.device)
    metrics: dict[str, float] = {}

    if density_weight != 0:
        l = line_density_loss(
            beziers, grid_size=density_grid_size, threshold=density_threshold
        )
        total = total + density_weight * l
        metrics["loss/plotter_density"] = l.item()

    if curvature_weight != 0:
        l = curvature_loss(beziers)
        total = total + curvature_weight * l
        metrics["loss/plotter_curvature"] = l.item()

    if length_weight != 0:
        l = total_length_loss(beziers)
        total = total + length_weight * l
        metrics["loss/plotter_length"] = l.item()

    if uniformity_weight != 0:
        l = spatial_uniformity_loss(beziers)
        total = total + uniformity_weight * l
        metrics["loss/plotter_uniformity"] = l.item()

    return total, metrics
