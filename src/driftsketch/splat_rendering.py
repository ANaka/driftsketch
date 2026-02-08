"""Differentiable Bezier splatting renderer in pure PyTorch.

Samples anisotropic 2D Gaussians along cubic Bezier curves and splats them
onto a pixel grid. Fully vectorized, no loops over batch/strokes/samples.

Based on Bezier Splatting (https://arxiv.org/abs/2503.16424).
"""

import torch
import torch.nn.functional as F


def _evaluate_bezier(control_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate cubic Bezier curves at parameter values t.

    Uses De Casteljau-equivalent Bernstein basis:
        B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3

    Args:
        control_points: (B, 32, 4, 2) control points.
        t: (K,) parameter values in [0, 1].

    Returns:
        (B, 32, K, 2) positions on curves.
    """
    # t: (K,) -> (1, 1, K, 1) for broadcasting
    t = t.view(1, 1, -1, 1)
    omt = 1.0 - t  # one minus t

    # Bernstein basis polynomials: (1, 1, K, 1)
    b0 = omt * omt * omt
    b1 = 3.0 * omt * omt * t
    b2 = 3.0 * omt * t * t
    b3 = t * t * t

    # control_points: (B, 32, 4, 2) -> extract each point: (B, 32, 1, 2)
    p0 = control_points[:, :, 0:1, :]
    p1 = control_points[:, :, 1:2, :]
    p2 = control_points[:, :, 2:3, :]
    p3 = control_points[:, :, 3:4, :]

    # (B, 32, K, 2)
    return b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3


def _evaluate_bezier_tangent(
    control_points: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """Evaluate tangent (derivative) of cubic Bezier curves at parameter values t.

    B'(t) = 3[(1-t)^2 (P1-P0) + 2(1-t)t (P2-P1) + t^2 (P3-P2)]

    Args:
        control_points: (B, 32, 4, 2) control points.
        t: (K,) parameter values in [0, 1].

    Returns:
        (B, 32, K, 2) tangent vectors.
    """
    t = t.view(1, 1, -1, 1)
    omt = 1.0 - t

    p0 = control_points[:, :, 0:1, :]
    p1 = control_points[:, :, 1:2, :]
    p2 = control_points[:, :, 2:3, :]
    p3 = control_points[:, :, 3:4, :]

    # Derivative basis
    d0 = omt * omt
    d1 = 2.0 * omt * t
    d2 = t * t

    return 3.0 * (d0 * (p1 - p0) + d1 * (p2 - p1) + d2 * (p3 - p2))


def splat_render_beziers(
    beziers: torch.Tensor,
    canvas_size: int = 224,
    stroke_width: float = 1.5,
    num_samples: int = 20,
    pixel_chunk_size: int = 2048,
) -> torch.Tensor:
    """Render Bezier sketches via Gaussian splatting.

    Pure PyTorch, fully differentiable, batched.

    Args:
        beziers: (B, 32, 4, 2) cubic Bezier control points in [-1, 1].
        canvas_size: Output image resolution (square).
        stroke_width: Gaussian sigma perpendicular to stroke, in pixels.
        num_samples: Number of sample points per Bezier curve (K).
        pixel_chunk_size: Number of pixels to process at once (memory control).

    Returns:
        (B, H, W) grayscale image. White background (1.0), black strokes (0.0).
    """
    B, num_strokes, _, _ = beziers.shape
    device = beziers.device
    dtype = beziers.dtype
    H = W = canvas_size
    K = num_samples

    # 1. Sample positions and tangents along curves
    t_vals = torch.linspace(0, 1, K, device=device, dtype=dtype)
    positions = _evaluate_bezier(beziers, t_vals)  # (B, 32, K, 2)
    tangents = _evaluate_bezier_tangent(beziers, t_vals)  # (B, 32, K, 2)

    # 2. Map positions from [-1, 1] to [0, canvas_size]
    means = (positions + 1.0) / 2.0 * canvas_size  # (B, 32, K, 2)

    # 3. Compute rotation angles from tangent vectors
    angles = torch.atan2(tangents[..., 1], tangents[..., 0])  # (B, 32, K)

    # 4. Compute sigma_along: half-distance between consecutive samples
    # Distance between consecutive sample points in pixel space
    means_flat = means.reshape(B, num_strokes, K, 2)
    diffs = means_flat[:, :, 1:, :] - means_flat[:, :, :-1, :]  # (B, 32, K-1, 2)
    dists = torch.norm(diffs, dim=-1)  # (B, 32, K-1)

    # Pad: first sample gets distance to second, last gets distance to second-to-last
    sigma_along = torch.zeros(B, num_strokes, K, device=device, dtype=dtype)
    sigma_along[:, :, 1:] += dists
    sigma_along[:, :, :-1] += dists
    # Average for interior points (they got both left and right distances)
    sigma_along[:, :, 1:-1] /= 2.0
    # Multiply by 0.5 to get half-distance (so Gaussians overlap sufficiently)
    sigma_along = sigma_along * 0.5
    # Clamp to minimum to avoid degenerate Gaussians
    sigma_along = sigma_along.clamp(min=0.1)

    sigma_across = torch.full_like(sigma_along, stroke_width)  # (B, 32, K)

    # 5. Precompute inverse covariance components
    # Rotation matrix R rotates from local (along, across) to world (x, y):
    #   R = [[cos, -sin], [sin, cos]]
    # Covariance: Sigma = R @ diag(sa^2, sc^2) @ R^T
    # Inverse: Sigma^{-1} = R @ diag(1/sa^2, 1/sc^2) @ R^T
    # For Mahalanobis: d^T Sigma^{-1} d = (cos*dx + sin*dy)^2/sa^2 + (-sin*dx + cos*dy)^2/sc^2
    cos_a = torch.cos(angles)  # (B, 32, K)
    sin_a = torch.sin(angles)  # (B, 32, K)
    inv_sa2 = 1.0 / (sigma_along * sigma_along)  # (B, 32, K)
    inv_sc2 = 1.0 / (sigma_across * sigma_across)  # (B, 32, K)

    # Flatten Gaussians: (B, num_strokes * K) = (B, G)
    G = num_strokes * K
    means_flat = means.reshape(B, G, 2)  # (B, G, 2)
    cos_flat = cos_a.reshape(B, G)  # (B, G)
    sin_flat = sin_a.reshape(B, G)  # (B, G)
    inv_sa2_flat = inv_sa2.reshape(B, G)  # (B, G)
    inv_sc2_flat = inv_sc2.reshape(B, G)  # (B, G)

    # 6. Build pixel grid
    # pixel centers at 0.5, 1.5, ..., (H-0.5)
    py = torch.arange(H, device=device, dtype=dtype) + 0.5  # (H,)
    px = torch.arange(W, device=device, dtype=dtype) + 0.5  # (W,)
    grid_y, grid_x = torch.meshgrid(py, px, indexing="ij")  # (H, W) each
    pixels = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)  # (H*W, 2)
    total_pixels = pixels.shape[0]

    # 7. Compute alpha accumulation in chunks
    output = torch.zeros(B, total_pixels, device=device, dtype=dtype)

    for start in range(0, total_pixels, pixel_chunk_size):
        end = min(start + pixel_chunk_size, total_pixels)
        chunk_pixels = pixels[start:end]  # (P, 2)
        P = chunk_pixels.shape[0]

        # Displacement: pixel - mean for all Gaussians
        # chunk_pixels: (P, 2) -> (1, 1, P, 2)
        # means_flat: (B, G, 2) -> (B, G, 1, 2)
        dx = chunk_pixels[None, None, :, 0] - means_flat[:, :, None, 0]  # (B, G, P)
        dy = chunk_pixels[None, None, :, 1] - means_flat[:, :, None, 1]  # (B, G, P)

        # Rotate displacement into local frame
        # along = cos * dx + sin * dy
        # across = -sin * dx + cos * dy
        cos_exp = cos_flat[:, :, None]  # (B, G, 1)
        sin_exp = sin_flat[:, :, None]  # (B, G, 1)

        d_along = cos_exp * dx + sin_exp * dy  # (B, G, P)
        d_across = -sin_exp * dx + cos_exp * dy  # (B, G, P)

        # Mahalanobis distance squared
        inv_sa2_exp = inv_sa2_flat[:, :, None]  # (B, G, 1)
        inv_sc2_exp = inv_sc2_flat[:, :, None]  # (B, G, 1)

        mahal_sq = d_along * d_along * inv_sa2_exp + d_across * d_across * inv_sc2_exp
        # (B, G, P)

        # Gaussian alpha (clamp exponent for numerical stability)
        alpha = torch.exp(-0.5 * mahal_sq.clamp(max=20.0))  # (B, G, P)

        # Sum over all Gaussians
        output[:, start:end] = alpha.sum(dim=1)  # (B, P)

    # 8. Reshape and compose: white background, black strokes
    output = output.reshape(B, H, W)
    result = 1.0 - output.clamp(0.0, 1.0)

    return result
