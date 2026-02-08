"""Auxiliary losses for aesthetic sketch quality.

Three categories:
1. Geometric stroke quality — pure tensor ops on Bezier control points
2. Pixel-space rendering loss — handled in train.py via rendering.py
3. LPIPS perceptual loss — VGG feature comparison on rendered images
"""

import torch
import torch.nn as nn


def stroke_smoothness_loss(beziers: torch.Tensor) -> torch.Tensor:
    """Penalize curvature by measuring how far interior control points deviate from the chord.

    For each cubic Bezier (P0, P1, P2, P3), computes the perpendicular distance of
    P1 and P2 from the line P0→P3. Straight strokes produce zero loss.

    Args:
        beziers: (B, 32, 4, 2) control points.

    Returns:
        Scalar loss (mean over batch and strokes).
    """
    p0 = beziers[:, :, 0]  # (B, 32, 2)
    p1 = beziers[:, :, 1]
    p2 = beziers[:, :, 2]
    p3 = beziers[:, :, 3]

    # Direction vector of the chord P0→P3
    chord = p3 - p0  # (B, 32, 2)
    chord_len = chord.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # (B, 32, 1)
    chord_unit = chord / chord_len  # (B, 32, 2)

    # Perpendicular distance of P1 from chord line
    v1 = p1 - p0  # (B, 32, 2)
    # Cross product in 2D: |v x u| = |v_x * u_y - v_y * u_x|
    dist1 = (v1[..., 0] * chord_unit[..., 1] - v1[..., 1] * chord_unit[..., 0]).abs()

    # Perpendicular distance of P2 from chord line
    v2 = p2 - p0
    dist2 = (v2[..., 0] * chord_unit[..., 1] - v2[..., 1] * chord_unit[..., 0]).abs()

    return (dist1 + dist2).mean()


def degenerate_stroke_loss(beziers: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    """Penalize collapsed strokes where all 4 control points are near-identical.

    Uses relu(threshold - max_pairwise_dist)^2 so that strokes with sufficient
    spread incur zero loss.

    Args:
        beziers: (B, 32, 4, 2) control points.
        threshold: Minimum desired max pairwise distance.

    Returns:
        Scalar loss (mean over batch and strokes).
    """
    B, S, _, _ = beziers.shape
    # Compute all pairwise distances between the 4 control points
    # (B, 32, 4, 1, 2) - (B, 32, 1, 4, 2) -> (B, 32, 4, 4, 2)
    diffs = beziers.unsqueeze(3) - beziers.unsqueeze(2)
    dists = diffs.norm(dim=-1)  # (B, 32, 4, 4)
    max_dist = dists.amax(dim=(-1, -2))  # (B, 32)

    penalty = torch.relu(threshold - max_dist).pow(2)
    return penalty.mean()


def coverage_uniformity_loss(beziers: torch.Tensor) -> torch.Tensor:
    """Encourage spatial spread of strokes across the canvas.

    Computes mean pairwise distance between stroke midpoints (P0+P3)/2.
    Returns 1/(mean_dist + 0.1), so clustered strokes produce higher loss.

    Args:
        beziers: (B, 32, 4, 2) control points.

    Returns:
        Scalar loss (mean over batch).
    """
    # Stroke midpoints
    midpoints = (beziers[:, :, 0] + beziers[:, :, 3]) / 2  # (B, 32, 2)

    # Pairwise distances between midpoints: (B, 32, 1, 2) - (B, 1, 32, 2)
    diffs = midpoints.unsqueeze(2) - midpoints.unsqueeze(1)  # (B, 32, 32, 2)
    dists = diffs.norm(dim=-1)  # (B, 32, 32)

    # Mean of upper triangle (exclude diagonal)
    S = midpoints.shape[1]
    mask = torch.triu(torch.ones(S, S, device=beziers.device, dtype=torch.bool), diagonal=1)
    mean_dist = dists[:, mask].mean(dim=-1)  # (B,)

    return (1.0 / (mean_dist + 0.1)).mean()


def compute_geometric_losses(
    beziers: torch.Tensor,
    w_smoothness: float = 0.0,
    w_degenerate: float = 0.0,
    w_coverage: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Convenience wrapper computing all geometric losses.

    Args:
        beziers: (B, 32, 4, 2) control points.
        w_smoothness: Weight for smoothness loss.
        w_degenerate: Weight for degenerate stroke loss.
        w_coverage: Weight for coverage uniformity loss.

    Returns:
        (total_loss, log_dict) where log_dict has individual loss values.
    """
    total = torch.tensor(0.0, device=beziers.device)
    logs: dict[str, float] = {}

    if w_smoothness > 0:
        l = stroke_smoothness_loss(beziers)
        total = total + w_smoothness * l
        logs["loss/geo_smoothness"] = l.item()

    if w_degenerate > 0:
        l = degenerate_stroke_loss(beziers)
        total = total + w_degenerate * l
        logs["loss/geo_degenerate"] = l.item()

    if w_coverage > 0:
        l = coverage_uniformity_loss(beziers)
        total = total + w_coverage * l
        logs["loss/geo_coverage"] = l.item()

    return total, logs


class LPIPSLoss(nn.Module):
    """Wraps the lpips package for perceptual loss on grayscale rendered sketches.

    Converts grayscale (N, H, W) images to (N, 3, H, W) in [-1, 1] for LPIPS.
    VGG weights are frozen.
    """

    def __init__(self, net: str = "vgg") -> None:
        super().__init__()
        import lpips

        self.fn = lpips.LPIPS(net=net)
        # Freeze all parameters
        for p in self.fn.parameters():
            p.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute LPIPS between grayscale images.

        Args:
            pred: (N, H, W) predicted grayscale images in [0, 1].
            target: (N, H, W) target grayscale images in [0, 1].

        Returns:
            Scalar LPIPS loss (mean over batch).
        """
        # Convert to (N, 3, H, W) in [-1, 1]
        pred_3ch = pred.unsqueeze(1).expand(-1, 3, -1, -1) * 2.0 - 1.0
        target_3ch = target.unsqueeze(1).expand(-1, 3, -1, -1) * 2.0 - 1.0
        return self.fn(pred_3ch, target_3ch).mean()
