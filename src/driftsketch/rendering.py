"""Differentiable rendering utilities using pydiffvg (easydiffvg).

Renders Bezier stroke sketches to raster images with full gradient support.
"""

import torch


def render_beziers_differentiable(
    beziers: torch.Tensor,
    canvas_size: int = 128,
    stroke_width: float = 2.0,
    softness: float = 1.0,
) -> torch.Tensor:
    """Differentiably render a single sketch's Bezier strokes.

    Args:
        beziers: (32, 4, 2) cubic Bezier control points in [-1, 1].
        canvas_size: Output image size in pixels.
        stroke_width: Width of rendered strokes in pixels.
        softness: Edge softness for anti-aliasing (higher = smoother gradients).

    Returns:
        (H, W) grayscale image tensor -- white background (1.0), black strokes (0.0).
    """
    import pydiffvg

    num_strokes = beziers.shape[0]

    # Map from [-1, 1] to [0, canvas_size]
    points_canvas = (beziers + 1) / 2 * canvas_size

    shapes = []
    shape_groups = []

    for i in range(num_strokes):
        stroke_pts = points_canvas[i]  # (4, 2)
        path = pydiffvg.Path(
            num_control_points=torch.tensor([2]),
            points=stroke_pts,
            stroke_width=torch.tensor(stroke_width),
            is_closed=False,
        )
        shapes.append(path)
        group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([i]),
            fill_color=None,
            stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        )
        shape_groups.append(group)

    # Render to (H, W, 4) RGBA
    image = pydiffvg.render_differentiable(
        canvas_size, canvas_size, shapes, shape_groups
    )

    # Extract alpha channel and invert: white bg (1.0), black strokes (0.0)
    alpha = image[:, :, 3]
    return 1.0 - alpha


def render_lines_differentiable(
    lines: torch.Tensor,
    canvas_size: int = 128,
    stroke_width: float = 2.0,
    softness: float = 1.0,
) -> torch.Tensor:
    """Differentiably render a single sketch's line segments.

    Args:
        lines: (N, 2, 2) or (N, 4) line segments in [-1, 1].
            Each line is (start_point, end_point).
        canvas_size: Output image size in pixels.
        stroke_width: Width of rendered strokes in pixels.
        softness: Edge softness for anti-aliasing (higher = smoother gradients).

    Returns:
        (H, W) grayscale image tensor -- white background (1.0), black strokes (0.0).
    """
    import pydiffvg

    if lines.dim() == 2 and lines.shape[1] == 4:
        lines = lines.reshape(-1, 2, 2)

    num_strokes = lines.shape[0]

    # Map from [-1, 1] to [0, canvas_size]
    points_canvas = (lines + 1) / 2 * canvas_size

    shapes = []
    shape_groups = []

    for i in range(num_strokes):
        stroke_pts = points_canvas[i]  # (2, 2)
        path = pydiffvg.Path(
            num_control_points=torch.tensor([0]),
            points=stroke_pts,
            stroke_width=torch.tensor(stroke_width),
            is_closed=False,
        )
        shapes.append(path)
        group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([i]),
            fill_color=None,
            stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        )
        shape_groups.append(group)

    # Render to (H, W, 4) RGBA
    image = pydiffvg.render_differentiable(
        canvas_size, canvas_size, shapes, shape_groups
    )

    # Extract alpha channel and invert: white bg (1.0), black strokes (0.0)
    alpha = image[:, :, 3]
    return 1.0 - alpha


def render_batch_lines(
    lines: torch.Tensor,
    canvas_size: int = 128,
    stroke_width: float = 2.0,
    max_render: int = 4,
) -> torch.Tensor:
    """Render a batch of line segment sketches.

    Args:
        lines: (B, N, 4) or (B, N, 2, 2) batch of line sketches.
        canvas_size: Output image size.
        stroke_width: Stroke width in pixels.
        max_render: Max samples to render (no batch rendering in pydiffvg).

    Returns:
        (n, H, W) stacked grayscale images where n = min(B, max_render).
    """
    n = min(lines.shape[0], max_render)
    images = []
    for i in range(n):
        img = render_lines_differentiable(
            lines[i], canvas_size=canvas_size, stroke_width=stroke_width
        )
        images.append(img)
    return torch.stack(images)


def render_batch_beziers(
    beziers: torch.Tensor,
    canvas_size: int = 128,
    stroke_width: float = 2.0,
    max_render: int = 4,
) -> torch.Tensor:
    """Render a batch of Bezier sketches.

    Args:
        beziers: (B, 32, 4, 2) batch of sketches.
        canvas_size: Output image size.
        stroke_width: Stroke width in pixels.
        max_render: Max samples to render (no batch rendering in pydiffvg).

    Returns:
        (n, H, W) stacked grayscale images where n = min(B, max_render).
    """
    n = min(beziers.shape[0], max_render)
    images = []
    for i in range(n):
        img = render_beziers_differentiable(
            beziers[i], canvas_size=canvas_size, stroke_width=stroke_width
        )
        images.append(img)
    return torch.stack(images)
