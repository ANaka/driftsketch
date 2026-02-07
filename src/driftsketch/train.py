"""Training pipeline for Conditional Flow Matching sketch generation."""

import argparse
import math
import os

import torch
import torch.nn.functional as F
import wandb

from driftsketch.model import VectorSketchTransformer


def generate_batch(
    batch_size: int, num_points: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of synthetic sketch data.

    Returns (points, labels) where points is (B, N, 2) and labels is (B,)
    integer tensor. First half = circles (label 0), second half = squares
    (label 1).
    """
    half = batch_size // 2
    points_list = []
    labels_list = []

    # --- Circles (label 0) ---
    angles = torch.linspace(0, 2 * math.pi, num_points + 1, device=device)[:num_points]
    circle_base = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # (N, 2)
    circle_base = circle_base.unsqueeze(0).expand(half, -1, -1)  # (half, N, 2)

    scale = torch.empty(half, 1, 1, device=device).uniform_(0.3, 1.0)
    offset = torch.empty(half, 1, 2, device=device).uniform_(-0.5, 0.5)
    noise = torch.randn(half, num_points, 2, device=device) * 0.02

    circles = circle_base * scale + offset + noise
    points_list.append(circles)
    labels_list.append(torch.zeros(half, dtype=torch.long, device=device))

    # --- Squares (label 1) ---
    points_per_edge = num_points // 4
    remainder = num_points - points_per_edge * 4

    edge_points = []
    # Top edge: x from -1 to 1, y = 1
    n = points_per_edge + (1 if remainder > 0 else 0)
    t = torch.linspace(0, 1, n + 1, device=device)[:n]
    edge_points.append(torch.stack([t * 2 - 1, torch.ones(n, device=device)], dim=-1))
    # Right edge: x = 1, y from 1 to -1
    n = points_per_edge + (1 if remainder > 1 else 0)
    t = torch.linspace(0, 1, n + 1, device=device)[:n]
    edge_points.append(torch.stack([torch.ones(n, device=device), 1 - t * 2], dim=-1))
    # Bottom edge: x from 1 to -1, y = -1
    n = points_per_edge + (1 if remainder > 2 else 0)
    t = torch.linspace(0, 1, n + 1, device=device)[:n]
    edge_points.append(torch.stack([1 - t * 2, -torch.ones(n, device=device)], dim=-1))
    # Left edge: x = -1, y from -1 to 1
    n = points_per_edge
    t = torch.linspace(0, 1, n + 1, device=device)[:n]
    edge_points.append(torch.stack([-torch.ones(n, device=device), t * 2 - 1], dim=-1))

    square_base = torch.cat(edge_points, dim=0)[:num_points]  # (N, 2)
    square_base = square_base.unsqueeze(0).expand(half, -1, -1)  # (half, N, 2)

    scale = torch.empty(half, 1, 1, device=device).uniform_(0.3, 1.0)
    offset = torch.empty(half, 1, 2, device=device).uniform_(-0.5, 0.5)
    noise = torch.randn(half, num_points, 2, device=device) * 0.02

    squares = square_base * scale + offset + noise
    points_list.append(squares)
    labels_list.append(torch.ones(half, dtype=torch.long, device=device))

    points = torch.cat(points_list, dim=0)  # (B, N, 2)
    labels = torch.cat(labels_list, dim=0)  # (B,)
    return points, labels


def train() -> None:
    """Train the CFM sketch generation model."""
    parser = argparse.ArgumentParser(description="Train DriftSketch CFM model")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    wandb.init(
        project="driftsketch",
        config={"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr},
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = VectorSketchTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        x1, labels = generate_batch(args.batch_size, model.num_points, device)

        # Sample noise (source distribution)
        x0 = torch.randn_like(x1)

        # Sample time uniformly in [0, 1], shaped for broadcasting over (N, 2)
        t = torch.rand(args.batch_size, 1, 1, device=device)

        # Interpolate
        xt = (1 - t) * x0 + t * x1

        # Target velocity field
        v_target = x1 - x0

        # Predict velocity
        v_pred = model(xt, t.squeeze(-1).squeeze(-1), labels)

        loss = F.mse_loss(v_pred, v_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({"loss": loss.item()})

        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{args.epochs}  loss={loss.item():.6f}")

    # Save checkpoint
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, "model.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    wandb.finish()


if __name__ == "__main__":
    train()
