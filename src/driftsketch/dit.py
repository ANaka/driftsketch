"""DiT (Diffusion Transformer) architecture for Bezier sketch generation.

Uses adaptive layer norm (adaLN-Zero) conditioning instead of cross-attention,
with SwiGLU feed-forward networks.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class SwiGLUFeedForward(nn.Module):
    """Gated linear unit with SiLU activation.

    hidden_dim is set to (8/3) * dim to match parameter count of a 4x GELU MLP.
    """

    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(8 / 3 * dim)
            # Round to nearest multiple of 8 for efficiency
            hidden_dim = ((hidden_dim + 7) // 8) * 8
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w_down(nn.functional.silu(self.w_gate(x)) * self.w_up(x)))


class AdaLNModulation(nn.Module):
    """Projects conditioning vector to per-layer scale/shift/gate parameters.

    Produces 6 modulation parameters: (shift1, scale1, gate1, shift2, scale2, gate2)
    for two sub-layers (attention + MLP).
    """

    def __init__(self, cond_dim: int, dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(cond_dim, 6 * dim)
        # Zero-initialize so gates start at zero (identity behavior)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, cond: Tensor) -> tuple[Tensor, ...]:
        """Returns (shift1, scale1, gate1, shift2, scale2, gate2), each (B, 1, dim)."""
        out = self.linear(self.silu(cond))  # (B, 6*dim)
        out = out.unsqueeze(1)  # (B, 1, 6*dim)
        return out.chunk(6, dim=-1)


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning.

    Self-attention + SwiGLU MLP, both modulated by conditioning.
    """

    def __init__(self, dim: int, num_heads: int, cond_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = SwiGLUFeedForward(dim, dropout=dropout)
        self.adaLN = AdaLNModulation(cond_dim, dim)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(cond)

        # Self-attention with adaLN
        h = self.norm1(x) * (1 + scale1) + shift1
        h, _ = self.attn(h, h, h)
        x = x + gate1 * h

        # MLP with adaLN
        h = self.norm2(x) * (1 + scale2) + shift2
        h = self.mlp(h)
        x = x + gate2 * h

        return x


class BezierSketchDiT(nn.Module):
    """DiT-style Conditional Flow Matching model for Bezier sketch generation.

    Uses adaLN-Zero conditioning instead of cross-attention. Conditioning
    (time + class or time + CLIP) is fused into a single vector that modulates
    every layer via adaptive layer norm.

    Forward signature is compatible with BezierSketchTransformer.
    """

    def __init__(
        self,
        num_strokes: int = 32,
        coords_per_stroke: int = 8,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 8,
        num_classes: int = 10,
        clip_dim: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_strokes = num_strokes
        self.coords_per_stroke = coords_per_stroke
        self.embed_dim = embed_dim
        self.clip_dim = clip_dim

        # Stroke embedding
        self.stroke_proj = nn.Linear(coords_per_stroke, embed_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, num_strokes, embed_dim) * 0.02
        )

        # Time embedding: sinusoidal -> MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Class embedding -> conditioning vector
        self.class_embed = nn.Embedding(num_classes, 128)
        self.class_mlp = nn.Sequential(
            nn.Linear(128, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # CLIP projection (pool to single vector)
        if clip_dim > 0:
            self.clip_pool = nn.Sequential(
                nn.Linear(clip_dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim),
            )

        # Null conditioning for CFG
        self.null_cond = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

        # DiT blocks
        cond_dim = embed_dim
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, cond_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Final layer: adaLN + output projection
        self.final_adaLN_silu = nn.SiLU()
        self.final_adaLN_linear = nn.Linear(embed_dim, 2 * embed_dim)
        nn.init.zeros_(self.final_adaLN_linear.weight)
        nn.init.zeros_(self.final_adaLN_linear.bias)
        self.final_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.output_proj = nn.Linear(embed_dim, coords_per_stroke)
        # Zero-initialize output projection for stable training
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _sinusoidal_embedding(self, t: Tensor) -> Tensor:
        """Create sinusoidal time embedding. t: (B,) -> (B, embed_dim)."""
        dim = self.embed_dim
        half_dim = dim // 2
        freqs = torch.exp(
            torch.arange(0, dim, 2, device=t.device, dtype=t.dtype)
            * -(math.log(10000.0) / dim)
        )
        args = t.unsqueeze(-1) * freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        class_label: Tensor,
        memory: Tensor | None = None,
        clip_features: Tensor | None = None,
        cfg_mask: Tensor | None = None,
    ) -> Tensor:
        """Predict the velocity field for the flow.

        Args:
            x: (B, num_strokes, coords_per_stroke) noisy Bezier strokes
            t: (B,) or (B, 1) time values in [0, 1]
            class_label: (B,) integer class labels
            memory: unused, kept for API compatibility
            clip_features: optional (B, L, clip_dim) raw CLIP patch features
            cfg_mask: optional (B,) bool tensor, True = drop conditioning

        Returns:
            (B, num_strokes, coords_per_stroke) predicted velocity field
        """
        t = t.view(-1)
        B = x.shape[0]

        # 1. Embed strokes
        h = self.stroke_proj(x) + self.pos_encoding

        # 2. Build conditioning vector: time + class/clip
        time_cond = self.time_mlp(self._sinusoidal_embedding(t))  # (B, embed_dim)

        if clip_features is not None and self.clip_dim > 0:
            # Pool CLIP features to single vector via mean
            clip_pooled = clip_features.mean(dim=1)  # (B, clip_dim)
            class_cond = self.clip_pool(clip_pooled)  # (B, embed_dim)
        else:
            class_cond = self.class_mlp(self.class_embed(class_label))  # (B, embed_dim)

        cond = time_cond + class_cond  # (B, embed_dim)

        # Apply CFG mask — replace with null conditioning
        if cfg_mask is not None:
            null = self.null_cond.expand(B, -1)  # (B, embed_dim)
            cond = torch.where(cfg_mask.unsqueeze(-1), time_cond + null, cond)

        # 3. DiT blocks
        for block in self.blocks:
            h = block(h, cond)

        # 4. Final adaLN + output
        shift, scale = self.final_adaLN_linear(self.final_adaLN_silu(cond)).unsqueeze(1).chunk(2, dim=-1)
        h = self.final_norm(h) * (1 + scale) + shift

        return self.output_proj(h)
