import math

import torch
import torch.nn as nn
from torch import Tensor


class VectorSketchTransformer(nn.Module):
    """Conditional Flow Matching model for vector sketch generation."""

    def __init__(
        self,
        num_points: int = 64,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 6,
        num_classes: int = 2,
        cond_dim: int = 64,
    ):
        super().__init__()
        self.num_points = num_points
        self.embed_dim = embed_dim

        # Point embedding: Linear(2 -> embed_dim) + learnable positional encoding
        self.point_proj = nn.Linear(2, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, num_points, embed_dim) * 0.02)

        # Time embedding: sinusoidal features -> MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Class embedding: lookup table -> MLP
        self.class_embed = nn.Embedding(num_classes, cond_dim)
        self.class_mlp = nn.Sequential(
            nn.Linear(cond_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Transformer backbone (pre-LN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, 2)

    def _sinusoidal_embedding(self, t: Tensor) -> Tensor:
        """Create sinusoidal time embedding.

        Args:
            t: (B,) time values in [0, 1]

        Returns:
            (B, embed_dim) sinusoidal features
        """
        dim = self.embed_dim
        half_dim = dim // 2
        freqs = torch.exp(
            torch.arange(0, dim, 2, device=t.device, dtype=t.dtype)
            * -(math.log(10000.0) / dim)
        )
        # t: (B,) -> (B, 1) * (half_dim,) -> (B, half_dim)
        args = t.unsqueeze(-1) * freqs
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, x: Tensor, t: Tensor, class_label: Tensor) -> Tensor:
        """Predict the velocity field for the flow.

        Args:
            x: (B, N, 2) point sequences
            t: (B,) or (B, 1) time values in [0, 1]
            class_label: (B,) integer class labels

        Returns:
            (B, N, 2) predicted velocity field
        """
        # Flatten t to (B,) if needed
        t = t.view(-1)

        # 1. Embed points: (B, N, 2) -> (B, N, embed_dim)
        point_emb = self.point_proj(x) + self.pos_encoding

        # 2. Embed time: (B,) -> (B, embed_dim) -> (B, 1, embed_dim)
        time_emb = self.time_mlp(self._sinusoidal_embedding(t)).unsqueeze(1)

        # 3. Embed class: (B,) -> (B, embed_dim) -> (B, 1, embed_dim)
        class_emb = self.class_mlp(self.class_embed(class_label)).unsqueeze(1)

        # 4. Concat tokens: [time, class, points] -> (B, N+2, embed_dim)
        tokens = torch.cat([time_emb, class_emb, point_emb], dim=1)

        # 5. Transformer
        tokens = self.transformer(tokens)

        # 6. Extract point tokens (last N)
        point_tokens = tokens[:, 2:, :]

        # 7. Output head -> (B, N, 2)
        return self.output_proj(self.output_norm(point_tokens))
