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



class CLIPImageProjector(nn.Module):
    """Projects CLIP patch features to model embedding dimension."""

    def __init__(self, clip_dim: int = 768, embed_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(clip_dim),
            nn.Linear(clip_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)

class BezierSketchTransformer(nn.Module):
    """Conditional Flow Matching model for Bezier stroke sketch generation.

    Operates on cubic Bezier strokes (4 control points per stroke, flattened to 8 coords).
    Uses a TransformerDecoder with class-derived memory tokens for conditioning.
    """

    def __init__(
        self,
        num_strokes: int = 32,
        coords_per_stroke: int = 8,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 8,
        num_classes: int = 10,
        num_memory_tokens: int = 4,
        clip_dim: int = 0,
    ):
        super().__init__()
        self.num_strokes = num_strokes
        self.coords_per_stroke = coords_per_stroke
        self.embed_dim = embed_dim
        self.num_memory_tokens = num_memory_tokens
        self.clip_dim = clip_dim

        # Stroke embedding: Linear(coords_per_stroke -> embed_dim) + learnable positional encoding
        self.stroke_proj = nn.Linear(coords_per_stroke, embed_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, num_strokes, embed_dim) * 0.02
        )

        # Time embedding: sinusoidal features -> MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Class embedding -> memory tokens
        self.class_embed = nn.Embedding(num_classes, 128)
        self.class_to_memory = nn.Sequential(
            nn.Linear(128, embed_dim * num_memory_tokens),
            nn.SiLU(),
        )

        # CLIP image projection
        if clip_dim > 0:
            self.clip_proj = CLIPImageProjector(clip_dim, embed_dim)

        # Null memory for classifier-free guidance
        self.null_memory = nn.Parameter(
            torch.randn(1, num_memory_tokens, embed_dim) * 0.02
        )

        # Transformer decoder (pre-LN)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        self.memory_norm = nn.LayerNorm(embed_dim)

        # Output head
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, coords_per_stroke)

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
            memory: optional (B, L, embed_dim) external memory (e.g. CLIP features)
            clip_features: optional (B, L, clip_dim) raw CLIP patch features
            cfg_mask: optional (B,) bool tensor, True = drop conditioning

        Returns:
            (B, num_strokes, coords_per_stroke) predicted velocity field
        """
        # Flatten t to (B,) if needed
        t = t.view(-1)
        B = x.shape[0]

        # 1. Embed strokes: (B, 32, 8) -> (B, 32, embed_dim)
        stroke_emb = self.stroke_proj(x) + self.pos_encoding

        # 2. Embed time: (B,) -> (B, embed_dim) -> (B, 1, embed_dim)
        time_emb = self.time_mlp(self._sinusoidal_embedding(t)).unsqueeze(1)

        # 3. Build queries: [time_emb, stroke_embs] -> (B, 33, embed_dim)
        queries = torch.cat([time_emb, stroke_emb], dim=1)

        # 4. Build memory tokens
        if clip_features is not None and self.clip_dim > 0:
            memory = self.clip_proj(clip_features)  # (B, L, embed_dim)
        elif memory is None:
            class_emb = self.class_embed(class_label)  # (B, 128)
            memory = self.class_to_memory(class_emb)  # (B, embed_dim * num_memory_tokens)
            memory = memory.view(B, self.num_memory_tokens, self.embed_dim)

        # Apply CFG mask — replace conditioned memory with null_memory for masked samples
        if cfg_mask is not None:
            null_mem = self.null_memory.expand(B, -1, -1)  # (B, num_memory_tokens, embed_dim)
            # Handle different memory lengths (e.g., 50 CLIP tokens vs 4 null tokens)
            if null_mem.shape[1] != memory.shape[1]:
                null_mem = null_mem[:, :1, :].expand(B, memory.shape[1], -1)
            mask_expanded = cfg_mask.view(B, 1, 1)  # (B, 1, 1)
            memory = torch.where(mask_expanded, null_mem, memory)

        memory = self.memory_norm(memory)

        # 5. Transformer decoder
        out = self.transformer(tgt=queries, memory=memory)

        # 6. Extract stroke tokens (skip first time token)
        stroke_tokens = out[:, 1:, :]

        # 7. Output head -> (B, 32, 8)
        return self.output_proj(self.output_norm(stroke_tokens))
