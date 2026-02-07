"""Frozen CLIP image encoder for extracting patch-level features."""

import torch
import torch.nn as nn
from torch import Tensor


class FrozenCLIPImageEncoder(nn.Module):
    """Wraps an open_clip ViT to extract frozen patch-level image features.

    Outputs (B, 50, 768) for 224x224 input — 1 CLS token + 49 patches (7x7 grid).
    All parameters are frozen; this module is inference-only.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
    ):
        super().__init__()
        import open_clip

        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.visual = clip_model.visual
        self._preprocess = preprocess
        # Use the transformer width (768 for ViT-B-32), not the projection
        # output_dim (512). We want the raw patch-level hidden states.
        self._feature_dim = self.visual.conv1.out_channels

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def get_transform(self):
        """Return the CLIP preprocessing transform (resize, normalize, etc.)."""
        return self._preprocess

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        """Extract patch-level features from images.

        Runs the visual encoder up through the transformer and final LayerNorm,
        returning ALL token outputs (CLS + patches) rather than just the pooled
        CLS embedding.

        Args:
            images: (B, 3, 224, 224) preprocessed image tensors.

        Returns:
            (B, L, D) patch features where L = 1 + grid_h * grid_w
            (e.g. L=50 for ViT-B-32 with 224x224 input), D=768.
        """
        v = self.visual

        # Patch embedding + CLS token + positional embedding + ln_pre
        x = v._embeds(images)  # (B, 1 + num_patches, width)

        # Transformer blocks
        x = v.transformer(x)  # (B, 1 + num_patches, width)

        # Final layer norm applied to all tokens
        x = v.ln_post(x)  # (B, 1 + num_patches, width)

        return x
