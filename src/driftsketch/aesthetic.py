"""LAION Aesthetic Predictor V2 wrapper for differentiable aesthetic scoring.

Uses CLIP ViT-L/14 image embeddings fed into a lightweight MLP trained on
~441K human-rated images (SAC + LAION-Logos + AVA). Outputs a continuous
aesthetic score on a 1-10 scale.

Both CLIP and the MLP are frozen, but gradients flow through the forward pass
(approach B) so the upstream renderer receives a gradient signal.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# CLIP image normalization constants (ImageNet stats used by OpenAI CLIP)
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_WEIGHTS_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor"
    "/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
)
_WEIGHTS_FILENAME = "sac+logos+ava1-l14-linearMSE.pth"


class AestheticMLP(nn.Module):
    """Fully linear MLP matching the LAION improved-aesthetic-predictor architecture.

    No activation functions between layers (the 'linearMSE' variant).
    Input: L2-normalized CLIP ViT-L/14 embeddings (768-dim).
    Output: scalar aesthetic score (~1-10 range).
    """

    def __init__(self, input_size: int = 768) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class AestheticScorer(nn.Module):
    """Differentiable aesthetic scorer using CLIP ViT-L/14 + LAION aesthetic MLP.

    Wraps a frozen CLIP image encoder and frozen aesthetic MLP. Gradients flow
    through the CLIP forward pass (approach B from the design doc) so that
    upstream differentiable renderers receive gradient signal.

    Typical usage::

        scorer = AestheticScorer(device="cuda")
        loss, scores = scorer.compute_loss(grayscale_images)
        loss.backward()  # gradients flow through CLIP to rendered images

    Args:
        model_path: Path to the aesthetic MLP weights. If ``None``, weights
            are auto-downloaded from GitHub to the ``weights/`` directory.
        clip_model_name: open_clip model name (default ``"ViT-L-14"``).
        clip_pretrained: open_clip pretrained weight tag (default ``"openai"``).
        device: Device to load models onto.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        clip_model_name: str = "ViT-L-14",
        clip_pretrained: str = "openai",
        device: str = "cpu",
    ) -> None:
        super().__init__()

        # -- Load CLIP ViT-L/14 (image encoder only, frozen, fp16) ----------
        import open_clip

        clip_model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained
        )
        self.clip_visual = clip_model.visual.to(device=device, dtype=torch.float16)
        self.clip_visual.requires_grad_(False)
        self.clip_visual.eval()

        # -- Load aesthetic MLP (fp32) ---------------------------------------
        self.aesthetic_mlp = AestheticMLP(input_size=768)
        weights_path = self._resolve_weights(model_path)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.aesthetic_mlp.load_state_dict(state_dict)
        self.aesthetic_mlp.to(device=device, dtype=torch.float32)
        self.aesthetic_mlp.requires_grad_(False)
        self.aesthetic_mlp.eval()

        # -- Store CLIP normalization as buffers (move with .to()) -----------
        self.register_buffer(
            "_clip_mean",
            torch.tensor(_CLIP_MEAN, dtype=torch.float32, device=device).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_clip_std",
            torch.tensor(_CLIP_STD, dtype=torch.float32, device=device).view(1, 3, 1, 1),
        )

        self._device = device
        logger.info(
            "AestheticScorer loaded: CLIP %s (%s) fp16, MLP fp32 on %s",
            clip_model_name,
            clip_pretrained,
            device,
        )

    @staticmethod
    def _resolve_weights(model_path: Optional[str]) -> Path:
        """Resolve the aesthetic MLP weights path, downloading if needed."""
        if model_path is not None:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"Aesthetic MLP weights not found at {path}"
                )
            return path

        # Auto-download to weights/ directory
        weights_dir = Path("weights")
        weights_dir.mkdir(parents=True, exist_ok=True)
        path = weights_dir / _WEIGHTS_FILENAME

        if not path.exists():
            logger.info("Downloading aesthetic predictor weights to %s ...", path)
            torch.hub.download_url_to_file(_WEIGHTS_URL, str(path))
            logger.info("Download complete.")

        return path

    def prepare_images(
        self, grayscale_images: Tensor, target_size: int = 224
    ) -> Tensor:
        """Convert grayscale rendered images to CLIP-ready RGB tensors.

        Args:
            grayscale_images: ``(B, H, W)`` grayscale images in [0, 1] from
                pydiffvg rendering.
            target_size: Spatial resolution for CLIP input (default 224).

        Returns:
            ``(B, 3, target_size, target_size)`` normalized RGB tensor.
        """
        # (B, H, W) -> (B, 3, H, W) by expanding grayscale to RGB
        images = grayscale_images.unsqueeze(1).expand(-1, 3, -1, -1)

        # Resize if needed (bilinear for differentiability)
        _, _, h, w = images.shape
        if h != target_size or w != target_size:
            images = F.interpolate(
                images,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )

        # Apply CLIP normalization (differentiable)
        images = (images - self._clip_mean) / self._clip_std

        return images

    def score(self, images: Tensor) -> Tensor:
        """Compute aesthetic scores for preprocessed images.

        Gradients flow through the CLIP forward pass (approach B). CLIP
        parameters are frozen but the computation graph is preserved so
        upstream renderers receive gradient signal.

        Args:
            images: ``(B, 3, H, W)`` preprocessed (CLIP-normalized) images.

        Returns:
            ``(B, 1)`` aesthetic scores (~1-10 range).
        """
        # Cast to fp16 for CLIP, keeping gradient flow
        images_fp16 = images.to(dtype=torch.float16)

        # Forward through CLIP visual encoder — NO torch.no_grad()!
        # Parameters are frozen but we need the computation graph.
        clip_emb = self._encode_clip_image(images_fp16)  # (B, 768), fp16

        # L2-normalize embeddings
        clip_emb = F.normalize(clip_emb, p=2, dim=-1)

        # Aesthetic MLP in fp32
        scores = self.aesthetic_mlp(clip_emb.float())  # (B, 1)

        return scores

    def _encode_clip_image(self, images: Tensor) -> Tensor:
        """Run CLIP visual encoder, returning the pooled CLS embedding.

        Uses the open_clip visual encoder internals to get the projected
        image embedding (not the raw patch tokens).

        Args:
            images: ``(B, 3, H, W)`` in fp16.

        Returns:
            ``(B, 768)`` image embeddings.
        """
        v = self.clip_visual
        # open_clip ViT forward: patch embed → transformer → pool → project
        x = v(images)  # (B, embed_dim)
        return x

    def compute_loss(
        self, grayscale_images: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Compute aesthetic loss from grayscale rendered sketches.

        Calls :meth:`prepare_images` then :meth:`score`. Returns a loss
        (negative mean score, for minimization) and raw scores for logging.

        Args:
            grayscale_images: ``(B, H, W)`` grayscale images in [0, 1].

        Returns:
            A tuple of ``(loss, scores)`` where ``loss`` is a scalar
            (``-scores.mean()``) and ``scores`` is ``(B, 1)``.
        """
        images = self.prepare_images(grayscale_images)
        scores = self.score(images)
        loss = -scores.mean()
        return loss, scores

    def train(self, mode: bool = True) -> "AestheticScorer":
        """Override to keep sub-modules always in eval mode."""
        # AestheticScorer itself doesn't have trainable params.
        # Keep CLIP and MLP in eval to preserve dropout/batchnorm behavior.
        super().train(mode)
        self.clip_visual.eval()
        self.aesthetic_mlp.eval()
        return self
