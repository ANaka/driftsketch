"""CLIP perceptual loss for distillation training.

Renders Bezier sketches, augments them (critical for avoiding adversarial solutions),
encodes with CLIP, and computes cosine similarity against target image features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

from driftsketch.rendering import render_batch_beziers


class CLIPPerceptualLoss(nn.Module):
    """Render beziers -> augment -> encode with CLIP -> cosine similarity loss.

    Augmentation is essential per CLIPDraw — without it, optimization finds
    adversarial patterns that score high on CLIP but look nothing like sketches.
    """

    def __init__(self, clip_encoder: nn.Module, canvas_size: int = 224, num_augmentations: int = 4):
        super().__init__()
        self.clip_encoder = clip_encoder
        self.canvas_size = canvas_size
        self.num_augmentations = num_augmentations

        # CLIP normalization (standard OpenCLIP values)
        self.aug = T.Compose([
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomRotation(15),
            T.RandomHorizontalFlip(0.5),
            T.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    def forward(
        self,
        beziers: torch.Tensor,
        target_clip_features: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute CLIP perceptual loss between rendered sketches and target features.

        Args:
            beziers: (K, 32, 4, 2) Bezier control points.
            target_clip_features: (K, L, D) CLIP features from target images.

        Returns:
            (loss, metrics_dict) where loss is scalar and metrics has logging info.
        """
        K = beziers.shape[0]

        # 1. Render to grayscale images
        rendered = render_batch_beziers(
            beziers, canvas_size=self.canvas_size, max_render=K
        )  # (K, H, W)

        # 2. Expand grayscale to 3-channel RGB
        rendered_rgb = rendered.unsqueeze(1).expand(-1, 3, -1, -1)  # (K, 3, H, W)

        # 3. Augment: apply augmentations to each image
        augmented = []
        for i in range(K):
            img = rendered_rgb[i]  # (3, H, W)
            for _ in range(self.num_augmentations):
                augmented.append(self.aug(img))
        augmented = torch.stack(augmented)  # (K * num_aug, 3, 224, 224)

        # 4. Encode with CLIP (WITH gradients — encoder params are frozen)
        sketch_features = self.clip_encoder(augmented)  # (K * num_aug, L, D)

        # Pool over token dimension
        sketch_pooled = sketch_features.mean(dim=1)  # (K * num_aug, D)

        # Average augmented features per sample
        sketch_pooled = sketch_pooled.view(K, self.num_augmentations, -1).mean(dim=1)  # (K, D)

        # Pool target features over token dimension
        target_pooled = target_clip_features.mean(dim=1)  # (K, D)

        # 5. Cosine similarity loss
        cos_sim = F.cosine_similarity(sketch_pooled, target_pooled, dim=-1)  # (K,)
        loss = (1 - cos_sim).mean()

        metrics = {
            "loss/clip_cosine": loss.item(),
            "metric/avg_cosine_sim": cos_sim.mean().item(),
        }

        return loss, metrics
