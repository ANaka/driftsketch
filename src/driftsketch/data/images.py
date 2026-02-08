"""Image-only dataset for distillation training."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class ImageDataset(Dataset):
    """Load images from any directory (flat or ImageFolder-style).

    Recursively finds all image files and returns transformed tensors.
    No labels or paired sketches — used for distillation with CLIP perceptual loss.
    """

    def __init__(
        self,
        root: str | Path,
        transform: callable | None = None,
        extensions: tuple[str, ...] = _IMAGE_EXTENSIONS,
    ):
        self.root = Path(root)
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(f"Image directory not found: {self.root}")

        self.paths: list[Path] = sorted(
            p for p in self.root.rglob("*") if p.suffix.lower() in extensions
        )
        if not self.paths:
            raise ValueError(
                f"No images found in {self.root} with extensions {extensions}"
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {"image": img}
