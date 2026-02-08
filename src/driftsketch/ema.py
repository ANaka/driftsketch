"""Exponential Moving Average (EMA) of model parameters."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMA:
    """Maintains an exponential moving average of model parameters.

    Usage::

        ema = EMA(model, decay=0.9999)
        # In training loop:
        ema.update()
        # For sampling:
        ema.apply()
        model(...)  # uses EMA weights
        ema.restore()
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.model = model
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self) -> None:
        """Update shadow parameters with current model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].lerp_(param.data, 1.0 - self.decay)

    def apply(self) -> None:
        """Swap model parameters with shadow (EMA) parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """Restore model parameters from backup (undo apply)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return shadow parameters for checkpointing."""
        return copy.deepcopy(self.shadow)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load shadow parameters from a checkpoint."""
        self.shadow = copy.deepcopy(state_dict)
