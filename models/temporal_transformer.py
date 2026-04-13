from __future__ import annotations

import math

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("torch is required for model code. Install PyTorch before training.") from exc


class DayOfYearEncoding(nn.Module):
    """Sin/cos day-of-year encoding projected into transformer dimension."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.LayerNorm(dim))

    def forward(self, time_doy: torch.Tensor) -> torch.Tensor:
        doy = torch.clamp(time_doy.float(), min=1.0, max=366.0)
        angle = 2.0 * math.pi * doy / 366.0
        enc = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)
        return self.proj(enc)


class MaskedTemporalPool(nn.Module):
    def forward(self, x: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        weights = time_mask.float().unsqueeze(-1)
        summed = (x * weights).sum(dim=1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return summed / denom
