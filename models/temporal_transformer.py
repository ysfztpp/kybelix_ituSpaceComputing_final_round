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
        raw = time_doy.float()
        valid = raw > 0
        doy = torch.clamp(raw, min=1.0, max=366.0)
        angle = 2.0 * math.pi * doy / 366.0
        enc = torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)
        projected = self.proj(enc)
        return projected * valid.unsqueeze(-1)


class FourierDayOfYearEncoding(nn.Module):
    """Multi-harmonic Fourier encoding for day-of-year."""

    def __init__(self, dim: int, harmonics: int = 4) -> None:
        super().__init__()
        self.harmonics = max(1, int(harmonics))
        self.proj = nn.Sequential(nn.Linear(self.harmonics * 2, dim), nn.GELU(), nn.LayerNorm(dim))

    def forward(self, time_doy: torch.Tensor) -> torch.Tensor:
        raw = time_doy.float()
        valid = raw > 0
        doy = torch.clamp(raw, min=1.0, max=366.0)
        base_angle = 2.0 * math.pi * doy / 366.0
        features: list[torch.Tensor] = []
        for harmonic in range(1, self.harmonics + 1):
            angle = harmonic * base_angle
            features.extend([torch.sin(angle), torch.cos(angle)])
        enc = torch.stack(features, dim=-1)
        projected = self.proj(enc)
        return projected * valid.unsqueeze(-1)


class Time2VecDayOfYearEncoding(nn.Module):
    """Time2Vec-style encoding with one linear and several periodic channels."""

    def __init__(self, dim: int, periodic_dims: int = 7) -> None:
        super().__init__()
        self.periodic_dims = max(1, int(periodic_dims))
        self.linear_weight = nn.Parameter(torch.randn(1))
        self.linear_bias = nn.Parameter(torch.zeros(1))
        self.periodic_weight = nn.Parameter(torch.randn(self.periodic_dims))
        self.periodic_bias = nn.Parameter(torch.zeros(self.periodic_dims))
        self.proj = nn.Sequential(nn.Linear(self.periodic_dims + 1, dim), nn.GELU(), nn.LayerNorm(dim))

    def forward(self, time_doy: torch.Tensor) -> torch.Tensor:
        raw = time_doy.float()
        valid = raw > 0
        scaled = torch.clamp(raw, min=1.0, max=366.0) / 366.0
        linear = scaled.unsqueeze(-1) * self.linear_weight + self.linear_bias
        periodic = torch.sin(scaled.unsqueeze(-1) * self.periodic_weight + self.periodic_bias)
        enc = torch.cat([linear, periodic], dim=-1)
        projected = self.proj(enc)
        return projected * valid.unsqueeze(-1)


class LearnableFourierEncoding(nn.Module):
    """Fourier encoding with learnable frequencies and phase offsets.

    Unlike Time2Vec there is no linear term, preventing the model from
    learning a monotone DOY→stage mapping. Frequencies are initialised from
    standard Fourier harmonics but are free to shift during training.
    """

    def __init__(self, dim: int, harmonics: int = 6) -> None:
        super().__init__()
        H = max(1, int(harmonics))
        init_freqs = torch.tensor([2.0 * math.pi * k / 366.0 for k in range(1, H + 1)])
        self.freqs = nn.Parameter(init_freqs)
        self.phases = nn.Parameter(torch.zeros(H))
        self.proj = nn.Sequential(nn.Linear(H * 2, dim), nn.GELU(), nn.LayerNorm(dim))

    def forward(self, time_doy: torch.Tensor) -> torch.Tensor:
        raw = time_doy.float()
        valid = raw > 0
        doy = torch.clamp(raw, min=1.0, max=366.0)
        angles = doy.unsqueeze(-1) * self.freqs + self.phases  # [..., H]
        enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.proj(enc) * valid.unsqueeze(-1)


def build_time_encoding(encoding_type: str, dim: int, harmonics: int = 4) -> nn.Module:
    mode = str(encoding_type or "sincos").strip().lower()
    if mode in {"sincos", "default", "mlp_sincos"}:
        return DayOfYearEncoding(dim)
    if mode in {"fourier", "multi_fourier"}:
        return FourierDayOfYearEncoding(dim, harmonics=harmonics)
    if mode in {"time2vec", "t2v"}:
        periodic_dims = max(2, harmonics * 2 - 1)
        return Time2VecDayOfYearEncoding(dim, periodic_dims=periodic_dims)
    if mode in {"learnable_fourier", "lf"}:
        return LearnableFourierEncoding(dim, harmonics=harmonics)
    raise ValueError(f"unsupported time encoding type: {encoding_type}")


class MaskedTemporalPool(nn.Module):
    def forward(self, x: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        weights = time_mask.float().unsqueeze(-1)
        summed = (x * weights).sum(dim=1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return summed / denom
