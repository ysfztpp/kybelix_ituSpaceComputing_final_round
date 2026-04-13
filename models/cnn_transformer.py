from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("torch is required for model code. Install PyTorch before training.") from exc

from .cnn_encoder import PatchCNNEncoder
from .temporal_transformer import DayOfYearEncoding, MaskedTemporalPool


@dataclass(frozen=True)
class CNNTransformerConfig:
    in_channels: int = 12
    patch_size: int = 15
    cnn_embedding_dim: int = 128
    transformer_dim: int = 128
    transformer_layers: int = 3
    attention_heads: int = 4
    dropout: float = 0.15
    num_crop_classes: int = 3
    num_phenophases: int = 7


class CNNTransformerBaseline(nn.Module):
    """CNN per timestep, Transformer over acquisition dates, two prediction heads."""

    def __init__(self, config: CNNTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.cnn = PatchCNNEncoder(config.in_channels, config.cnn_embedding_dim, config.dropout)
        self.input_proj = nn.Identity() if config.cnn_embedding_dim == config.transformer_dim else nn.Linear(config.cnn_embedding_dim, config.transformer_dim)
        self.time_encoding = DayOfYearEncoding(config.transformer_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.transformer_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.transformer_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.transformer_layers)
        self.pool = MaskedTemporalPool()
        self.crop_head = nn.Sequential(nn.LayerNorm(config.transformer_dim), nn.Dropout(config.dropout), nn.Linear(config.transformer_dim, config.num_crop_classes))
        self.phenophase_head = nn.Sequential(nn.LayerNorm(config.transformer_dim), nn.Dropout(config.dropout), nn.Linear(config.transformer_dim, config.num_phenophases), nn.Sigmoid())

    def forward(self, patches: torch.Tensor, time_mask: torch.Tensor, time_doy: torch.Tensor) -> dict[str, torch.Tensor]:
        # patches: [batch, time, band, height, width]
        batch_size, timesteps, bands, height, width = patches.shape
        flat = patches.reshape(batch_size * timesteps, bands, height, width)
        encoded = self.cnn(flat).reshape(batch_size, timesteps, -1)
        x = self.input_proj(encoded) + self.time_encoding(time_doy)
        key_padding_mask = ~time_mask.bool()
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        pooled = self.pool(x, time_mask.bool())
        return {
            "crop_logits": self.crop_head(pooled),
            "phenophase_norm": self.phenophase_head(pooled),
        }
