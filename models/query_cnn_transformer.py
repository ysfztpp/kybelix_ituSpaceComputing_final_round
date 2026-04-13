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
class QueryCNNTransformerConfig:
    in_channels: int = 12
    patch_size: int = 15
    cnn_embedding_dim: int = 256
    transformer_dim: int = 256
    transformer_layers: int = 4
    attention_heads: int = 8
    dropout: float = 0.20
    num_crop_classes: int = 3
    num_phenophase_classes: int = 7


class QueryCNNTransformerClassifier(nn.Module):
    """PDF-aligned model: full time series + query date -> crop class + stage class."""

    def __init__(self, config: QueryCNNTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.cnn = PatchCNNEncoder(config.in_channels, config.cnn_embedding_dim, config.dropout)
        self.input_proj = nn.Identity() if config.cnn_embedding_dim == config.transformer_dim else nn.Linear(config.cnn_embedding_dim, config.transformer_dim)
        self.time_encoding = DayOfYearEncoding(config.transformer_dim)
        self.query_encoding = DayOfYearEncoding(config.transformer_dim)
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
        fused_dim = config.transformer_dim * 2
        self.crop_head = nn.Sequential(nn.LayerNorm(fused_dim), nn.Dropout(config.dropout), nn.Linear(fused_dim, config.num_crop_classes))
        self.stage_head = nn.Sequential(nn.LayerNorm(fused_dim), nn.Dropout(config.dropout), nn.Linear(fused_dim, config.num_phenophase_classes))

    def forward(self, patches: torch.Tensor, time_mask: torch.Tensor, time_doy: torch.Tensor, query_doy: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, timesteps, bands, height, width = patches.shape
        flat = patches.reshape(batch_size * timesteps, bands, height, width)
        encoded = self.cnn(flat).reshape(batch_size, timesteps, -1)
        x = self.input_proj(encoded) + self.time_encoding(time_doy)
        x = self.transformer(x, src_key_padding_mask=~time_mask.bool())
        pooled = self.pool(x, time_mask.bool())
        query = self.query_encoding(query_doy).reshape(batch_size, -1)
        fused = torch.cat([pooled, query], dim=1)
        return {"crop_logits": self.crop_head(fused), "stage_logits": self.stage_head(fused)}
