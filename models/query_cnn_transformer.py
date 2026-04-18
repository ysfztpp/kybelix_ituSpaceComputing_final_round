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
    use_query_doy: bool = True
    use_time_doy: bool = True
    aux_feature_dim: int = 0
    aux_hidden_dim: int = 128


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
        if config.aux_feature_dim > 0:
            self.aux_mlp = nn.Sequential(
                nn.LayerNorm(config.aux_feature_dim),
                nn.Linear(config.aux_feature_dim, config.aux_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.LayerNorm(config.aux_hidden_dim),
            )
            aux_out_dim = config.aux_hidden_dim
        else:
            self.aux_mlp = None
            aux_out_dim = 0
        fused_dim = config.transformer_dim * 2 + aux_out_dim
        self.crop_head = nn.Sequential(nn.LayerNorm(fused_dim), nn.Dropout(config.dropout), nn.Linear(fused_dim, config.num_crop_classes))
        self.stage_head = nn.Sequential(nn.LayerNorm(fused_dim), nn.Dropout(config.dropout), nn.Linear(fused_dim, config.num_phenophase_classes))

    def forward(
        self,
        patches: torch.Tensor,
        time_mask: torch.Tensor,
        time_doy: torch.Tensor,
        query_doy: torch.Tensor,
        aux_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, timesteps, bands, height, width = patches.shape
        flat = patches.reshape(batch_size * timesteps, bands, height, width)
        encoded = self.cnn(flat).reshape(batch_size, timesteps, -1)
        x = self.input_proj(encoded)
        if self.config.use_time_doy:
            x = x + self.time_encoding(time_doy)
        x = self.transformer(x, src_key_padding_mask=~time_mask.bool())
        pooled = self.pool(x, time_mask.bool())
        if self.config.use_query_doy:
            query = self.query_encoding(query_doy).reshape(batch_size, -1)
        else:
            query = torch.zeros_like(pooled)
        pieces = [pooled, query]
        if self.aux_mlp is not None:
            if aux_features is None:
                raise ValueError("aux_features must be provided when aux_feature_dim > 0")
            pieces.append(self.aux_mlp(aux_features.float()))
        fused = torch.cat(pieces, dim=1)
        return {"crop_logits": self.crop_head(fused), "stage_logits": self.stage_head(fused)}
