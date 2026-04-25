from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("torch is required for model code. Install PyTorch before training.") from exc

from .cnn_encoder import PatchCNNEncoder
from .temporal_transformer import MaskedTemporalPool, build_time_encoding


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
    time_encoding_type: str = "sincos"
    query_encoding_type: str = "sincos"
    time_encoding_harmonics: int = 4
    aux_feature_dim: int = 0
    aux_hidden_dim: int = 128
    aux_target: str = "shared"


class QueryCNNTransformerClassifier(nn.Module):
    """PDF-aligned model: full patch time series + query date -> crop + stage.

    The main path remains the real Sentinel-2 patch tensor. Optional auxiliary
    features are fused only after temporal pooling, so feature engineering can be
    tested without replacing the CNN/Transformer image-time-series learner.
    """

    def __init__(self, config: QueryCNNTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.cnn = PatchCNNEncoder(config.in_channels, config.cnn_embedding_dim, config.dropout)
        self.input_proj = nn.Identity() if config.cnn_embedding_dim == config.transformer_dim else nn.Linear(config.cnn_embedding_dim, config.transformer_dim)
        self.time_encoding = build_time_encoding(config.time_encoding_type, config.transformer_dim, harmonics=config.time_encoding_harmonics)
        self.query_encoding = build_time_encoding(config.query_encoding_type, config.transformer_dim, harmonics=config.time_encoding_harmonics)
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
        aux_target = str(config.aux_target or "shared").lower()
        if aux_target not in {"shared", "stage_only", "crop_only"}:
            raise ValueError("aux_target must be one of: shared, stage_only, crop_only")
        self.aux_target = aux_target
        if config.aux_feature_dim > 0:
            # Auxiliary features are compact phenology/index summaries. They are
            # deliberately kept in a small side branch to avoid dominating the
            # raw patch encoder.
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
        base_dim = config.transformer_dim * 2
        crop_dim = base_dim + aux_out_dim if aux_target in {"shared", "crop_only"} else base_dim
        stage_dim = base_dim + aux_out_dim if aux_target in {"shared", "stage_only"} else base_dim
        self.crop_head = nn.Sequential(nn.LayerNorm(crop_dim), nn.Dropout(config.dropout), nn.Linear(crop_dim, config.num_crop_classes))
        self.stage_head = nn.Sequential(nn.LayerNorm(stage_dim), nn.Dropout(config.dropout), nn.Linear(stage_dim, config.num_phenophase_classes))

    def forward(
        self,
        patches: torch.Tensor,
        time_mask: torch.Tensor,
        time_doy: torch.Tensor,
        query_doy: torch.Tensor,
        aux_features: torch.Tensor | None = None,
        query_doy_mask: torch.Tensor | None = None,
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
            if query_doy_mask is not None:
                query = query * query_doy_mask.float().reshape(batch_size, 1)
        else:
            query = torch.zeros_like(pooled)
        base = torch.cat([pooled, query], dim=1)
        crop_features = base
        stage_features = base
        if self.aux_mlp is not None:
            if aux_features is None:
                raise ValueError("aux_features must be provided when aux_feature_dim > 0")
            aux = self.aux_mlp(aux_features.float())
            if self.aux_target in {"shared", "crop_only"}:
                crop_features = torch.cat([crop_features, aux], dim=1)
            if self.aux_target in {"shared", "stage_only"}:
                stage_features = torch.cat([stage_features, aux], dim=1)
        return {"crop_logits": self.crop_head(crop_features), "stage_logits": self.stage_head(stage_features)}
