from __future__ import annotations

import math
from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("torch is required for model code. Install PyTorch before training.") from exc

from .temporal_transformer import build_time_encoding


def _build_encoder(dim: int, layers: int, heads: int, dropout: float, mlp_ratio: float) -> nn.TransformerEncoder:
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=heads,
        dim_feedforward=int(dim * mlp_ratio),
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=False,
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


class SignedRelativeFourierEncoding(nn.Module):
    """Encode signed query-to-acquisition lags with learnable Fourier features."""

    def __init__(self, dim: int, harmonics: int = 8) -> None:
        super().__init__()
        self.harmonics = max(1, int(harmonics))
        scales = torch.tensor([2.0 * math.pi * k / 366.0 for k in range(1, self.harmonics + 1)], dtype=torch.float32)
        self.freqs = nn.Parameter(scales)
        self.phases = nn.Parameter(torch.zeros(self.harmonics, dtype=torch.float32))
        self.proj = nn.Sequential(nn.Linear(self.harmonics * 2 + 1, dim), nn.GELU(), nn.LayerNorm(dim))

    def forward(self, relative_days: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        values = relative_days.float().clamp(min=-366.0, max=366.0)
        angles = values.unsqueeze(-1) * self.freqs + self.phases
        signed = values.unsqueeze(-1) / 366.0
        enc = torch.cat([signed, torch.sin(angles), torch.cos(angles)], dim=-1)
        projected = self.proj(enc)
        if valid_mask is None:
            return projected
        return projected * valid_mask.unsqueeze(-1).float()


@dataclass(frozen=True)
class QueryTSViTConfig:
    in_channels: int = 12
    patch_size: int = 15
    token_patch_size: int = 3
    transformer_dim: int = 320
    temporal_layers: int = 6
    spatial_layers: int = 6
    attention_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.15
    num_crop_classes: int = 3
    num_phenophase_classes: int = 7
    use_query_doy: bool = True
    use_time_doy: bool = True
    time_encoding_type: str = "fourier"
    query_encoding_type: str = "fourier"
    time_encoding_harmonics: int = 8
    aux_feature_dim: int = 0
    aux_hidden_dim: int = 128
    aux_target: str = "shared"
    use_relative_query_bias: bool = True
    use_direct_query_logit_bias: bool = True


class QueryTSViTClassifier(nn.Module):
    """Factorized temporo-spatial ViT for Sentinel-2 patch time series."""

    def __init__(self, config: QueryTSViTConfig) -> None:
        super().__init__()
        if config.patch_size % config.token_patch_size != 0:
            raise ValueError("patch_size must be divisible by token_patch_size")
        self.config = config
        self.grid_size = config.patch_size // config.token_patch_size
        self.num_spatial_tokens = self.grid_size * self.grid_size
        self.patch_embed = nn.Conv2d(
            config.in_channels,
            config.transformer_dim,
            kernel_size=config.token_patch_size,
            stride=config.token_patch_size,
            bias=False,
        )
        self.token_norm = nn.LayerNorm(config.transformer_dim)
        self.time_encoding = build_time_encoding(
            config.time_encoding_type,
            config.transformer_dim,
            harmonics=config.time_encoding_harmonics,
        )
        self.query_encoding = build_time_encoding(
            config.query_encoding_type,
            config.transformer_dim,
            harmonics=config.time_encoding_harmonics,
        )
        self.relative_query_encoding = SignedRelativeFourierEncoding(
            config.transformer_dim,
            harmonics=config.time_encoding_harmonics,
        )
        self.temporal_encoder = _build_encoder(
            config.transformer_dim,
            config.temporal_layers,
            config.attention_heads,
            config.dropout,
            config.mlp_ratio,
        )
        self.crop_temporal_query = nn.Parameter(torch.randn(1, 1, config.transformer_dim) * 0.02)
        self.stage_temporal_query = nn.Parameter(torch.randn(1, 1, config.transformer_dim) * 0.02)
        self.crop_temporal_pool = nn.MultiheadAttention(
            config.transformer_dim,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.stage_temporal_pool = nn.MultiheadAttention(
            config.transformer_dim,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.spatial_pos = nn.Parameter(torch.randn(1, self.num_spatial_tokens, config.transformer_dim) * 0.02)
        self.crop_class_tokens = nn.Parameter(torch.randn(1, config.num_crop_classes, config.transformer_dim) * 0.02)
        self.stage_class_tokens = nn.Parameter(torch.randn(1, config.num_phenophase_classes, config.transformer_dim) * 0.02)
        self.crop_spatial_encoder = _build_encoder(
            config.transformer_dim,
            config.spatial_layers,
            config.attention_heads,
            config.dropout,
            config.mlp_ratio,
        )
        self.stage_spatial_encoder = _build_encoder(
            config.transformer_dim,
            config.spatial_layers,
            config.attention_heads,
            config.dropout,
            config.mlp_ratio,
        )
        aux_target = str(config.aux_target or "shared").lower()
        if aux_target not in {"shared", "stage_only", "crop_only"}:
            raise ValueError("aux_target must be one of: shared, stage_only, crop_only")
        self.aux_target = aux_target
        if config.aux_feature_dim > 0:
            self.aux_mlp = nn.Sequential(
                nn.LayerNorm(config.aux_feature_dim),
                nn.Linear(config.aux_feature_dim, config.aux_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.LayerNorm(config.aux_hidden_dim),
            )
            if aux_target in {"shared", "crop_only"}:
                self.crop_aux_proj = nn.Linear(config.aux_hidden_dim, config.transformer_dim)
            else:
                self.crop_aux_proj = None
            if aux_target in {"shared", "stage_only"}:
                self.stage_aux_proj = nn.Linear(config.aux_hidden_dim, config.transformer_dim)
            else:
                self.stage_aux_proj = None
        else:
            self.aux_mlp = None
            self.crop_aux_proj = None
            self.stage_aux_proj = None
        self.crop_head = nn.Sequential(
            nn.LayerNorm(config.transformer_dim),
            nn.Linear(config.transformer_dim, 1),
        )
        self.stage_head = nn.Sequential(
            nn.LayerNorm(config.transformer_dim),
            nn.Linear(config.transformer_dim, 1),
        )
        if config.use_direct_query_logit_bias and config.use_query_doy:
            direct_query_dim = config.transformer_dim * 2
            self.crop_query_logit_bias = nn.Sequential(
                nn.LayerNorm(direct_query_dim),
                nn.Linear(direct_query_dim, config.num_crop_classes),
            )
            self.stage_query_logit_bias = nn.Sequential(
                nn.LayerNorm(direct_query_dim),
                nn.Linear(direct_query_dim, config.num_phenophase_classes),
            )
        else:
            self.crop_query_logit_bias = None
            self.stage_query_logit_bias = None

    def _encode_query_context(
        self,
        time_doy: torch.Tensor,
        time_mask: torch.Tensor,
        query_doy: torch.Tensor,
        query_doy_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = time_doy.shape[0]
        if not self.config.use_query_doy:
            zeros = torch.zeros(batch_size, self.config.transformer_dim, device=time_doy.device, dtype=time_doy.dtype)
            return zeros, zeros
        query_context = self.query_encoding(query_doy)
        if query_doy_mask is not None:
            query_context = query_context * query_doy_mask.float().reshape(batch_size, 1)
        if self.config.use_relative_query_bias:
            rel_days = query_doy.unsqueeze(1) - time_doy
            rel_context = self.relative_query_encoding(rel_days, time_mask.bool())
            rel_context = (rel_context * time_mask.unsqueeze(-1).float()).sum(dim=1) / time_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            if query_doy_mask is not None:
                rel_context = rel_context * query_doy_mask.float().reshape(batch_size, 1)
        else:
            rel_context = torch.zeros_like(query_context)
        return query_context, rel_context

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
        tokens = self.patch_embed(flat)
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens = self.token_norm(tokens)
        tokens = tokens.reshape(batch_size, timesteps, self.num_spatial_tokens, self.config.transformer_dim)
        tokens = tokens.permute(0, 2, 1, 3).reshape(batch_size * self.num_spatial_tokens, timesteps, self.config.transformer_dim)

        if self.config.use_time_doy:
            time_context = self.time_encoding(time_doy).unsqueeze(1).expand(-1, self.num_spatial_tokens, -1, -1)
            time_context = time_context.reshape(batch_size * self.num_spatial_tokens, timesteps, self.config.transformer_dim)
            tokens = tokens + time_context

        query_context, relative_context = self._encode_query_context(time_doy, time_mask, query_doy, query_doy_mask)
        if self.config.use_query_doy and self.config.use_relative_query_bias:
            rel_days = query_doy.unsqueeze(1) - time_doy
            rel_bias = self.relative_query_encoding(rel_days, time_mask.bool()).unsqueeze(1).expand(-1, self.num_spatial_tokens, -1, -1)
            rel_bias = rel_bias.reshape(batch_size * self.num_spatial_tokens, timesteps, self.config.transformer_dim)
            tokens = tokens + rel_bias

        temporal_mask = ~time_mask.bool().unsqueeze(1).expand(-1, self.num_spatial_tokens, -1).reshape(batch_size * self.num_spatial_tokens, timesteps)
        temporal_features = self.temporal_encoder(tokens, src_key_padding_mask=temporal_mask)

        crop_query = (self.crop_temporal_query + query_context.unsqueeze(1) + relative_context.unsqueeze(1)).repeat_interleave(self.num_spatial_tokens, dim=0)
        crop_sites, _ = self.crop_temporal_pool(crop_query, temporal_features, temporal_features, key_padding_mask=temporal_mask)
        crop_sites = crop_sites.reshape(batch_size, self.num_spatial_tokens, self.config.transformer_dim)

        stage_query = (self.stage_temporal_query + query_context.unsqueeze(1) + relative_context.unsqueeze(1)).repeat_interleave(self.num_spatial_tokens, dim=0)
        stage_sites, _ = self.stage_temporal_pool(stage_query, temporal_features, temporal_features, key_padding_mask=temporal_mask)
        stage_sites = stage_sites.reshape(batch_size, self.num_spatial_tokens, self.config.transformer_dim)

        crop_sites = crop_sites + self.spatial_pos
        stage_sites = stage_sites + self.spatial_pos
        crop_tokens = torch.cat([self.crop_class_tokens.expand(batch_size, -1, -1), crop_sites], dim=1)
        stage_tokens = torch.cat([self.stage_class_tokens.expand(batch_size, -1, -1), stage_sites], dim=1)

        crop_encoded = self.crop_spatial_encoder(crop_tokens)[:, : self.config.num_crop_classes]
        stage_encoded = self.stage_spatial_encoder(stage_tokens)[:, : self.config.num_phenophase_classes]

        if self.aux_mlp is not None:
            if aux_features is None:
                raise ValueError("aux_features must be provided when aux_feature_dim > 0")
            aux = self.aux_mlp(aux_features.float())
            if self.crop_aux_proj is not None:
                crop_encoded = crop_encoded + self.crop_aux_proj(aux).unsqueeze(1)
            if self.stage_aux_proj is not None:
                stage_encoded = stage_encoded + self.stage_aux_proj(aux).unsqueeze(1)

        crop_logits = self.crop_head(crop_encoded).squeeze(-1)
        stage_logits = self.stage_head(stage_encoded).squeeze(-1)
        if self.crop_query_logit_bias is not None and self.stage_query_logit_bias is not None:
            direct_query = torch.cat([query_context, relative_context], dim=1)
            crop_logits = crop_logits + self.crop_query_logit_bias(direct_query)
            stage_logits = stage_logits + self.stage_query_logit_bias(direct_query)
        return {"crop_logits": crop_logits, "stage_logits": stage_logits}
