from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("torch is required for model code. Install PyTorch before training.") from exc

from preprocessing.constants import PHENOPHASE_DOY_RANK_TO_STAGE_ID

from .cnn_encoder import PatchCNNEncoder
from .temporal_transformer import build_time_encoding


def _masked_mean_pool(x: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
    weights = time_mask.float().unsqueeze(-1)
    summed = (x * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return summed / denom


def _masked_max_pool(x: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
    invalid_fill = torch.finfo(x.dtype).min
    masked = x.masked_fill(~time_mask.bool().unsqueeze(-1), invalid_fill)
    pooled = masked.max(dim=1).values
    all_invalid = ~time_mask.bool().any(dim=1, keepdim=True)
    return torch.where(all_invalid, torch.zeros_like(pooled), pooled)


@dataclass(frozen=True)
class SeasonTimelineTransformerConfig:
    in_channels: int = 12
    patch_size: int = 15
    cnn_embedding_dim: int = 256
    transformer_dim: int = 256
    transformer_layers: int = 4
    attention_heads: int = 8
    dropout: float = 0.20
    num_crop_classes: int = 3
    num_phenophase_classes: int = 7
    use_time_doy: bool = True
    time_encoding_type: str = "fourier"
    time_encoding_harmonics: int = 6
    timeline_hidden_dim: int = 128
    stage_temperature_days: float = 12.0
    max_first_boundary_offset_days: float = 120.0
    min_transition_span_days: float = 45.0
    max_transition_span_days: float = 210.0
    min_transition_gap_days: float = 7.0


class SeasonTimelineTransformerClassifier(nn.Module):
    """Predict crop directly and rice phenology via an image-derived seasonal timeline.

    The stage branch does not encode query day into the image representation.
    Instead it predicts ordered stage-transition DOYs from the time series and
    converts the provided query_doy into stage logits only at the final lookup.
    """

    def __init__(self, config: SeasonTimelineTransformerConfig) -> None:
        super().__init__()
        if config.num_phenophase_classes != 7:
            raise ValueError("SeasonTimelineTransformerClassifier expects 7 phenophase classes")
        self.config = config
        self.cnn = PatchCNNEncoder(config.in_channels, config.cnn_embedding_dim, config.dropout)
        self.input_proj = nn.Identity() if config.cnn_embedding_dim == config.transformer_dim else nn.Linear(config.cnn_embedding_dim, config.transformer_dim)
        self.time_encoding = build_time_encoding(config.time_encoding_type, config.transformer_dim, harmonics=config.time_encoding_harmonics)
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
        feature_dim = config.transformer_dim * 2
        self.crop_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(config.dropout),
            nn.Linear(feature_dim, config.num_crop_classes),
        )
        self.timeline_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, config.timeline_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.timeline_hidden_dim, 7),
        )

    def _predict_stage_boundaries(self, stage_features: torch.Tensor, time_doy: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        params = self.timeline_head(stage_features)
        first_boundary_raw = params[:, 0]
        total_span_raw = params[:, 1]
        gap_logits = params[:, 2:]

        time_weights = time_mask.float()
        time_reference = (time_doy.float() * time_weights).sum(dim=1) / time_weights.sum(dim=1).clamp_min(1.0)
        first_boundary = time_reference + float(self.config.max_first_boundary_offset_days) * torch.tanh(first_boundary_raw)

        min_span = float(self.config.min_transition_span_days)
        max_span = float(self.config.max_transition_span_days)
        total_span = min_span + (max_span - min_span) * torch.sigmoid(total_span_raw)

        min_gap = float(self.config.min_transition_gap_days)
        available_gap_mass = (total_span - min_gap * gap_logits.shape[1]).clamp_min(0.0)
        gap_weights = torch.softmax(gap_logits, dim=1)
        gaps = min_gap + gap_weights * available_gap_mass.unsqueeze(1)
        cumulative_gaps = torch.cumsum(gaps, dim=1)
        return torch.cat([first_boundary.unsqueeze(1), first_boundary.unsqueeze(1) + cumulative_gaps], dim=1)

    def _timeline_logits(self, boundaries: torch.Tensor, query_doy: torch.Tensor, query_doy_mask: torch.Tensor | None) -> torch.Tensor:
        query = query_doy.float()
        if query_doy_mask is not None:
            mask = query_doy_mask.float().reshape(-1)
            query = query * mask + boundaries[:, 0] * (1.0 - mask)
        query = query.unsqueeze(1)
        temperature = max(float(self.config.stage_temperature_days), 1e-3)
        transitions = torch.sigmoid((query - boundaries) / temperature)
        probs_chronological = torch.cat(
            [
                1.0 - transitions[:, :1],
                transitions[:, :-1] - transitions[:, 1:],
                transitions[:, -1:],
            ],
            dim=1,
        ).clamp_min(1e-6)
        probs_chronological = probs_chronological / probs_chronological.sum(dim=1, keepdim=True)
        logits_chronological = torch.log(probs_chronological)
        logits = torch.empty_like(logits_chronological)
        for doy_rank, stage_id in enumerate(PHENOPHASE_DOY_RANK_TO_STAGE_ID):
            logits[:, stage_id] = logits_chronological[:, doy_rank]
        return logits

    def forward(
        self,
        patches: torch.Tensor,
        time_mask: torch.Tensor,
        time_doy: torch.Tensor,
        query_doy: torch.Tensor,
        aux_features: torch.Tensor | None = None,
        query_doy_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del aux_features  # This model intentionally avoids query-derived auxiliary shortcuts.

        batch_size, timesteps, bands, height, width = patches.shape
        flat = patches.reshape(batch_size * timesteps, bands, height, width)
        encoded = self.cnn(flat).reshape(batch_size, timesteps, -1)
        x = self.input_proj(encoded)
        if self.config.use_time_doy:
            x = x + self.time_encoding(time_doy)
        x = self.transformer(x, src_key_padding_mask=~time_mask.bool())

        pooled_mean = _masked_mean_pool(x, time_mask.bool())
        pooled_max = _masked_max_pool(x, time_mask.bool())
        features = torch.cat([pooled_mean, pooled_max], dim=1)

        crop_logits = self.crop_head(features)
        stage_boundaries = self._predict_stage_boundaries(features, time_doy, time_mask.bool())
        stage_logits = self._timeline_logits(stage_boundaries, query_doy, query_doy_mask)
        return {
            "crop_logits": crop_logits,
            "stage_logits": stage_logits,
            "stage_transition_doys": stage_boundaries,
        }
