from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .query_cnn_transformer import QueryCNNTransformerClassifier, QueryCNNTransformerConfig
from .query_tsvit import QueryTSViTClassifier, QueryTSViTConfig

MODEL_TYPES = {
    "query_cnn_transformer": (QueryCNNTransformerConfig, QueryCNNTransformerClassifier),
    "query_tsvit": (QueryTSViTConfig, QueryTSViTClassifier),
}


def normalize_model_type(value: str | None) -> str:
    model_type = str(value or "query_cnn_transformer").strip().lower()
    if model_type not in MODEL_TYPES:
        raise ValueError(f"unsupported model_type: {model_type}")
    return model_type


def build_model_config(model_type: str, model_config_data: dict[str, Any]):
    config_cls, _model_cls = MODEL_TYPES[normalize_model_type(model_type)]
    return config_cls(**{key: value for key, value in model_config_data.items() if key in config_cls.__annotations__})


def build_model(model_type: str, model_config):
    _config_cls, model_cls = MODEL_TYPES[normalize_model_type(model_type)]
    return model_cls(model_config)


def config_asdict(model_config) -> dict[str, Any]:
    return asdict(model_config)
