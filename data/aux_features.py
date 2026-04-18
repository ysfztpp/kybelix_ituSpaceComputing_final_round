from __future__ import annotations

from collections.abc import Sequence

import numpy as np


DEFAULT_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
INDEX_NAMES = ("ndvi", "evi", "ndmi", "nbr", "ndre", "savi", "gndvi")
STAT_NAMES = ("median", "std", "min", "max", "amplitude", "valid_ratio")
EPS = 1e-6


def aux_feature_names(bands: Sequence[str] = DEFAULT_BANDS) -> list[str]:
    names: list[str] = []
    for band in bands:
        names.extend([f"band_{band.lower()}_{stat}" for stat in STAT_NAMES])
    for index_name in INDEX_NAMES:
        names.extend([f"{index_name}_{stat}" for stat in STAT_NAMES])
    names.extend(["query_doy_scaled", "nearest_time_delta_scaled"])
    names.extend([f"nearest_band_{band.lower()}_median" for band in bands])
    names.extend([f"nearest_{index_name}" for index_name in INDEX_NAMES])
    return names


def aux_feature_dim(bands: Sequence[str] = DEFAULT_BANDS) -> int:
    return len(aux_feature_names(bands))


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    out = np.full_like(numerator, np.nan, dtype=np.float32)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > EPS)
    out[valid] = numerator[valid] / denominator[valid]
    return np.clip(out, -5.0, 5.0)


def _series_stats(series: np.ndarray) -> list[float]:
    valid = series[np.isfinite(series)]
    valid_ratio = float(valid.size / max(series.size, 1))
    if valid.size == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, valid_ratio]
    min_value = float(np.min(valid))
    max_value = float(np.max(valid))
    return [
        float(np.median(valid)),
        float(np.std(valid)),
        min_value,
        max_value,
        max_value - min_value,
        valid_ratio,
    ]


def _band_time_medians(patches: np.ndarray, valid_pixel_mask: np.ndarray, time_mask: np.ndarray) -> np.ndarray:
    timesteps, bands = patches.shape[:2]
    series = np.full((timesteps, bands), np.nan, dtype=np.float32)
    for t in range(timesteps):
        if not bool(time_mask[t]):
            continue
        for b in range(bands):
            valid = valid_pixel_mask[t, b].astype(bool)
            if valid.any():
                series[t, b] = float(np.median(patches[t, b][valid]))
    return series


def _index_series(band_series: np.ndarray, band_to_index: dict[str, int]) -> dict[str, np.ndarray]:
    def band(name: str) -> np.ndarray:
        return band_series[:, band_to_index[name]]

    blue = band("B02")
    green = band("B03")
    red = band("B04")
    red_edge = band("B05")
    nir = band("B08")
    narrow_nir = band("B8A")
    swir1 = band("B11")
    swir2 = band("B12")

    return {
        "ndvi": _safe_divide(nir - red, nir + red),
        "evi": _safe_divide(2.5 * (nir - red), nir + 6.0 * red - 7.5 * blue + 1.0),
        "ndmi": _safe_divide(nir - swir1, nir + swir1),
        "nbr": _safe_divide(nir - swir2, nir + swir2),
        "ndre": _safe_divide(narrow_nir - red_edge, narrow_nir + red_edge),
        "savi": _safe_divide(1.5 * (nir - red), nir + red + 0.5),
        "gndvi": _safe_divide(nir - green, nir + green),
    }


def compute_aux_features(
    patches: np.ndarray,
    valid_pixel_mask: np.ndarray,
    time_mask: np.ndarray,
    time_doy: np.ndarray,
    query_doy: float,
    bands: Sequence[str] = DEFAULT_BANDS,
) -> np.ndarray:
    """Build compact phenology/spectral features from the raw patch time series.

    The CNN still receives the real [T, B, H, W] patch tensor. These features are
    only an auxiliary branch for explicit vegetation-index and validity signals.
    """

    bands = tuple(str(band) for band in bands)
    band_to_index = {band: index for index, band in enumerate(bands)}
    missing = [band for band in DEFAULT_BANDS if band not in band_to_index]
    if missing:
        raise ValueError(f"aux feature extraction requires Sentinel-2 bands {missing}")

    time_mask_bool = time_mask.astype(bool)
    band_series = _band_time_medians(patches.astype(np.float32, copy=False), valid_pixel_mask.astype(bool), time_mask_bool)
    indices = _index_series(band_series, band_to_index)
    query_value = float(query_doy)
    if not np.isfinite(query_value):
        query_value = 183.0

    features: list[float] = []
    for band_index in range(len(bands)):
        features.extend(_series_stats(band_series[:, band_index]))
    for index_name in INDEX_NAMES:
        features.extend(_series_stats(indices[index_name]))

    valid_times = np.where(time_mask_bool & np.isfinite(time_doy.astype(np.float32)) & (time_doy.astype(np.float32) > 0))[0]
    features.append(float(np.clip(query_value, 1.0, 366.0) / 366.0))
    if valid_times.size:
        nearest_pos = int(valid_times[np.argmin(np.abs(time_doy[valid_times].astype(np.float32) - query_value))])
        delta_scaled = float(abs(float(time_doy[nearest_pos]) - query_value) / 366.0)
        nearest_bands = band_series[nearest_pos]
        nearest_indices = [indices[index_name][nearest_pos] for index_name in INDEX_NAMES]
    else:
        delta_scaled = 1.0
        nearest_bands = np.zeros(len(bands), dtype=np.float32)
        nearest_indices = [0.0 for _ in INDEX_NAMES]
    features.append(delta_scaled)
    features.extend(np.nan_to_num(nearest_bands, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32).tolist())
    features.extend(np.nan_to_num(np.asarray(nearest_indices, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).tolist())

    out = np.asarray(features, dtype=np.float32)
    expected = aux_feature_dim(bands)
    if out.shape[0] != expected:
        raise RuntimeError(f"aux feature dimension mismatch: got {out.shape[0]}, expected {expected}")
    return out
