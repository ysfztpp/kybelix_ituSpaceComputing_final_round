from __future__ import annotations

from collections.abc import Sequence

import numpy as np


DEFAULT_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
INDEX_NAMES = ("ndvi", "evi", "ndmi", "nbr", "ndre", "savi", "gndvi")
LIGHT_INDEX_NAMES = ("ndvi", "evi", "ndmi", "nbr")
STAT_NAMES = ("median", "std", "min", "max", "amplitude", "valid_ratio")
EPS = 1e-6


def _normalise_feature_set(feature_set: str) -> str:
    value = str(feature_set or "summary").strip().lower()
    aliases = {
        "c04": "summary",
        "summary": "summary",
        "default": "summary",
        "c05": "phenology",
        "phenology": "phenology",
        "phenology_aux": "phenology",
        "c11": "phenology_light",
        "phenology_light": "phenology_light",
        "stage_phenology": "phenology_light",
        "stage_phenology_light": "phenology_light",
    }
    if value not in aliases:
        raise ValueError(f"unknown aux feature set {feature_set!r}; expected one of {sorted(set(aliases))}")
    return aliases[value]


def _summary_feature_names(bands: Sequence[str]) -> list[str]:
    names: list[str] = []
    for band in bands:
        names.extend([f"band_{band.lower()}_{stat}" for stat in STAT_NAMES])
    for index_name in INDEX_NAMES:
        names.extend([f"{index_name}_{stat}" for stat in STAT_NAMES])
    names.extend(["query_doy_scaled", "nearest_time_delta_scaled"])
    names.extend([f"nearest_band_{band.lower()}_median" for band in bands])
    names.extend([f"nearest_{index_name}" for index_name in INDEX_NAMES])
    return names


def _phenology_feature_names() -> list[str]:
    names = [
        "query_minus_nearest_image_doy_scaled",
        "query_minus_first_image_doy_scaled",
        "query_minus_last_image_doy_scaled",
        "query_minus_estimated_ndvi_peak_doy_scaled",
        "query_minus_estimated_ndvi_greenup_doy_scaled",
        "query_minus_estimated_ndvi_senescence_doy_scaled",
        "relative_position_in_ndvi_season",
        "ndvi_query_minus_min_doy_scaled",
        "ndvi_gap_to_max",
        "ndvi_gap_to_min",
    ]
    for index_name in INDEX_NAMES:
        names.extend(
            [
                f"{index_name}_interp_at_query",
                f"{index_name}_nearest_to_query",
                f"{index_name}_max_value",
                f"{index_name}_min_value",
                f"{index_name}_amplitude",
                f"{index_name}_max_doy_scaled",
                f"{index_name}_query_minus_max_doy_scaled",
                f"{index_name}_slope_before_query_30d",
                f"{index_name}_slope_after_query_30d",
            ]
        )
    return names


def _phenology_light_feature_names() -> list[str]:
    names = [
        "query_minus_nearest_image_doy_scaled",
        "query_minus_first_image_doy_scaled",
        "query_minus_last_image_doy_scaled",
        "query_minus_estimated_ndvi_peak_doy_scaled",
        "query_minus_estimated_ndvi_greenup_doy_scaled",
        "query_minus_estimated_ndvi_senescence_doy_scaled",
        "relative_position_in_ndvi_season",
    ]
    for index_name in LIGHT_INDEX_NAMES:
        names.extend(
            [
                f"{index_name}_interp_at_query",
                f"{index_name}_nearest_to_query",
                f"{index_name}_query_minus_max_doy_scaled",
                f"{index_name}_slope_before_query_30d",
                f"{index_name}_slope_after_query_30d",
            ]
        )
    return names


def aux_feature_names(bands: Sequence[str] = DEFAULT_BANDS, feature_set: str = "summary") -> list[str]:
    normalized = _normalise_feature_set(feature_set)
    if normalized == "phenology_light":
        return _phenology_light_feature_names()
    names = _summary_feature_names(bands)
    if normalized == "phenology":
        names.extend(_phenology_feature_names())
    return names


def aux_feature_dim(bands: Sequence[str] = DEFAULT_BANDS, feature_set: str = "summary") -> int:
    return len(aux_feature_names(bands, feature_set=feature_set))


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


def _day_delta_scaled(query_doy: float, event_doy: float) -> float:
    if not np.isfinite(event_doy):
        return 0.0
    return float(np.clip((query_doy - event_doy) / 366.0, -2.0, 2.0))


def _valid_curve(series: np.ndarray, time_doy: np.ndarray, time_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    doys = time_doy.astype(np.float32, copy=False)
    values = series.astype(np.float32, copy=False)
    valid = time_mask.astype(bool) & np.isfinite(doys) & (doys > 0) & np.isfinite(values)
    if not valid.any():
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    valid_doys = doys[valid]
    valid_values = values[valid]
    order = np.argsort(valid_doys)
    return valid_doys[order], valid_values[order]


def _interp_at_query(doys: np.ndarray, values: np.ndarray, query_doy: float) -> float:
    if doys.size == 0:
        return 0.0
    if doys.size == 1:
        return float(values[0])
    return float(np.interp(query_doy, doys, values))


def _nearest_curve_value(doys: np.ndarray, values: np.ndarray, query_doy: float) -> tuple[float, float]:
    if doys.size == 0:
        return 0.0, query_doy
    nearest_index = int(np.argmin(np.abs(doys - query_doy)))
    return float(values[nearest_index]), float(doys[nearest_index])


def _slope_30d(doys: np.ndarray, values: np.ndarray, query_doy: float, side: str) -> float:
    if doys.size < 2:
        return 0.0
    if side == "before":
        candidates = np.where(doys <= query_doy)[0]
        if candidates.size >= 2:
            first, second = int(candidates[-2]), int(candidates[-1])
        else:
            first, second = 0, 1
    elif side == "after":
        candidates = np.where(doys >= query_doy)[0]
        if candidates.size >= 2:
            first, second = int(candidates[0]), int(candidates[1])
        else:
            first, second = doys.size - 2, doys.size - 1
    else:
        raise ValueError("side must be before or after")
    delta_days = float(doys[second] - doys[first])
    if abs(delta_days) <= EPS:
        return 0.0
    return float(np.clip(((float(values[second]) - float(values[first])) / delta_days) * 30.0, -5.0, 5.0))


def _ndvi_event_doys(doys: np.ndarray, ndvi: np.ndarray, query_doy: float) -> tuple[float, float, float, float]:
    if doys.size == 0:
        return query_doy, query_doy, query_doy, query_doy
    peak_index = int(np.argmax(ndvi))
    min_index = int(np.argmin(ndvi))
    min_value = float(np.min(ndvi))
    max_value = float(np.max(ndvi))
    amplitude = max_value - min_value
    if amplitude <= EPS:
        return float(doys[peak_index]), float(doys[0]), float(doys[-1]), float(doys[min_index])

    greenup_threshold = min_value + 0.20 * amplitude
    senescence_threshold = min_value + 0.50 * amplitude

    before_peak = np.arange(0, peak_index + 1)
    green_candidates = before_peak[ndvi[before_peak] >= greenup_threshold]
    greenup_doy = float(doys[int(green_candidates[0])]) if green_candidates.size else float(doys[0])

    after_peak = np.arange(peak_index, doys.size)
    sen_candidates = after_peak[ndvi[after_peak] <= senescence_threshold]
    senescence_doy = float(doys[int(sen_candidates[0])]) if sen_candidates.size else float(doys[-1])
    if senescence_doy <= greenup_doy:
        senescence_doy = float(doys[-1])

    return float(doys[peak_index]), greenup_doy, senescence_doy, float(doys[min_index])


def _phenology_features(indices: dict[str, np.ndarray], time_doy: np.ndarray, time_mask: np.ndarray, query_doy: float) -> list[float]:
    valid_times = np.where(time_mask.astype(bool) & np.isfinite(time_doy.astype(np.float32)) & (time_doy.astype(np.float32) > 0))[0]
    if valid_times.size:
        valid_doys = time_doy[valid_times].astype(np.float32)
        nearest_image_doy = float(valid_doys[int(np.argmin(np.abs(valid_doys - query_doy)))])
        first_image_doy = float(np.min(valid_doys))
        last_image_doy = float(np.max(valid_doys))
    else:
        nearest_image_doy = first_image_doy = last_image_doy = query_doy

    ndvi_doys, ndvi_values = _valid_curve(indices["ndvi"], time_doy, time_mask)
    ndvi_peak_doy, greenup_doy, senescence_doy, ndvi_min_doy = _ndvi_event_doys(ndvi_doys, ndvi_values, query_doy)
    season_span = max(senescence_doy - greenup_doy, 1.0)
    relative_position = float(np.clip((query_doy - greenup_doy) / season_span, -1.0, 2.0))
    ndvi_interp = _interp_at_query(ndvi_doys, ndvi_values, query_doy)
    ndvi_max = float(np.max(ndvi_values)) if ndvi_values.size else 0.0
    ndvi_min = float(np.min(ndvi_values)) if ndvi_values.size else 0.0

    features: list[float] = [
        _day_delta_scaled(query_doy, nearest_image_doy),
        _day_delta_scaled(query_doy, first_image_doy),
        _day_delta_scaled(query_doy, last_image_doy),
        _day_delta_scaled(query_doy, ndvi_peak_doy),
        _day_delta_scaled(query_doy, greenup_doy),
        _day_delta_scaled(query_doy, senescence_doy),
        relative_position,
        _day_delta_scaled(query_doy, ndvi_min_doy),
        float(np.clip(ndvi_max - ndvi_interp, -5.0, 5.0)),
        float(np.clip(ndvi_interp - ndvi_min, -5.0, 5.0)),
    ]

    for index_name in INDEX_NAMES:
        doys, values = _valid_curve(indices[index_name], time_doy, time_mask)
        interp = _interp_at_query(doys, values, query_doy)
        nearest, _nearest_doy = _nearest_curve_value(doys, values, query_doy)
        if values.size:
            max_index = int(np.argmax(values))
            max_value = float(values[max_index])
            min_value = float(np.min(values))
            max_doy = float(doys[max_index])
        else:
            max_value = min_value = max_doy = 0.0
        features.extend(
            [
                float(np.clip(interp, -5.0, 5.0)),
                float(np.clip(nearest, -5.0, 5.0)),
                float(np.clip(max_value, -5.0, 5.0)),
                float(np.clip(min_value, -5.0, 5.0)),
                float(np.clip(max_value - min_value, 0.0, 5.0)),
                float(np.clip(max_doy / 366.0, 0.0, 1.0)) if max_doy > 0 else 0.0,
                _day_delta_scaled(query_doy, max_doy),
                _slope_30d(doys, values, query_doy, "before"),
                _slope_30d(doys, values, query_doy, "after"),
            ]
        )
    return features


def _phenology_light_features(indices: dict[str, np.ndarray], time_doy: np.ndarray, time_mask: np.ndarray, query_doy: float) -> list[float]:
    valid_times = np.where(time_mask.astype(bool) & np.isfinite(time_doy.astype(np.float32)) & (time_doy.astype(np.float32) > 0))[0]
    if valid_times.size:
        valid_doys = time_doy[valid_times].astype(np.float32)
        nearest_image_doy = float(valid_doys[int(np.argmin(np.abs(valid_doys - query_doy)))])
        first_image_doy = float(np.min(valid_doys))
        last_image_doy = float(np.max(valid_doys))
    else:
        nearest_image_doy = first_image_doy = last_image_doy = query_doy

    ndvi_doys, ndvi_values = _valid_curve(indices["ndvi"], time_doy, time_mask)
    ndvi_peak_doy, greenup_doy, senescence_doy, _ndvi_min_doy = _ndvi_event_doys(ndvi_doys, ndvi_values, query_doy)
    season_span = max(senescence_doy - greenup_doy, 1.0)
    features: list[float] = [
        _day_delta_scaled(query_doy, nearest_image_doy),
        _day_delta_scaled(query_doy, first_image_doy),
        _day_delta_scaled(query_doy, last_image_doy),
        _day_delta_scaled(query_doy, ndvi_peak_doy),
        _day_delta_scaled(query_doy, greenup_doy),
        _day_delta_scaled(query_doy, senescence_doy),
        float(np.clip((query_doy - greenup_doy) / season_span, -1.0, 2.0)),
    ]

    for index_name in LIGHT_INDEX_NAMES:
        doys, values = _valid_curve(indices[index_name], time_doy, time_mask)
        interp = _interp_at_query(doys, values, query_doy)
        nearest, _nearest_doy = _nearest_curve_value(doys, values, query_doy)
        max_doy = float(doys[int(np.argmax(values))]) if values.size else query_doy
        features.extend(
            [
                float(np.clip(interp, -5.0, 5.0)),
                float(np.clip(nearest, -5.0, 5.0)),
                _day_delta_scaled(query_doy, max_doy),
                _slope_30d(doys, values, query_doy, "before"),
                _slope_30d(doys, values, query_doy, "after"),
            ]
        )
    return features


def compute_aux_features(
    patches: np.ndarray,
    valid_pixel_mask: np.ndarray,
    time_mask: np.ndarray,
    time_doy: np.ndarray,
    query_doy: float,
    bands: Sequence[str] = DEFAULT_BANDS,
    feature_set: str = "summary",
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

    normalized_feature_set = _normalise_feature_set(feature_set)
    if normalized_feature_set == "phenology_light":
        features = _phenology_light_features(indices, time_doy.astype(np.float32, copy=False), time_mask_bool, query_value)
        out = np.asarray(features, dtype=np.float32)
        expected = aux_feature_dim(bands, feature_set=feature_set)
        if out.shape[0] != expected:
            raise RuntimeError(f"aux feature dimension mismatch: got {out.shape[0]}, expected {expected}")
        return out

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
    if normalized_feature_set == "phenology":
        features.extend(_phenology_features(indices, time_doy.astype(np.float32, copy=False), time_mask_bool, query_value))

    out = np.asarray(features, dtype=np.float32)
    expected = aux_feature_dim(bands, feature_set=feature_set)
    if out.shape[0] != expected:
        raise RuntimeError(f"aux feature dimension mismatch: got {out.shape[0]}, expected {expected}")
    return out
