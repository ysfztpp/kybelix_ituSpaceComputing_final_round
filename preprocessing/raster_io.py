from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import (
    INVALID_FILL_VALUE,
    VALID_REFLECTANCE_MAX_INCLUSIVE,
    VALID_REFLECTANCE_MIN_EXCLUSIVE,
)

try:
    import rasterio
    from rasterio.windows import Window
except ImportError as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "rasterio is required for this project. PIL reads these float GeoTIFFs incorrectly. "
        "Install with `python3 -m pip install rasterio`."
    ) from exc


@dataclass(frozen=True)
class RasterMeta:
    path: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    width: int
    height: int
    pixel_size_x: float
    pixel_size_y: float
    crs: str
    transform: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class PatchExtraction:
    patch: np.ndarray
    pixel_x: int
    pixel_y: int
    pixel_x_float: float
    pixel_y_float: float
    border_margin_pixels: int
    center_clamped: bool


def raster_meta_from_src(path: Path, src) -> RasterMeta:
    transform = src.transform
    bounds = src.bounds
    return RasterMeta(
        path=str(path),
        xmin=float(bounds.left),
        ymin=float(bounds.bottom),
        xmax=float(bounds.right),
        ymax=float(bounds.top),
        width=int(src.width),
        height=int(src.height),
        pixel_size_x=float(abs(transform.a)),
        pixel_size_y=float(abs(transform.e)),
        crs=str(src.crs) if src.crs else "",
        transform=(transform.a, transform.b, transform.c, transform.d, transform.e, transform.f),
    )


def read_raster_meta(path: Path) -> RasterMeta:
    with rasterio.open(path) as src:
        return raster_meta_from_src(path, src)


def contains_point(meta: RasterMeta, lon: float, lat: float) -> bool:
    return meta.xmin <= lon < meta.xmax and meta.ymin <= lat < meta.ymax


def bbox_distance_deg(meta: RasterMeta, lon: float, lat: float) -> float:
    dx = 0.0
    if lon < meta.xmin:
        dx = meta.xmin - lon
    elif lon > meta.xmax:
        dx = lon - meta.xmax

    dy = 0.0
    if lat < meta.ymin:
        dy = meta.ymin - lat
    elif lat > meta.ymax:
        dy = lat - meta.ymax
    return math.hypot(dx, dy)


def lonlat_to_pixel(meta: RasterMeta, lon: float, lat: float) -> tuple[float, float, int, int]:
    pixel_x_float = (lon - meta.xmin) / meta.pixel_size_x
    pixel_y_float = (meta.ymax - lat) / meta.pixel_size_y
    return pixel_x_float, pixel_y_float, int(round(pixel_x_float)), int(round(pixel_y_float))


def patch_fits_without_padding(meta: RasterMeta, pixel_x: int, pixel_y: int, patch_size: int) -> bool:
    half = patch_size // 2
    return half <= pixel_x < (meta.width - half) and half <= pixel_y < (meta.height - half)


def border_margin_pixels(meta: RasterMeta, pixel_x: int, pixel_y: int) -> int:
    return min(pixel_x, pixel_y, meta.width - 1 - pixel_x, meta.height - 1 - pixel_y)


def _extract_patch_from_open_src(src, meta: RasterMeta, lon: float, lat: float, patch_size: int) -> PatchExtraction:
    half = patch_size // 2
    pixel_x_float, pixel_y_float, pixel_x, pixel_y = lonlat_to_pixel(meta, lon, lat)
    center_clamped = False

    if pixel_x < 0:
        pixel_x = 0
        center_clamped = True
    elif pixel_x >= src.width:
        pixel_x = src.width - 1
        center_clamped = True

    if pixel_y < 0:
        pixel_y = 0
        center_clamped = True
    elif pixel_y >= src.height:
        pixel_y = src.height - 1
        center_clamped = True

    x0 = pixel_x - half
    x1 = pixel_x + half + 1
    y0 = pixel_y - half
    y1 = pixel_y + half + 1

    crop_left = max(0, x0)
    crop_top = max(0, y0)
    crop_right = min(src.width, x1)
    crop_bottom = min(src.height, y1)
    if crop_right <= crop_left or crop_bottom <= crop_top:
        raise ValueError("empty patch after coordinate clamping")

    window = Window(crop_left, crop_top, crop_right - crop_left, crop_bottom - crop_top)
    patch = src.read(1, window=window, out_dtype="float32", masked=False)
    patch = np.asarray(patch, dtype=np.float32)

    pad_left = crop_left - x0
    pad_top = crop_top - y0
    pad_right = x1 - crop_right
    pad_bottom = y1 - crop_bottom
    if any(value > 0 for value in (pad_left, pad_top, pad_right, pad_bottom)):
        patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="edge")

    if patch.shape != (patch_size, patch_size):
        raise ValueError(f"unexpected patch shape {patch.shape}, expected {(patch_size, patch_size)}")

    return PatchExtraction(
        patch=patch.astype(np.float32, copy=False),
        pixel_x=int(pixel_x),
        pixel_y=int(pixel_y),
        pixel_x_float=float(pixel_x_float),
        pixel_y_float=float(pixel_y_float),
        border_margin_pixels=int(border_margin_pixels(meta, pixel_x, pixel_y)),
        center_clamped=bool(center_clamped),
    )


def extract_patch_edge(path: Path, lon: float, lat: float, patch_size: int) -> PatchExtraction:
    with rasterio.open(path) as src:
        meta = raster_meta_from_src(path, src)
        return _extract_patch_from_open_src(src, meta, lon, lat, patch_size)


def extract_patch_edge_from_src(src, meta: RasterMeta, lon: float, lat: float, patch_size: int) -> PatchExtraction:
    return _extract_patch_from_open_src(src, meta, lon, lat, patch_size)


def clean_patch_values(
    patch: np.ndarray,
    min_exclusive: float = VALID_REFLECTANCE_MIN_EXCLUSIVE,
    max_inclusive: float = VALID_REFLECTANCE_MAX_INCLUSIVE,
    fill_value: float = INVALID_FILL_VALUE,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(patch) & (patch > min_exclusive) & (patch <= max_inclusive)
    cleaned = np.where(valid, patch, fill_value).astype(np.float32, copy=False)
    return cleaned, valid.astype(bool, copy=False)
