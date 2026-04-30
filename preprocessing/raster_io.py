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


@dataclass(frozen=True)
class PatchWindow:
    pixel_x: int
    pixel_y: int
    pixel_x_float: float
    pixel_y_float: float
    crop_left: int
    crop_top: int
    crop_right: int
    crop_bottom: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
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


def _patch_window(src, meta: RasterMeta, lon: float, lat: float, patch_size: int) -> PatchWindow:
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

    pad_left = crop_left - x0
    pad_top = crop_top - y0
    pad_right = x1 - crop_right
    pad_bottom = y1 - crop_bottom

    return PatchWindow(
        pixel_x=int(pixel_x),
        pixel_y=int(pixel_y),
        pixel_x_float=float(pixel_x_float),
        pixel_y_float=float(pixel_y_float),
        crop_left=int(crop_left),
        crop_top=int(crop_top),
        crop_right=int(crop_right),
        crop_bottom=int(crop_bottom),
        pad_left=int(pad_left),
        pad_top=int(pad_top),
        pad_right=int(pad_right),
        pad_bottom=int(pad_bottom),
        border_margin_pixels=int(border_margin_pixels(meta, pixel_x, pixel_y)),
        center_clamped=bool(center_clamped),
    )


def _patch_extraction_from_array(patch: np.ndarray, spec: PatchWindow, patch_size: int) -> PatchExtraction:
    if any(value > 0 for value in (spec.pad_left, spec.pad_top, spec.pad_right, spec.pad_bottom)):
        patch = np.pad(patch, ((spec.pad_top, spec.pad_bottom), (spec.pad_left, spec.pad_right)), mode="edge")

    if patch.shape != (patch_size, patch_size):
        raise ValueError(f"unexpected patch shape {patch.shape}, expected {(patch_size, patch_size)}")

    return PatchExtraction(
        patch=patch.astype(np.float32, copy=False),
        pixel_x=spec.pixel_x,
        pixel_y=spec.pixel_y,
        pixel_x_float=spec.pixel_x_float,
        pixel_y_float=spec.pixel_y_float,
        border_margin_pixels=spec.border_margin_pixels,
        center_clamped=spec.center_clamped,
    )


def _extract_patch_from_open_src(src, meta: RasterMeta, lon: float, lat: float, patch_size: int) -> PatchExtraction:
    spec = _patch_window(src, meta, lon, lat, patch_size)
    window = Window(spec.crop_left, spec.crop_top, spec.crop_right - spec.crop_left, spec.crop_bottom - spec.crop_top)
    patch = src.read(1, window=window, out_dtype="float32", masked=False)
    patch = np.asarray(patch, dtype=np.float32)
    return _patch_extraction_from_array(patch, spec, patch_size)


def extract_patch_edge(path: Path, lon: float, lat: float, patch_size: int) -> PatchExtraction:
    with rasterio.open(path) as src:
        meta = raster_meta_from_src(path, src)
        return _extract_patch_from_open_src(src, meta, lon, lat, patch_size)


def extract_patch_edge_from_src(src, meta: RasterMeta, lon: float, lat: float, patch_size: int) -> PatchExtraction:
    return _extract_patch_from_open_src(src, meta, lon, lat, patch_size)


def extract_patches_edge_batched_from_src(
    src,
    meta: RasterMeta,
    points: list[tuple[float, float]],
    patch_size: int,
    max_union_pixels: int = 262144,
    max_overread_ratio: float = 6.0,
) -> tuple[list[PatchExtraction], bool, int]:
    specs = [_patch_window(src, meta, lon, lat, patch_size) for lon, lat in points]
    if not specs:
        return [], False, 0

    union_left = min(spec.crop_left for spec in specs)
    union_top = min(spec.crop_top for spec in specs)
    union_right = max(spec.crop_right for spec in specs)
    union_bottom = max(spec.crop_bottom for spec in specs)
    union_pixels = (union_right - union_left) * (union_bottom - union_top)
    exact_pixels = sum((spec.crop_right - spec.crop_left) * (spec.crop_bottom - spec.crop_top) for spec in specs)

    if union_pixels > max_union_pixels or union_pixels > int(max_overread_ratio * max(exact_pixels, 1)):
        out: list[PatchExtraction] = []
        pixels_read = 0
        for spec in specs:
            window = Window(spec.crop_left, spec.crop_top, spec.crop_right - spec.crop_left, spec.crop_bottom - spec.crop_top)
            patch = src.read(1, window=window, out_dtype="float32", masked=False)
            patch = np.asarray(patch, dtype=np.float32)
            pixels_read += int(patch.size)
            out.append(_patch_extraction_from_array(patch, spec, patch_size))
        return out, False, pixels_read

    window = Window(union_left, union_top, union_right - union_left, union_bottom - union_top)
    union = src.read(1, window=window, out_dtype="float32", masked=False)
    union = np.asarray(union, dtype=np.float32)
    out = []
    for spec in specs:
        row_start = spec.crop_top - union_top
        row_end = spec.crop_bottom - union_top
        col_start = spec.crop_left - union_left
        col_end = spec.crop_right - union_left
        out.append(_patch_extraction_from_array(union[row_start:row_end, col_start:col_end], spec, patch_size))
    return out, True, int(union.size)


def _block_shape(src) -> tuple[int, int]:
    try:
        block_height, block_width = src.block_shapes[0]
    except Exception:  # pragma: no cover
        block_height, block_width = src.height, src.width
    return max(1, int(block_height)), max(1, int(block_width))


def _block_window(src, block_row: int, block_col: int, block_height: int, block_width: int) -> tuple[Window, int, int, int, int]:
    row_start = block_row * block_height
    col_start = block_col * block_width
    row_end = min(src.height, row_start + block_height)
    col_end = min(src.width, col_start + block_width)
    return Window(col_start, row_start, col_end - col_start, row_end - row_start), row_start, col_start, row_end, col_end


def extract_patches_edge_block_cached_from_src(
    src,
    meta: RasterMeta,
    points: list[tuple[float, float]],
    patch_size: int,
    max_block_pixels: int = 1048576,
    max_overread_ratio: float = 12.0,
    fallback_union_pixels: int = 262144,
    fallback_union_overread_ratio: float = 6.0,
) -> tuple[list[PatchExtraction], str, int, int]:
    specs = [_patch_window(src, meta, lon, lat, patch_size) for lon, lat in points]
    if not specs:
        return [], "block", 0, 0

    block_height, block_width = _block_shape(src)
    exact_pixels = sum((spec.crop_right - spec.crop_left) * (spec.crop_bottom - spec.crop_top) for spec in specs)
    needed_blocks: set[tuple[int, int]] = set()
    blocks_by_spec: list[list[tuple[int, int]]] = []
    for spec in specs:
        spec_blocks = [
            (block_row, block_col)
            for block_row in range(spec.crop_top // block_height, ((spec.crop_bottom - 1) // block_height) + 1)
            for block_col in range(spec.crop_left // block_width, ((spec.crop_right - 1) // block_width) + 1)
        ]
        blocks_by_spec.append(spec_blocks)
        needed_blocks.update(spec_blocks)

    block_pixels = 0
    for block_row, block_col in needed_blocks:
        _window, row_start, col_start, row_end, col_end = _block_window(src, block_row, block_col, block_height, block_width)
        block_pixels += (row_end - row_start) * (col_end - col_start)
    if block_pixels > max_block_pixels or block_pixels > int(max_overread_ratio * max(exact_pixels, 1)):
        patches, used_batch, pixels_read = extract_patches_edge_batched_from_src(
            src, meta, points, patch_size, fallback_union_pixels, fallback_union_overread_ratio
        )
        return patches, "batch" if used_batch else "patch", 1 if used_batch else len(patches), int(pixels_read)

    block_cache: dict[tuple[int, int], tuple[np.ndarray, int, int]] = {}
    pixels_read = 0
    for block_row, block_col in sorted(needed_blocks):
        window, row_start, col_start, _row_end, _col_end = _block_window(src, block_row, block_col, block_height, block_width)
        block = src.read(1, window=window, out_dtype="float32", masked=False)
        block = np.asarray(block, dtype=np.float32)
        block_cache[(block_row, block_col)] = (block, row_start, col_start)
        pixels_read += int(block.size)

    out: list[PatchExtraction] = []
    for spec, spec_blocks in zip(specs, blocks_by_spec):
        crop = np.empty((spec.crop_bottom - spec.crop_top, spec.crop_right - spec.crop_left), dtype=np.float32)
        for block_row, block_col in spec_blocks:
            block, block_row_start, block_col_start = block_cache[(block_row, block_col)]
            row_start = max(spec.crop_top, block_row_start)
            col_start = max(spec.crop_left, block_col_start)
            row_end = min(spec.crop_bottom, block_row_start + block.shape[0])
            col_end = min(spec.crop_right, block_col_start + block.shape[1])
            crop[row_start - spec.crop_top : row_end - spec.crop_top, col_start - spec.crop_left : col_end - spec.crop_left] = block[
                row_start - block_row_start : row_end - block_row_start,
                col_start - block_col_start : col_end - block_col_start,
            ]
        out.append(_patch_extraction_from_array(crop, spec, patch_size))

    return out, "block", len(block_cache), int(pixels_read)


def clean_patch_values(
    patch: np.ndarray,
    min_exclusive: float = VALID_REFLECTANCE_MIN_EXCLUSIVE,
    max_inclusive: float = VALID_REFLECTANCE_MAX_INCLUSIVE,
    fill_value: float = INVALID_FILL_VALUE,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(patch) & (patch > min_exclusive) & (patch <= max_inclusive)
    cleaned = np.where(valid, patch, fill_value).astype(np.float32, copy=False)
    return cleaned, valid.astype(bool, copy=False)
