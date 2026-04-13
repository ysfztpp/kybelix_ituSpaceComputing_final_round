from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "#fbfaf7",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "axes.titleweight": "bold",
    "grid.color": "#d8d4ca",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "font.family": "DejaVu Sans",
}


def _save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def _histogram(path: Path, values: pd.Series, title: str, xlabel: str, bins: int = 20) -> None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    with plt.rc_context(PLOT_STYLE):
        plt.figure(figsize=(7.8, 4.4))
        if values.empty:
            plt.text(0.5, 0.5, "No data", ha="center", va="center")
        else:
            plt.hist(values, bins=min(bins, max(4, values.nunique())), color="#34656d", edgecolor="white")
            plt.axvline(values.median(), color="#b44b35", linewidth=2, label=f"median={values.median():.3g}")
            plt.legend(frameon=False)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Sample count")
        plt.grid(axis="y")
    _save_fig(path)


def _bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str, log_y: bool = False) -> None:
    with plt.rc_context(PLOT_STYLE):
        width = max(7.0, 0.65 * max(1, len(labels)))
        plt.figure(figsize=(width, 4.8))
        x = np.arange(len(labels))
        plt.bar(x, values, color="#7b8f5c", edgecolor="white")
        plt.xticks(x, labels, rotation=35, ha="right")
        plt.title(title)
        plt.ylabel(ylabel)
        if log_y:
            plt.yscale("log")
        for i, value in enumerate(values):
            plt.text(i, max(value, 1) * (1.03 if not log_y else 1.08), f"{value:g}", ha="center", va="bottom", fontsize=8)
        plt.grid(axis="y")
    _save_fig(path)


def _sample_matrix(group: pd.DataFrame, value_col: str) -> np.ndarray:
    size = int(max(group["patch_row"].max(), group["patch_col"].max()) + 1)
    arr = np.full((size, size), np.nan, dtype=float)
    for row in group.itertuples(index=False):
        arr[int(row.patch_row), int(row.patch_col)] = float(getattr(row, value_col))
    return arr


def _sample_panel(path: Path, group: pd.DataFrame, title: str) -> None:
    raw = _sample_matrix(group, "raw_value")
    cleaned = _sample_matrix(group, "cleaned_value")
    mask = _sample_matrix(group.assign(valid=group["valid"].astype(float)), "valid")
    finite_values = raw[np.isfinite(raw)]
    if finite_values.size:
        vmin = float(np.nanpercentile(finite_values, 2))
        vmax = float(np.nanpercentile(finite_values, 98))
    else:
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
        fig.suptitle(title, fontsize=11, fontweight="bold")
        panels = [(raw, "Raw read", "viridis", vmin, vmax), (cleaned, "Cleaned tensor", "viridis", vmin, vmax), (mask, "Valid mask", "gray", 0, 1)]
        for ax, (arr, name, cmap, lo, hi) in zip(axes, panels):
            im = ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_title(name, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()


def _value_counts(metadata: pd.DataFrame, column: str) -> tuple[list[str], list[float]]:
    if column not in metadata.columns or metadata.empty:
        return [], []
    counts = metadata[column].value_counts(dropna=False).sort_index()
    return [str(label) for label in counts.index.tolist()], [float(value) for value in counts.values.tolist()]


def write_preprocessing_report(report_dir: Path, report: dict[str, Any], metadata: pd.DataFrame, sample_rows: pd.DataFrame | None = None) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = report_dir / "charts"
    samples_dir = report_dir / "samples"
    charts_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    (report_dir / "summary.json").write_text(json.dumps(report, indent=2, default=str))

    chart_files: list[str] = []
    if not metadata.empty:
        _histogram(charts_dir / "valid_pixel_ratio_hist.png", metadata["valid_pixel_ratio"], "Valid Pixel Ratio by Sample", "valid pixels / observed pixels")
        _histogram(charts_dir / "missing_band_cells_hist.png", metadata["missing_band_cells"], "Missing Band Cells by Sample", "missing date-band cells")
        _histogram(charts_dir / "time_steps_hist.png", metadata["time_steps_kept"], "Timesteps Kept by Sample", "time steps")
        chart_files.extend(["charts/valid_pixel_ratio_hist.png", "charts/missing_band_cells_hist.png", "charts/time_steps_hist.png"])

        labels, values = _value_counts(metadata, "crop_type")
        if labels:
            _bar(charts_dir / "crop_type_counts.png", labels, values, "Crop Type Counts", "training samples")
            chart_files.append("charts/crop_type_counts.png")

        labels, values = _value_counts(metadata, "resolved_region_id")
        if labels:
            _bar(charts_dir / "samples_by_region.png", labels, values, "Samples by Resolved Region", "samples")
            chart_files.append("charts/samples_by_region.png")

    quality_labels = ["valid_pixels", "invalid_pixels", "missing_band_cells", "edge_samples", "center_clamped_samples"]
    quality_values = [
        float(report.get("valid_patch_pixels", 0)),
        float(report.get("invalid_patch_pixels", 0)),
        float(report.get("missing_band_cells", 0)),
        float(report.get("requires_edge_replication_count", 0)),
        float(report.get("samples_with_center_clamping", 0)),
    ]
    _bar(charts_dir / "quality_summary_log_counts.png", quality_labels, quality_values, "Preprocessing Quality Summary", "count (log scale)", log_y=True)
    chart_files.append("charts/quality_summary_log_counts.png")

    band_missing = report.get("missing_band_cells_by_band", {})
    if band_missing:
        labels = list(band_missing.keys())
        values = [float(band_missing[key]) for key in labels]
        _bar(charts_dir / "missing_band_cells_by_band.png", labels, values, "Missing Date-Band Cells by Band", "missing cells")
        chart_files.append("charts/missing_band_cells_by_band.png")

    band_valid = report.get("valid_pixel_ratio_by_band", {})
    if band_valid:
        labels = list(band_valid.keys())
        values = [float(band_valid[key]) for key in labels]
        _bar(charts_dir / "valid_pixel_ratio_by_band.png", labels, values, "Valid Pixel Ratio by Band", "valid ratio")
        chart_files.append("charts/valid_pixel_ratio_by_band.png")

    sample_files: list[str] = []
    if sample_rows is not None and not sample_rows.empty:
        sample_rows.to_csv(samples_dir / "patch_before_after_samples.csv", index=False)
        sample_files.append("samples/patch_before_after_samples.csv")
        for sample_id, group in sample_rows.groupby("sample_patch_id", sort=True):
            safe_label = str(sample_id).replace("/", "_").replace(" ", "_")
            sample_path = samples_dir / f"{safe_label}_before_after_mask.png"
            row0 = group.iloc[0]
            title = f"{safe_label}: point={row0['point_id']} date={row0['date']} band={row0['band_id']} group={row0.get('sample_group', '')}"
            _sample_panel(sample_path, group, title)
            sample_files.append(f"samples/{sample_path.name}")

    (report_dir / "chart_manifest.json").write_text(json.dumps({"charts": chart_files, "samples": sample_files}, indent=2))

    steps = [
        "Read config and resolve raw data paths.",
        "Read point labels and reduce repeated label rows to unique points for patch extraction.",
        "Scan Sentinel-2 TIFF folders.",
        "Parse TIFF filenames into region, date, processing level, band, and path.",
        "Drop unsupported bands outside the configured 12-band tensor.",
        "Resolve duplicate canonical TIFF names deterministically.",
        "Build one selected file index per region/date/band, preferring L2A over L1C.",
        "Read raster metadata with rasterio and build region bounding boxes.",
        "Map each point to its covering region; if multiple regions cover it, choose the best deterministic region.",
        "Convert longitude/latitude to pixel coordinates.",
        "Extract a true 15x15 patch for each sample, timestep, and configured band.",
        "Use edge replication if the patch crosses a raster border.",
        "Detect invalid pixels using configured finite/min/max reflectance thresholds.",
        "Keep valid raw values unchanged; fill invalid tensor positions only and preserve the mask.",
        "Write masks: valid_pixel_mask, band_mask, time_mask, time_doy, source index, and metadata arrays.",
        "Write the final NPZ dataset for CNN + Transformer training.",
        "Compute train-only normalization statistics separately.",
        "Write human-readable logs, article-friendly PNG charts, and deterministic before/after sample patch panels.",
    ]
    (report_dir / "steps_one_by_one.md").write_text("\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1)) + "\n")

    log_lines = [
        "# Preprocessing Log",
        "",
        "## Final Dataset",
        f"- Output NPZ: `{report.get('output_npz', '')}`",
        f"- Patch shape: `{report.get('patch_shape', '')}`",
        f"- Bands: `{', '.join(report.get('bands', []))}`",
        f"- Invalid value policy: `{report.get('invalid_value_policy', '')}`",
        "",
        "## Key Counts",
        f"- Input label rows: `{report.get('points_input_rows', 0)}`",
        f"- Unique points: `{report.get('unique_points', 0)}`",
        f"- Samples kept: `{report.get('samples_kept', 0)}`",
        f"- Samples dropped: `{report.get('samples_dropped', 0)}`",
        f"- Missing band cells: `{report.get('missing_band_cells', 0)}`",
        f"- Samples with missing bands: `{report.get('samples_with_missing_bands', 0)}`",
        f"- Valid pixel ratio: `{report.get('global_valid_pixel_ratio', 0)}`",
        f"- Edge replication samples: `{report.get('requires_edge_replication_count', 0)}`",
        f"- Center-clamped samples: `{report.get('samples_with_center_clamping', 0)}`",
        "",
        "## Generated Charts",
        *[f"- `{path}`" for path in chart_files],
        "",
        "## Before/After Samples",
        *[f"- `{path}`" for path in sample_files],
        "",
        "## Step-by-Step File",
        "- `steps_one_by_one.md`",
    ]
    (report_dir / "preprocessing_log.md").write_text("\n".join(log_lines) + "\n")
