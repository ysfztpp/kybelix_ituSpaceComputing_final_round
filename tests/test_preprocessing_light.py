from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.aux_features import aux_feature_dim, compute_aux_features
from data.splits import make_train_val_split
from data.transforms import NpzPatchNormalizer
from preprocessing.filename import parse_tiff_name
from preprocessing.raster_io import clean_patch_values, extract_patch_edge, lonlat_to_pixel, rasterio, read_raster_meta
from rasterio.transform import from_origin


def test_filename_parser() -> None:
    parsed = parse_tiff_name("region54-2018-10-03-00_00_2018-10-03-23_59_Sentinel-2_L2A_B8A_(Raw).tiff")
    assert parsed is not None
    assert parsed.region_id == "region54"
    assert parsed.start_norm == "2018-10-03-00-00"
    assert parsed.end_norm == "2018-10-03-23-59"
    assert parsed.band_id == "B8A"


def test_patch_extract_and_masks() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "region00_2018-01-01-00-00_2018-01-01-23-59_Sentinel-2_L2A_B04_(Raw).tiff"
        arr = np.arange(20 * 20, dtype=np.float32).reshape(20, 20) / 1000.0
        arr[0, 0] = 0.0
        arr[1, 1] = 3.0
        transform = from_origin(0.0, 20.0, 1.0, 1.0)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=20,
            width=20,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(arr, 1)

        meta = read_raster_meta(path)
        _, _, px, py = lonlat_to_pixel(meta, lon=0.2, lat=19.8)
        assert (px, py) == (0, 0)
        extracted = extract_patch_edge(path, lon=0.2, lat=19.8, patch_size=15)
        assert extracted.patch.shape == (15, 15)
        assert extracted.center_clamped is False
        cleaned, valid = clean_patch_values(extracted.patch, min_exclusive=0.0, max_inclusive=2.0, fill_value=0.0)
        assert cleaned.shape == (15, 15)
        assert valid.shape == (15, 15)
        assert valid.dtype == bool
        assert not valid[0, 0]
        assert cleaned[0, 0] == 0.0


def test_split_and_normalizer() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        metadata = pd.DataFrame(
            {
                "sample_index": [0, 1, 2, 3, 4, 5],
                "point_id": [10, 11, 12, 13, 14, 15],
                "resolved_region_id": ["r1", "r1", "r2", "r2", "r3", "r3"],
                "crop_type": ["corn", "corn", "rice", "rice", "soybean", "soybean"],
            }
        )
        metadata_csv = tmp_path / "metadata.csv"
        split_csv = tmp_path / "split.csv"
        metadata.to_csv(metadata_csv, index=False)
        summary = make_train_val_split(metadata_csv, split_csv, val_fraction=0.5, seed=1, stratify_by="crop_type")
        assert summary["total_count"] == 6
        assert split_csv.exists()

        stats_json = tmp_path / "stats.json"
        stats_json.write_text(
            '{"bands":["B1","B2"],"per_band":{"B1":{"mean":1,"std":2,"median":1,"iqr":2},"B2":{"mean":2,"std":4,"median":2,"iqr":4}}}'
        )
        normalizer = NpzPatchNormalizer(stats_json)
        patches = np.ones((3, 2, 5, 5), dtype=np.float32)
        mask = np.ones_like(patches, dtype=bool)
        out = normalizer(patches, mask)
        assert out.shape == patches.shape
        assert np.isclose(out[:, 0].mean(), 0.0)
        assert np.isclose(out[:, 1].mean(), -0.25)


def test_aux_features() -> None:
    bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
    patches = np.full((3, 12, 5, 5), 0.2, dtype=np.float32)
    patches[:, 7] = 0.7
    patches[:, 3] = 0.3
    patches[1, 10] = 0.4
    valid = np.ones_like(patches, dtype=bool)
    valid[2, :, 0, 0] = False
    time_mask = np.asarray([True, True, False])
    time_doy = np.asarray([100, 130, 160], dtype=np.int16)
    features = compute_aux_features(patches, valid, time_mask, time_doy, query_doy=120, bands=bands)
    assert features.shape == (aux_feature_dim(bands),)
    assert np.isfinite(features).all()
    phenology_features = compute_aux_features(patches, valid, time_mask, time_doy, query_doy=120, bands=bands, feature_set="phenology")
    assert phenology_features.shape == (aux_feature_dim(bands, feature_set="phenology"),)
    assert phenology_features.shape[0] > features.shape[0]
    assert np.isfinite(phenology_features).all()
    light_features = compute_aux_features(patches, valid, time_mask, time_doy, query_doy=120, bands=bands, feature_set="phenology_light")
    assert light_features.shape == (aux_feature_dim(bands, feature_set="phenology_light"),)
    assert light_features.shape[0] < features.shape[0]
    assert np.isfinite(light_features).all()


def main() -> None:
    test_filename_parser()
    test_patch_extract_and_masks()
    test_split_and_normalizer()
    test_aux_features()
    print("light preprocessing tests passed")


if __name__ == "__main__":
    main()
