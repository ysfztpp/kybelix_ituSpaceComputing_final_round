from __future__ import annotations

BAND_ORDER = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
PHENOPHASE_ORDER = [
    "Greenup",
    "MidGreenup",
    "Peak",
    "Maturity",
    "MidSenescence",
    "Senescence",
    "Dormancy",
]
PHENOPHASE_CHRONOLOGICAL_ORDER = [
    "Greenup",
    "MidGreenup",
    "Maturity",
    "Peak",
    "Senescence",
    "MidSenescence",
    "Dormancy",
]
PHENOPHASE_DOY_RANK_TO_STAGE_ID = [PHENOPHASE_ORDER.index(name) for name in PHENOPHASE_CHRONOLOGICAL_ORDER]
PHENOPHASE_STAGE_ID_TO_DOY_RANK = [PHENOPHASE_DOY_RANK_TO_STAGE_ID.index(stage_id) for stage_id in range(len(PHENOPHASE_ORDER))]
CROP_TYPE_ORDER = ["corn", "rice", "soybean"]

PATCH_SIZE = 15
PATCH_HALF = PATCH_SIZE // 2
INVALID_FILL_VALUE = 0.0

# Sentinel-2 reflectance in this competition reads as float reflectance-like values.
# We keep the valid raw values unchanged and only mask/fill values outside this range.
VALID_REFLECTANCE_MIN_EXCLUSIVE = 0.0
VALID_REFLECTANCE_MAX_INCLUSIVE = 2.0
