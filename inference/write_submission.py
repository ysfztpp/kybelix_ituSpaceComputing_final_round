from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

CROP_TYPE_NAMES = ["corn", "rice", "soybean"]
PHENOPHASE_NAMES = ["Greenup", "MidGreenup", "Peak", "Maturity", "MidSenescence", "Senescence", "Dormancy"]


def write_generic_result_json(predictions: dict[str, np.ndarray], output_json: Path) -> None:
    """Write a transparent JSON result.

    The original submission guide was removed from this workspace, so this writer is deliberately
    generic. When the exact platform schema is restored, adapt only this function and keep the
    preprocessing/model code unchanged.
    """
    rows: list[dict[str, Any]] = []
    for i, point_id in enumerate(predictions["point_id"]):
        crop_id = int(predictions["crop_type_id"][i])
        rows.append(
            {
                "point_id": int(point_id),
                "crop_type_id": crop_id,
                "crop_type": CROP_TYPE_NAMES[crop_id] if 0 <= crop_id < len(CROP_TYPE_NAMES) else "unknown",
                "phenophase_doy": {name: int(predictions["phenophase_doy"][i, j]) for j, name in enumerate(PHENOPHASE_NAMES)},
            }
        )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, indent=2))
