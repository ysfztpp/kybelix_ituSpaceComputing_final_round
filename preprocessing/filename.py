from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class ParsedTiffName:
    region_id: str
    start_raw: str
    end_raw: str
    start_norm: str
    end_norm: str
    start_dt: datetime | None
    end_dt: datetime | None
    level: str
    band_token: str
    band_id: str


def normalize_region_prefix(name: str) -> str:
    name = re.sub(r"^(region\d+)-", r"\1_", name)
    name = re.sub(r"^(region\d+)(\d{4}-)", r"\1_\2", name)
    return name


def canonical_name(name: str) -> str:
    name = normalize_region_prefix(Path(name).name)
    name = name.replace("(Raw)(1)", "(Raw)")
    return name


def normalize_timestamp(raw_value: str) -> tuple[str | None, datetime | None]:
    value = raw_value.replace("_", "-")
    parts = value.split("-")
    if len(parts) == 4:
        parts.append("00")
    if len(parts) != 5:
        return None, None
    normalized = "-".join(parts)
    try:
        parsed = datetime.strptime(normalized, "%Y-%m-%d-%H-%M")
    except ValueError:
        return normalized, None
    return normalized, parsed


def parse_tiff_name(name: str) -> ParsedTiffName | None:
    name = normalize_region_prefix(Path(name).name)
    if "_Sentinel-2_" not in name:
        return None

    prefix, suffix = name.split("_Sentinel-2_", 1)
    prefix_match = re.fullmatch(
        r"(region\d+)_(\d{4}-\d{2}-\d{2}-\d{2}(?:[-_]\d{2})?)_(\d{4}-\d{2}-\d{2}-\d{2}(?:[-_]\d{2})?)",
        prefix,
    )
    if prefix_match is None:
        return None

    region_id, start_raw, end_raw = prefix_match.groups()
    suffix_match = re.fullmatch(r"(L\d[A-Z]?)_(.+)\.tiff", suffix)
    if suffix_match is None:
        return None

    level, band_token = suffix_match.groups()
    start_norm, start_dt = normalize_timestamp(start_raw)
    end_norm, end_dt = normalize_timestamp(end_raw)
    if start_norm is None or end_norm is None:
        return None

    band_id = band_token.split("_", 1)[0]
    return ParsedTiffName(
        region_id=region_id,
        start_raw=start_raw,
        end_raw=end_raw,
        start_norm=start_norm,
        end_norm=end_norm,
        start_dt=start_dt,
        end_dt=end_dt,
        level=level,
        band_token=band_token,
        band_id=band_id,
    )


def level_rank(level: str) -> int:
    if level == "L2A":
        return 0
    if level == "L1C":
        return 1
    return 2


def doy_from_timestamp(value: str) -> int:
    return datetime.strptime(value, "%Y-%m-%d-%H-%M").timetuple().tm_yday
