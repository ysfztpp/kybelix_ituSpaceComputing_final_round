from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a controlled training experiment suite.")
    parser.add_argument("--suite", type=Path, default=ROOT / "configs" / "experiment_suite.json")
    parser.add_argument("--only", nargs="*", default=None, help="Optional experiment names to run.")
    parser.add_argument("--dry-run", action="store_true", help="Write resolved configs and print commands without training.")
    parser.add_argument("--force", action="store_true", help="Rerun experiments even when model.pt already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_path = resolve_path(args.suite)
    suite = json.loads(suite_path.read_text())
    base_config_path = resolve_path(suite["base_config"])
    base_config = json.loads(base_config_path.read_text())
    suite_name = str(suite.get("name", suite_path.stem))
    output_root = resolve_path(suite.get("output_root", f"artifacts/models/experiments/{suite_name}"))
    config_dir = output_root / "resolved_configs"
    log_dir = output_root / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    selected = set(args.only or [])
    manifest: dict[str, Any] = {
        "suite": suite_name,
        "suite_config": str(suite_path),
        "base_config": str(base_config_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "experiments": [],
    }

    for experiment in suite["experiments"]:
        name = str(experiment["name"])
        if selected and name not in selected:
            continue
        resolved = deepcopy(base_config)
        overrides = deepcopy(experiment.get("overrides", {}))
        output_dir = output_root / name
        overrides.setdefault("output_dir", str(output_dir.relative_to(ROOT) if output_dir.is_relative_to(ROOT) else output_dir))
        deep_update(resolved, overrides)
        resolved["preserve_output_dir"] = True
        config_path = config_dir / f"{name}.json"
        config_path.write_text(json.dumps(resolved, indent=2))

        command = [sys.executable, str(ROOT / "scripts" / "train.py"), "--config", str(config_path)]
        command.extend(experiment.get("flags", []))
        model_path = resolve_path(resolved["output_dir"]) / "model.pt"
        log_path = log_dir / f"{name}.log"
        row = {
            "name": name,
            "config": str(config_path),
            "log": str(log_path),
            "model": str(model_path),
            "command": command,
            "status": "pending",
        }

        if model_path.exists() and not args.force:
            row["status"] = "skipped_existing"
            manifest["experiments"].append(row)
            print(json.dumps(row, indent=2))
            continue
        if args.dry_run:
            row["status"] = "dry_run"
            manifest["experiments"].append(row)
            print(json.dumps(row, indent=2))
            continue

        print(json.dumps({"starting": name, "log": str(log_path), "command": command}, indent=2))
        with log_path.open("w") as log_file:
            completed = subprocess.run(command, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT, text=True)
        row["returncode"] = completed.returncode
        row["status"] = "completed" if completed.returncode == 0 else "failed"
        manifest["experiments"].append(row)
        print(json.dumps(row, indent=2))
        if completed.returncode != 0:
            break

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
