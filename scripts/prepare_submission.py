from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBMISSION_REPO = ROOT.parent / "track1_kybelix"
DEFAULT_TRAINED_MODEL = ROOT / "checkpoints" / "model.pt"
DEFAULT_SUBMISSION_CONFIG = ROOT / "configs" / "submission.json"

CODE_FILES = [
    "configs/submission.json",
    "data/__init__.py",
    "data/aux_features.py",
    "data/transforms.py",
    "models/__init__.py",
    "models/cnn_encoder.py",
    "models/model_factory.py",
    "models/query_cnn_transformer.py",
    "models/query_tsvit.py",
    "models/temporal_transformer.py",
    "preprocessing/__init__.py",
    "preprocessing/constants.py",
    "preprocessing/dataset.py",
    "preprocessing/filename.py",
    "preprocessing/inventory.py",
    "preprocessing/mapping.py",
    "preprocessing/normalization.py",
    "preprocessing/raster_io.py",
    "preprocessing/reporting.py",
    "scripts/submission_inference.py",
    "scripts/validate_submission.py",
    "training/__init__.py",
    "training/stage_decoding.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the GitLab submission repo with a chosen trained checkpoint.")
    parser.add_argument("--trained-model", type=Path, default=DEFAULT_TRAINED_MODEL)
    parser.add_argument("--submission-repo", type=Path, default=DEFAULT_SUBMISSION_REPO)
    parser.add_argument("--submission-config", type=Path, default=DEFAULT_SUBMISSION_CONFIG)
    parser.add_argument("--skip-validate", action="store_true", help="Copy files without running validate_submission.py in the target repo.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned copies without changing files.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def copy_file(src: Path, dst: Path, *, dry_run: bool) -> None:
    print(f"{src} -> {dst}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"[prepare_submission] missing {label}: {path}")
    if not path.is_file():
        raise SystemExit(f"[prepare_submission] {label} is not a file: {path}")


def require_repo(path: Path) -> None:
    if not (path / ".git").exists():
        raise SystemExit(f"[prepare_submission] target does not look like a git repo: {path}")


def validate_submission(submission_repo: Path) -> None:
    command = [sys.executable, "scripts/validate_submission.py"]
    subprocess.run(command, cwd=submission_repo, check=True)


def main() -> None:
    args = parse_args()
    trained_model = resolve(args.trained_model)
    submission_repo = resolve(args.submission_repo)
    submission_config = resolve(args.submission_config)

    require_file(trained_model, "trained checkpoint")
    require_file(submission_config, "submission config")
    require_repo(submission_repo)

    for relative in CODE_FILES:
        require_file(ROOT / relative, relative)

    copy_file(trained_model, submission_repo / "checkpoints" / "model.pt", dry_run=args.dry_run)
    copy_file(submission_config, submission_repo / "configs" / "submission.json", dry_run=args.dry_run)

    for relative in CODE_FILES:
        if relative == "configs/submission.json":
            continue
        copy_file(ROOT / relative, submission_repo / relative, dry_run=args.dry_run)

    if args.dry_run:
        print("[prepare_submission] dry run complete; no files changed.")
        return

    if not args.skip_validate:
        validate_submission(submission_repo)
    print("[prepare_submission] submission files are ready in the GitLab submission repo.")


if __name__ == "__main__":
    main()
