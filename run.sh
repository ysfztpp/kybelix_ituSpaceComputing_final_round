#!/usr/bin/env bash
set -euo pipefail

echo "[submission] start: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "[submission] cwd: $(pwd)"
echo "[submission] Python: $(python --version 2>&1)"
echo "[submission] repository files:"
find . -maxdepth 2 -type f \
  | sed 's#^\./##' \
  | sort \
  | grep -E '^(inference.py|run.sh|Dockerfile|configs/|scripts/|models/|data/|preprocessing/|training/|checkpoints/model.pt|artifacts/normalization/)' \
  | head -120

echo "[submission] input files:"
find /input -maxdepth 2 -type f | sort | head -120 || true

if [ -f /input/test_point.csv ]; then
  echo "[submission] using points file: /input/test_point.csv"
elif [ -f /input/points_test.csv ]; then
  echo "[submission] using points file: /input/points_test.csv"
else
  echo "[submission] ERROR: missing /input/test_point.csv or /input/points_test.csv" >&2
  exit 1
fi

echo "[submission] region_test TIFF count: $(find /input/region_test -maxdepth 1 -type f -name '*.tiff' 2>/dev/null | wc -l | tr -d ' ')"
echo "[submission] checkpoint:"
ls -lh checkpoints/model.pt

echo "[submission] validating package"
python scripts/validate_submission.py

echo "[submission] running inference"
python inference.py --config configs/submission.json

test -f /output/result.json
echo "[submission] wrote /output/result.json"
python - <<'PY'
import json
from pathlib import Path

path = Path("/output/result.json")
data = json.loads(path.read_text())
print(f"[submission] result rows: {len(data)}")
for index, (key, value) in enumerate(data.items()):
    if index >= 3:
        break
    print(f"[submission] sample result {index + 1}: {key} -> {value}")
PY
echo "[submission] done: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
