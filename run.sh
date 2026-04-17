#!/usr/bin/env bash
set -euo pipefail

echo "[submission] start: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "[submission] cwd: $(pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi
echo "[submission] Python executable: ${PYTHON_BIN}"
echo "[submission] Python version: $(${PYTHON_BIN} --version 2>&1)"
SUBMISSION_INPUT_ROOT="${INPUT_ROOT:-/input}"
SUBMISSION_OUTPUT_DIR="${OUTPUT_DIR:-/output}"
SUBMISSION_OUTPUT_JSON="${SUBMISSION_OUTPUT_DIR}/result.json"
SUBMISSION_WORK_DIR="${WORK_DIR:-/tmp/kybelix_submission_work}"
SUBMISSION_CONFIG="/tmp/kybelix_submission_config.json"
echo "[submission] input root: ${SUBMISSION_INPUT_ROOT}"
echo "[submission] output json: ${SUBMISSION_OUTPUT_JSON}"
echo "[submission] work dir: ${SUBMISSION_WORK_DIR}"
echo "[submission] repository files:"
find . -maxdepth 2 -type f \
  | sed 's#^\./##' \
  | sort \
  | grep -E '^(inference.py|run.sh|Dockerfile|configs/|scripts/|models/|data/|preprocessing/|training/|checkpoints/model.pt|artifacts/normalization/)' \
  | head -120

echo "[submission] input files:"
find -L "${SUBMISSION_INPUT_ROOT}" -maxdepth 2 -type f | sort | head -120 || true

if [ -f "${SUBMISSION_INPUT_ROOT}/test_point.csv" ]; then
  echo "[submission] using points file: ${SUBMISSION_INPUT_ROOT}/test_point.csv"
elif [ -f "${SUBMISSION_INPUT_ROOT}/points_test.csv" ]; then
  echo "[submission] using points file: ${SUBMISSION_INPUT_ROOT}/points_test.csv"
else
  echo "[submission] ERROR: missing ${SUBMISSION_INPUT_ROOT}/test_point.csv or ${SUBMISSION_INPUT_ROOT}/points_test.csv" >&2
  exit 1
fi

echo "[submission] region_test TIFF count: $(find -L "${SUBMISSION_INPUT_ROOT}/region_test" -maxdepth 1 -type f -name '*.tiff' 2>/dev/null | wc -l | tr -d ' ')"
echo "[submission] checkpoint:"
ls -lh checkpoints/model.pt

${PYTHON_BIN} - <<PY
import json
from pathlib import Path

config = json.loads(Path("configs/submission.json").read_text())
config["input_root"] = "${SUBMISSION_INPUT_ROOT}"
config["output_json"] = "${SUBMISSION_OUTPUT_JSON}"
config["work_dir"] = "${SUBMISSION_WORK_DIR}"
Path("${SUBMISSION_OUTPUT_DIR}").mkdir(parents=True, exist_ok=True)
Path("${SUBMISSION_WORK_DIR}").mkdir(parents=True, exist_ok=True)
Path("${SUBMISSION_CONFIG}").write_text(json.dumps(config, indent=2))
print("[submission] runtime config:")
print(Path("${SUBMISSION_CONFIG}").read_text())
PY

echo "[submission] validating package"
${PYTHON_BIN} scripts/validate_submission.py --config "${SUBMISSION_CONFIG}"

echo "[submission] running inference"
${PYTHON_BIN} inference.py --config "${SUBMISSION_CONFIG}"

test -f "${SUBMISSION_OUTPUT_JSON}"
echo "[submission] wrote ${SUBMISSION_OUTPUT_JSON}"
${PYTHON_BIN} - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ.get("OUTPUT_DIR", "/output")) / "result.json"
data = json.loads(path.read_text())
print(f"[submission] result rows: {len(data)}")
for index, (key, value) in enumerate(data.items()):
    if index >= 3:
        break
    print(f"[submission] sample result {index + 1}: {key} -> {value}")
PY
echo "[submission] done: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
