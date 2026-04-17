#!/usr/bin/env bash
set -euo pipefail

echo "[submission] Python: $(python --version 2>&1)"
echo "[submission] Listing /input"
find /input -maxdepth 2 -type f | head -80 || true

python scripts/submission_inference.py --config configs/submission.json

test -f /output/result.json
echo "[submission] wrote /output/result.json"
