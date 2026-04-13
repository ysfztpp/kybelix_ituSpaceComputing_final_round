#!/usr/bin/env bash
set -euo pipefail

echo "Input files:"
find /input -maxdepth 2 -type f | head -50 || true

CHECKPOINT_PATH="${CHECKPOINT_PATH:-artifacts/models/query_cnn_transformer_colab/model.pt}"
python3 scripts/run_query_inference.py \
  --input-root /input \
  --checkpoint "$CHECKPOINT_PATH" \
  --normalization-json artifacts/normalization/train_patch_band_stats.json \
  --output-json /output/result.json \
  --batch-size "${BATCH_SIZE:-32}" \
  --device "${DEVICE:-auto}"
