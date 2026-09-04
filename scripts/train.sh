#!/usr/bin/env bash
# Train the RangeView fusion detector.
#
# Best single-model recipe (yields test mAP 0.8424):
#   --epochs 14 --optimizer adamw --lr 7e-4 --lr-decay 0.8 --warmup-epochs 1 \
#   --no-vfe --no-iou-head --no-residuals
set -euo pipefail

cd "$(dirname "$0")/.."

exec uv run python -m rangeview.training.train "$@"
