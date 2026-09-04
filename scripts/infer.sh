#!/usr/bin/env bash
# Forward a checkpoint over a cache directory and save decoded predictions
# to a `.pt` file. Y-flip TTA is on by default (a validated +0.005 test gain).
#
# Example (test cache uses UUID filenames — pass --val-scenes -1):
#   scripts/infer.sh --ckpt artifacts/checkpoints/rangeview_epoch7.pth \
#                    --cache artifacts/cache_test_full \
#                    --val-scenes -1 \
#                    --out artifacts/test_preds_ep7.pt
set -euo pipefail

cd "$(dirname "$0")/.."

exec uv run python -m rangeview.inference.infer "$@"
