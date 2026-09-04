#!/usr/bin/env bash
# Score a checkpoint on held-out val scenes using the OFFICIAL frame-wise scorer.
#
# Example:
#   scripts/evaluate.sh --ckpt artifacts/checkpoints/rangeview_epoch7.pth \
#                       --cache artifacts/cache_train --val-scenes 31 34
set -euo pipefail

cd "$(dirname "$0")/.."

exec uv run python -m rangeview.eval.score_ckpt "$@"
