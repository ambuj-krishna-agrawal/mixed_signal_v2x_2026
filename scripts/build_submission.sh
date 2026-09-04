#!/usr/bin/env bash
# End-to-end: cross-model WBF variant → refiner → RSU filter → Bike ground-dim
# filter → zip.  Requires prediction caches from `scripts/infer.sh` for each
# checkpoint you want to fuse.
#
# Example:
#   scripts/build_submission.sh \
#     --preds-1024 artifacts/test_preds_v2.pt artifacts/test_preds_ep8.pt artifacts/test_preds_ep8cp.pt \
#     --preds-512  artifacts/test_preds_v2_512.pt artifacts/test_preds_ep8_512.pt artifacts/test_preds_ep8cp_512.pt \
#     --test-cache artifacts/cache_test_full \
#     --v-refiner  artifacts/refiner_ckpts_2x.pt \
#     --b-refiner  artifacts/refiner_ckpts.pt \
#     --out-dir    submission_out \
#     --work-dir   artifacts
set -euo pipefail

cd "$(dirname "$0")/.."

exec uv run python -m rangeview.submission.variant "$@"
