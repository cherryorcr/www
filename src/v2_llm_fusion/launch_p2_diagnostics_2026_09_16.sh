#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?dataset}"
DEVICE="${2:-cuda:0}"
ROOT="/home/pc/research/TAG_condensation"
CODE="$ROOT/code/graphclip_unified_v3_2026_09_13"
PY="/home/pc/miniconda3/envs/tagsam/bin/python"
export OMP_NUM_THREADS=4
cd "$ROOT"

bash "$CODE/launch_p2_rate_and_regression_2026_09_16.sh" "$DATASET" "$DEVICE"

"$PY" "$CODE/run_p2_numerical_guard.py" \
  --matrix "results/p2_token_controls_2026_09_16/$DATASET" \
  --output "results/p2_numerical_guard_2026_09_16/$DATASET" \
  --minilm "$CODE/minilm_from_graphclip" --device "$DEVICE"

"$PY" "$CODE/run_p2_metric_diagnostics.py" \
  --matrix "results/p2_token_controls_2026_09_16/$DATASET" \
  --output "results/p2_metric_diagnostics_2026_09_16/$DATASET" \
  --minilm "$CODE/minilm_from_graphclip" --device "$DEVICE"

echo "P2_DIAGNOSTICS_QUEUE_DONE $DATASET"
