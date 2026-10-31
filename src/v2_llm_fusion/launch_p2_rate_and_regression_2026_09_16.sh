#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?dataset}"
DEVICE="${2:-cuda:0}"
ROOT="/home/pc/research/TAG_condensation"
CODE="$ROOT/code/graphclip_unified_v3_2026_09_13"
PY="/home/pc/miniconda3/envs/tagsam/bin/python"
export OMP_NUM_THREADS=4
cd "$ROOT"

for rate in 0.05 0.025; do
  out="results/p2_legacy_regression_2026_09_16/${DATASET}_r${rate}"
  if [[ ! -f "$out/results.json" ]]; then
    "$PY" "$CODE/run_unified_v4_p2.py" \
      --dataset "$DATASET" --output "$out" --device "$DEVICE" \
      --minilm "$CODE/minilm_from_graphclip" --rate "$rate" \
      --budget-mode fixed-rate --allocation-mode dispersion --arms append_fixed
  fi
done

"$PY" "$CODE/run_p2_token_controls.py" \
  --reference "results/p2_legacy_regression_2026_09_16/${DATASET}_r0.025" \
  --output "results/p2_token_rate0025_2026_09_16/$DATASET" \
  --minilm "$CODE/minilm_from_graphclip" --device "$DEVICE" \
  --only concat_t64_g5_r0.5_a0.5 mean_pool set_pool diverse_concat_t64_g5_r0.75_a0.5

echo "P2_RATE_AND_REGRESSION_DONE $DATASET"
