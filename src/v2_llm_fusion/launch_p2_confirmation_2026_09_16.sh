#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?dataset}"
DEVICE="${2:-cuda:0}"
ROOT="/home/pc/research/TAG_condensation"
CODE="$ROOT/code/graphclip_unified_v3_2026_09_13"
PY="/home/pc/miniconda3/envs/tagsam/bin/python"
export OMP_NUM_THREADS=4
cd "$ROOT"

"$PY" "$CODE/run_p2_learned_set.py" \
  --reference "results/p2_text_reference_2026_09_16/$DATASET" \
  --output "results/p2_token_controls_2026_09_16/$DATASET" \
  --minilm "$CODE/minilm_from_graphclip" --device "$DEVICE"

"$PY" "$CODE/run_p2_artifact_confirmation.py" \
  --matrix "results/p2_token_controls_2026_09_16/$DATASET" \
  --output "results/p2_token_confirmation_2026_09_16/$DATASET" \
  --minilm "$CODE/minilm_from_graphclip" --device "$DEVICE" \
  --names concat_t64_g5_r0.5_a0.5 mean_pool_t64_g5_r0.5_a0.5 \
    weighted_pool_t64_g5_r0.5_a0.5 set_pool_t64_g5_r0.5_a0.5 \
    diverse_concat_t64_g5_r0.75_a0.5

echo "P2_CONFIRMATION_QUEUE_DONE $DATASET"
