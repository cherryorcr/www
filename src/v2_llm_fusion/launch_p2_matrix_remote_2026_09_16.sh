#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="/home/pc/miniconda3/envs/tagsam/bin/python"
MINILM="/home/pc/research/TAG_condensation/code/graphclip_unified_v3_2026_09_13/minilm_from_graphclip"
RUNNER="run_unified_v4_p2.py"
DATASET="${1:?dataset}"
DEVICE="${2:-cuda:0}"
REFERENCE="/home/pc/research/TAG_condensation/results/p2_text_reference_2026_09_16/${DATASET}"
OUT_ROOT="p2_text_matrix_2026_09_16"

run_one() {
  local operator="$1" recipe="${2:-joint_g5_c320}" lambda="${3:-0.5}" alpha="${4:-0.5}"
  local out="results/${OUT_ROOT}/${DATASET}_${operator}_${recipe}_l${lambda}_a${alpha}"
  if [[ -f "$out/results.json" ]]; then
    echo "SKIP completed $out"
    return
  fi
  echo "RUN $DATASET $operator $recipe lambda=$lambda alpha=$alpha"
  "$PY" "$ROOT/$RUNNER" \
    --dataset "$DATASET" \
    --minilm "$MINILM" \
    --rate 0.05 \
    --budget-mode fixed-rate \
    --allocation-mode dispersion \
    --n-stages 3 \
    --arms append_fixed \
    --text-operator "$operator" \
    --text-recipe "$recipe" \
    --text-diversity-lambda "$lambda" \
    --dual-prototype-alpha "$alpha" \
    --topology-reference "$REFERENCE" \
    --device "$DEVICE" \
    --output "$out"
}

# Fixed g=5/c=320 budget: compare text construction operators.
run_one concat joint_g5_c320
run_one diverse_concat joint_g5_c320 0.25
run_one diverse_concat joint_g5_c320 0.50
run_one diverse_concat joint_g5_c320 0.75
run_one dual_proto joint_g5_c320 0.50 0.25
run_one dual_proto joint_g5_c320 0.50 0.50
run_one dual_proto joint_g5_c320 0.50 0.75
run_one mean_pool joint_g5_c320
run_one weighted_pool joint_g5_c320
run_one set_pool joint_g5_c320
run_one no_reencode joint_g5_c320
run_one no_fuse joint_g5_c320
run_one no_text joint_g5_c320

# Text-budget controls: keep the same rows and topology reference.
run_one concat joint_g3_c192
run_one concat joint_g8_c512

echo "P2_MATRIX_DONE $DATASET"
