#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="/home/pc/miniconda3/envs/tagsam/bin/python"
MINILM="/home/pc/research/TAG_condensation/code/graphclip_unified_v3_2026_09_13/minilm_from_graphclip"
RUNNER="run_unified_v4_p2.py"

DATASET="${1:?dataset}"
OPERATOR="${2:?text operator}"
RECIPE="${3:-joint_g5_c320}"
DEVICE="${4:-cuda:0}"
OUT_ROOT="${5:-p2_text_2026_09_16}"
REFERENCE="${6:-}"

ARGS=(
  --dataset "$DATASET"
  --minilm "$MINILM"
  --rate 0.05
  --budget-mode fixed-rate
  --allocation-mode dispersion
  --n-stages 3
  --arms append_fixed
  --text-operator "$OPERATOR"
  --text-recipe "$RECIPE"
  --text-diversity-lambda 0.5
  --dual-prototype-alpha 0.5
  --device "$DEVICE"
  --output "results/$OUT_ROOT/${DATASET}_${OPERATOR}_${RECIPE}"
)
if [[ -n "$REFERENCE" ]]; then
  ARGS+=(--topology-reference "$REFERENCE")
fi
"$PY" "$ROOT/$RUNNER" "${ARGS[@]}"
