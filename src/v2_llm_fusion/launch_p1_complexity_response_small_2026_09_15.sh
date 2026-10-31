#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="/home/pc/miniconda3/envs/tagsam/bin/python"
MINILM="/home/pc/research/TAG_condensation/code/graphclip_unified_v3_2026_09_13/minilm_from_graphclip"
RUNNER="run_unified_v3_experimental.py"
OUT="p1_small_complexity_response_2026_09_15"

run_one() {
  local ds="$1" device="$2"
  "$PY" "$ROOT/$RUNNER" \
    --dataset "$ds" \
    --minilm "$MINILM" \
    --rate 0.05 \
    --budget-mode fixed-rate \
    --allocation-mode subcluster_complexity \
    --response-marginal \
    --response-base-allocation subcluster_complexity \
    --response-marginal-rounds 1 \
    --response-marginal-threshold 0.0 \
    --n-stages 3 \
    --device "$device" \
    --arms append_fixed \
    --output "$OUT/$ds"
}

run_one "${1:?dataset}" "${2:-cuda:0}"
