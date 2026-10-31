#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="/home/pc/miniconda3/envs/tagsam/bin/python"
MINILM="/home/pc/research/TAG_condensation/code/graphclip_unified_v3_2026_09_13/minilm_from_graphclip"
RUNNER="run_unified_v3_continual_controlled.py"
OUT="p1_budget_policies_2026_09_15"

run_case() {
  local ds="$1" mode="$2" tag="$3" total="${4:-}" device="${5:-cuda:1}"
  local cmd=("$PY" "$ROOT/$RUNNER" --dataset "$ds" --minilm "$MINILM"
             --rate 0.05 --budget-mode "$mode" --allocation-mode sqrt
             --n-stages 3 --device "$device" --arms append_fixed
             --output "$OUT/${ds}_${tag}")
  if [[ -n "$total" ]]; then cmd+=(--total-budget "$total"); fi
  if [[ "$mode" == "hybrid" ]]; then cmd+=(--min-increment 5); fi
  "${cmd[@]}"
}

device="${1:-cuda:1}"
for ds in cora citeseer; do
  if [[ "$ds" == "cora" ]]; then total=94; else total=111; fi
  run_case "$ds" fixed-total fixed_total "$total" "$device"
  run_case "$ds" hybrid hybrid_min5 "" "$device"
  run_case "$ds" adaptive-rate adaptive_rate "" "$device"
done
