#!/usr/bin/env bash
set -euo pipefail

# P1 audit rerun: keep/new quota is matched across allocation modes.
# The historical legacy_15 floor is intentionally not enabled here.
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="/home/pc/miniconda3/envs/tagsam/bin/python"
MINILM="/home/pc/research/TAG_condensation/code/graphclip_unified_v3_2026_09_13/minilm_from_graphclip"
OUT="p1_controlled_quota_2026_09_15"
RUNNER="run_unified_v3_continual_controlled.py"

run_one() {
  local ds="$1" rate="$2" alloc="$3" device="$4"
  "$PY" "$ROOT/$RUNNER" \
    --dataset "$ds" \
    --minilm "$MINILM" \
    --rate "$rate" \
    --budget-mode fixed-rate \
    --allocation-mode "$alloc" \
    --n-stages 3 \
    --device "$device" \
    --output "$OUT/p1_${ds}_r$(printf '%04d' "$(awk "BEGIN {print $rate*1000}")")_${alloc}" \
    --arms append_fixed k_center random_real
}

if [[ "${1:-}" == "all" ]]; then
  device="${2:-cuda:1}"
  for ds in cora citeseer; do
    for alloc in legacy_15 equal linear sqrt dispersion; do
      run_one "$ds" "0.05" "$alloc" "$device"
    done
  done
else
  run_one "${1:?dataset}" "${2:?rate}" "${3:?allocation}" "${4:-cuda:0}"
fi
