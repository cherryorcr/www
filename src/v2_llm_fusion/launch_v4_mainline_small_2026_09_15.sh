#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="/home/pc/miniconda3/envs/tagsam/bin/python"
MINILM="/home/pc/research/TAG_condensation/code/graphclip_unified_v3_2026_09_13/minilm_from_graphclip"
RUNNER="run_unified_v4_mainline_safe.py"
OUT="v4_mainline_safe_2026_09_15"

DATASET="${1:?dataset}"
DEVICE="${2:-cuda:0}"

"$PY" "$ROOT/$RUNNER" \
  --dataset "$DATASET" \
  --minilm "$MINILM" \
  --rate 0.05 \
  --budget-mode fixed-rate \
  --n-stages 3 \
  --arms append_fixed \
  --v4-mainline \
  --device "$DEVICE" \
  --output "results/$OUT/$DATASET"
