#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/pc/research/TAG_condensation
WORK=$ROOT/work_runs/cotag_q_arxiv_base4_20261018
OUT=$ROOT/results/cotag_q_arxiv_base4_matched_20261018
PY=/home/pc/miniconda3/envs/tagsam/bin/python
RUNNER=$WORK/v2_llm_fusion/run_unified_q_matrix_20261008.py
REFERENCE=$WORK/reference.json
TEXT_CACHE=$ROOT/results/tgc_large_full_history_20260926/arxiv_base4
MINILM=/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip

mkdir -p "$OUT"

# Reuse the already completed cell without rerunning it.
if [[ ! -f "$OUT/o9101_c271006/matrix/results.json" ]]; then
  mkdir -p "$OUT/o9101_c271006/matrix"
  cp "$ROOT/results/cotag_q_arxiv_base4_20261018/matrix/results.json" \
     "$OUT/o9101_c271006/matrix/results.json"
fi

orders=(9101 9102 9103)
seeds=(271006 271108 271109)
for seed in "${seeds[@]}"; do
  for order in "${orders[@]}"; do
    cell="$OUT/o${order}_c${seed}"
    result="$cell/matrix/results.json"
    if [[ -f "$result" ]] && grep -q '\"status\": \"complete\"' "$result"; then
      echo "{\"event\":\"reuse\",\"order\":$order,\"construction\":$seed}"
      continue
    fi
    mkdir -p "$cell/matrix"
    echo "{\"event\":\"start\",\"order\":$order,\"construction\":$seed}"
    "$PY" -u "$RUNNER" \
      --dataset arxiv \
      --reference "$REFERENCE" \
      --text-cache "$TEXT_CACHE" \
      --minilm "$MINILM" \
      --output "$cell/matrix" \
      --device cuda:0 \
      --arms uq_dm_edgefree cat_cctp gecc_hard \
      --order-seed "$order" \
      --construction-seed "$seed" \
      --budget-mode fixed_rate \
      --student-seeds 150 151 152 \
      --student-steps 2048 \
      --q-steps 30 \
      --sinkhorn-max-steps 2048 \
      > "$cell/run.log" 2>&1
    echo "{\"event\":\"finish\",\"order\":$order,\"construction\":$seed}"
  done
done

echo "{\"status\":\"complete\"}"
