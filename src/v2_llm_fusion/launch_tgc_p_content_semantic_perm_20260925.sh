#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/pc/miniconda3/envs/tagsam/bin/python
ROOT=/home/pc/research/TAG_condensation
WORK=$ROOT/work_runs/phase12_20260920_snapshot
RUNNER=$WORK/code/v2_llm_fusion/run_tgc_p_content_semantic_perm.py
R11=$ROOT/results/tgc_r11_c0_anchor_20260923b
MAP1=$ROOT/results/tgc_two_prototype_role_permutation_20260924
MAP2=$ROOT/results/tgc_two_prototype_role_permutation_20260925
CACHE=$ROOT/results/p5_rate_curve_small_20260919
MINILM=/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip
OUT=$ROOT/results/tgc_p_content_two_proto_20260925d

mkdir -p "$OUT/logs"

run_dataset() {
  local dataset="$1"
  local device="$2"
  for order in order19601 order19602 order19603; do
    local cell="$OUT/$dataset/$order"
    local log="$OUT/logs/${dataset}_${order}.log"
    if [[ -f "$cell/results.json" ]]; then
      echo "SKIP $dataset $order"
      continue
    fi
    mkdir -p "$cell"
    CUDA_VISIBLE_DEVICES="$device" PYTHONPATH="$ROOT/code/v2_llm_fusion:$WORK/code/v2_llm_fusion" \
      "$PYTHON" -u "$RUNNER" \
      --dataset "$dataset" \
      --order "$order" \
      --r11-source "$R11" \
      --mapping-root-stage1 "$MAP1" \
      --mapping-root-stage2 "$MAP2" \
      --output "$OUT" \
      --cache-root "$CACHE" \
      --minilm "$MINILM" \
      --device cuda:0 \
      --stages 0 1 2 \
      --report-seeds 150 151 152 153 154 155 \
      > "$log" 2>&1
  done
}

run_dataset cora 0 &
PID_CORA=$!
run_dataset citeseer 1 &
PID_CITESEER=$!
wait "$PID_CORA"
wait "$PID_CITESEER"
echo TGC_P_CONTENT_SEMANTIC_PERM_BATCH_COMPLETE
