#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/pc/miniconda3/envs/tagsam/bin/python
ROOT=/home/pc/research/TAG_condensation
SNAPSHOT=$ROOT/work_runs/phase12_20260920_snapshot
DATA_ROOT=/home/pc/research/GraphCLIP/processed_data
CACHE_ROOT=$ROOT/results/p5_rate_curve_small_20260919
C0_ROOT=$ROOT/results/tgc_fresh_r10_source_20260926
SEMANTIC_ROOT=$ROOT/results/tgc_fresh_semantic_recursive_20260926
PYTHONPATH_VALUE=$SNAPSHOT/code/v2_llm_fusion:$SNAPSHOT/code/frozen_v1_2026_09_09
RUNNER=$SNAPSHOT/code/v2_llm_fusion/run_teacher_guided_message_revision.py

run_group() {
  local dataset="$1" gpu="$2" mode="$3" mix="$4" swap="$5" out="$6"
  mkdir -p "$out/logs"
  for order in order19611 order19612; do
    for stage in 1 2; do
      local cell="$out/${dataset}_${order}_s${stage}"
      local log="$out/logs/${dataset}_${order}_s${stage}.log"
      if [[ -f "$cell/audit.json" ]]; then
        echo "SKIP $mode $dataset $order s$stage"
        continue
      fi
      if [[ -e "$cell" ]]; then
        echo "ERROR incomplete output exists: $cell" >&2
        return 2
      fi
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$PYTHONPATH_VALUE" \
        "$PYTHON" -u "$RUNNER" \
        --dataset "$dataset" \
        --order "$order" \
        --stage "$stage" \
        --data "$DATA_ROOT/$dataset.pt" \
        --text-cache "$CACHE_ROOT/${dataset}_r005/text_emb_pool_only.pt" \
        --c0-root "$C0_ROOT" \
        --semantic-root "$SEMANTIC_ROOT" \
        --output "$cell" \
        --mode "$mode" \
        --device cuda:0 \
        --teacher-steps 4 \
        --inner-steps 2 \
        --outer-steps 4 \
        --batch 16 \
        --top-r 8 \
        --message-mix "$mix" \
        --swap-fraction "$swap" \
        --old-radius 0.05 \
        --new-radius 0.25 \
        --teacher-seeds 9300 9301 \
        --validation-teacher-seeds 9310 \
        > "$log" 2>&1
    done
  done
}

OUT_X=$ROOT/results/tgr_attribution_xonly_mix0_small_20260927
OUT_T=$ROOT/results/tgr_attribution_topology_only_mix20_small_20260927

run_group cora 0 x_only 0.0 0.10 "$OUT_X" &
PID_X_CORA=$!
run_group citeseer 0 x_only 0.0 0.10 "$OUT_X" &
PID_X_CITE=$!
wait "$PID_X_CORA"
wait "$PID_X_CITE"

run_group cora 0 topology_only 0.20 0.10 "$OUT_T" &
PID_T_CORA=$!
run_group citeseer 0 topology_only 0.20 0.10 "$OUT_T" &
PID_T_CITE=$!
wait "$PID_T_CORA"
wait "$PID_T_CITE"

echo TGR_ATTRIBUTION_20260927_COMPLETE
