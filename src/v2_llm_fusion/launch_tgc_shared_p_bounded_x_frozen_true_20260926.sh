#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/pc/miniconda3/envs/tagsam/bin/python
ROOT=/home/pc/research/TAG_condensation
WORK=$ROOT/work_runs/safe_joint_kernel_v2_20260920_snapshot
SOURCE=$ROOT/results/tgc_r10_joint_confirmation_independent_20260925
CACHE=$ROOT/results/p5_rate_curve_small_20260919
MINILM=/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip
OUT=$ROOT/results/tgc_shared_p_bounded_x_frozen_topology_true_20260926
RUNNER=$WORK/code/v2_llm_fusion/run_tgc_shared_p_bounded_x_20260926.py
PYTHONPATH_VALUE=$ROOT/code/v2_llm_fusion:$WORK/code/v2_llm_fusion

mkdir -p "$OUT/logs"

run_dataset() {
  local dataset="$1"
  local device="$2"
  for order in order19605 order19606 order19608; do
    for stage in 1 2; do
      local cell="$OUT/${dataset}_${order}_s${stage}"
      local log="$OUT/logs/${dataset}_${order}_s${stage}.log"
      if [[ -f "$cell/results.json" ]]; then
        echo "SKIP $dataset $order s$stage"
        continue
      fi
      CUDA_VISIBLE_DEVICES="$device" PYTHONPATH="$PYTHONPATH_VALUE" \
        "$PYTHON" -u "$RUNNER" \
        --dataset "$dataset" \
        --order "$order" \
        --stage "$stage" \
        --source-root "$SOURCE" \
        --cache-root "$CACHE" \
        --output "$OUT" \
        --minilm "$MINILM" \
        --device cuda:0 \
        --outer-steps 8 \
        --inner-steps 8 \
        --freeze-topology \
        > "$log" 2>&1
    done
  done
}

run_dataset cora 0 &
PID_CORA=$!
run_dataset citeseer 1 &
PID_CITESEER=$!
wait "$PID_CORA"
wait "$PID_CITESEER"
echo TGC_SHARED_P_BOUNDED_X_TRUE_FROZEN_TOPOLOGY_COMPLETE
