#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/pc/miniconda3/envs/tagsam/bin/python
ROOT=/home/pc/research/TAG_condensation
WORK=$ROOT/work_runs/phase12_20260920_snapshot
SOURCE=$ROOT/results/tgc_fresh_r10_source_20260926
OUT=$ROOT/results/tgc_fresh_role_aware_20260926
CACHE=$ROOT/results/p5_rate_curve_small_20260919
MINILM=/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip
PYTHONPATH_VALUE=$ROOT/code/v2_llm_fusion:$WORK/code/v2_llm_fusion

run_dataset() {
  local dataset="$1"
  local device="$2"
  for order in order19611 order19612; do
    for stage in 1 2; do
      CUDA_VISIBLE_DEVICES="$device" PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON" -u \
        "$WORK/code/v2_llm_fusion/run_tgc_role_aware_assignment_20260925.py" \
        --dataset "$dataset" --order "$order" --stage "$stage" \
        --source-root "$SOURCE" --cache-root "$CACHE" --output "$OUT" \
        --minilm "$MINILM" --device cuda:0 \
        --report-seeds 150 151 152 153 154 155 \
        > "$OUT/${dataset}_${order}_s${stage}.log" 2>&1
    done
  done
}

mkdir -p "$OUT"
run_dataset "$1" "$2"
