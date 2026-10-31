#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/pc/miniconda3/envs/tagsam/bin/python
ROOT=/home/pc/research/TAG_condensation
SNAP=$ROOT/work_runs/tgr_student_consistent_20260928
DATA_ROOT=/home/pc/research/GraphCLIP/processed_data
CACHE_ROOT=$ROOT/results/p5_rate_curve_small_20260919
OUT=$ROOT/results/tgr_student_consistent_independent_20260928
PYTHONPATH_VALUE=$SNAP/code/v2_llm_fusion:$SNAP/code/frozen_v1_2026_09_09

for dataset in cora citeseer; do
  CONF_ROOT=$ROOT/results/tgr_cross_edge_confirmation_20260928/$dataset
  for stage in 1 2; do
    cell="$OUT/${dataset}_order19613_s${stage}"
    if [[ -f "$cell/protocol.json" ]]; then
      echo "SKIP $dataset order19613 s$stage"
      continue
    fi
    mkdir -p "$OUT/logs"
    PYTHONPATH="$PYTHONPATH_VALUE" CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u \
      "$SNAP/code/v2_llm_fusion/run_tgr_cross_edge_transport.py" \
      --dataset "$dataset" --order order19613 --stage "$stage" \
      --data "$DATA_ROOT/$dataset.pt" \
      --text-cache "$CACHE_ROOT/${dataset}_r005/text_emb_pool_only.pt" \
      --c0-root "$CONF_ROOT/source" --semantic-root "$CONF_ROOT/semantic" \
      --output "$cell" --device cuda:0 \
      --modes c0 student_consistent --cross-degree 1 \
      --random-seed 9821 --student-consistent-seeds 17 \
      --student-consistent-steps 1 --student-consistent-lr 0.001 \
      --student-consistent-batch 16 --student-consistent-max-swaps 1 \
      --student-consistent-new-margin 0.0 \
      --student-consistent-old-drop 0.0 \
      --student-consistent-total-increase 0.0 \
      > "$OUT/logs/${dataset}_order19613_s${stage}.log" 2>&1
  done
done
echo TGR_STUDENT_CONSISTENT_INDEPENDENT_CONSTRUCTION_20260928_COMPLETE
