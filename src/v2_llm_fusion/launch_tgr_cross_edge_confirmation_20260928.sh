#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/pc/research/TAG_condensation
SNAP=$ROOT/work_runs/tgr_cross_edge_confirmation_20260928
PYTHON=/home/pc/miniconda3/envs/tagsam/bin/python
export PYTHONPATH=$SNAP/code/v2_llm_fusion:$SNAP/code/frozen_v1_2026_09_09
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES="${2:-0}"
DATASET="$1"
OUT=$ROOT/results/tgr_cross_edge_confirmation_20260928
for phase in prepare evaluate audit; do
  "$PYTHON" -u "$SNAP/code/v2_llm_fusion/run_tgr_cross_edge_confirmation.py" \
    --phase "$phase" --dataset "$DATASET" --order order19613 \
    --output "$OUT" --device cuda:0
done
echo "TGR_CROSS_EDGE_CONFIRMATION_COMPLETE $DATASET"
