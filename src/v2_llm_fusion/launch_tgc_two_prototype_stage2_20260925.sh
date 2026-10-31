#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/pc/miniconda3/envs/tagsam/bin/python
ROOT=/home/pc/research/TAG_condensation
WORK=$ROOT/work_runs/tgc_two_prototype_smoke_20260924
SOURCE=$ROOT/results/tgc_r10_joint_confirmation_20260923b
CACHE=$ROOT/results/p5_rate_curve_small_20260919
OUT=$ROOT/results/tgc_two_prototype_role_permutation_20260925
RUNNER=$WORK/code/v2_llm_fusion/run_tgc_two_prototype_role_permutation.py

mkdir -p "$OUT"
for dataset in cora citeseer; do
  for order in order19601 order19602 order19603; do
    cell="$OUT/${dataset}_${order}_s2"
    log="$OUT/${dataset}_${order}_s2.log"
    "$PYTHON" -u "$RUNNER" \
      --dataset "$dataset" \
      --order "$order" \
      --stage 2 \
      --source-root "$SOURCE" \
      --cache-root "$CACHE" \
      --output "$cell" \
      --random-seed 20260924 \
      --report-seeds 150 151 152 153 154 155 \
      > "$log" 2>&1
  done
done
echo TGC_TWO_PROTOTYPE_STAGE2_BATCH_COMPLETE
