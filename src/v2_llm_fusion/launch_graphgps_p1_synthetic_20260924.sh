#!/usr/bin/env bash
set -euo pipefail

source /home/pc/miniconda3/etc/profile.d/conda.sh
conda activate tagsam
cd /home/pc/research/TAG_condensation
export PYTHONPATH=/home/pc/research/GraphCLIP

ROOT=/home/pc/research/TAG_condensation/results/graphgps_calibration_p1_20260924
mkdir -p "$ROOT"
for dataset in cora citeseer; do
  for stage in 0 1 2; do
    out="$ROOT/${dataset}_order19601_s${stage}_synthetic.json"
    if [[ -s "$out" ]]; then
      echo "SKIP $out"
      continue
    fi
    python code/v2_llm_fusion/run_graphgps_static_calibration.py \
      --dataset "$dataset" --order order19601 --stage "$stage" \
      --arms c0 r11 kcenter \
      --student-seeds 150 151 152 153 154 155 \
      --batch-size 4 --eval-batch-size 8 --output "$out"
  done
done
