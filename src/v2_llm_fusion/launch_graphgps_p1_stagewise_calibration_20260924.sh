#!/usr/bin/env bash
set -euo pipefail

source /home/pc/miniconda3/etc/profile.d/conda.sh
conda activate tagsam
cd /home/pc/research/TAG_condensation
export PYTHONPATH=/home/pc/research/GraphCLIP

ROOT=/home/pc/research/TAG_condensation/results/graphgps_p1_stagewise_calibration_20260924
mkdir -p "$ROOT"

# Diagnostic only: same stage, same arms, same steps. This does not replace
# the completed six-seed P1 matrix and is not used for method selection.
for dataset in cora citeseer; do
  for steps in 336 1024; do
    out="$ROOT/${dataset}_order19601_s0_steps${steps}.json"
    if [[ -s "$out" ]]; then
      echo "SKIP $out"
      continue
    fi
    python code/v2_llm_fusion/run_graphgps_static_calibration.py \
      --dataset "$dataset" --order order19601 --stage 0 \
      --arms c0 raw_pool \
      --student-seeds 150 151 152 \
      --student-steps "$steps" \
      --batch-size 4 --eval-batch-size 8 --output "$out"
  done
done
