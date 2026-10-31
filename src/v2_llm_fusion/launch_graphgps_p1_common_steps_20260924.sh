#!/usr/bin/env bash
set -euo pipefail

source /home/pc/miniconda3/etc/profile.d/conda.sh
conda activate tagsam
cd /home/pc/research/TAG_condensation
export PYTHONPATH=/home/pc/research/GraphCLIP

ROOT=/home/pc/research/TAG_condensation/results/graphgps_p1_common_steps_20260924
mkdir -p "$ROOT"

# Match the C0 stage-wise update budget for every arm. This is the fair
# correction to the exploratory row-scaled steps used in the first P1 pass.
for dataset in cora citeseer; do
  for stage in 0 1 2; do
    case "$stage" in
      0) steps=336 ;;
      1) steps=944 ;;
      2) steps=1504 ;;
    esac
    out="$ROOT/${dataset}_order19601_s${stage}_common_steps${steps}.json"
    if [[ -s "$out" ]]; then
      echo "SKIP $out"
      continue
    fi
    python code/v2_llm_fusion/run_graphgps_static_calibration.py \
      --dataset "$dataset" --order order19601 --stage "$stage" \
      --arms kcenter raw_pool \
      --student-seeds 150 151 152 153 154 155 \
      --student-steps "$steps" \
      --batch-size 4 --eval-batch-size 8 --output "$out"
  done
done
