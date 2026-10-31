#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/home/pc/research/TAG_condensation
TASK_SOURCE="$TASK_ROOT/work_runs/phase12_20260920_snapshot/code/v2_llm_fusion"
TASK_PY=/home/pc/miniconda3/envs/tagsam/bin/python
TASK_RESULTS="$TASK_ROOT/results/phase12_frozen_20260920"
if [[ "${1:-}" == launch ]]; then
    mkdir -p "$TASK_RESULTS"
    nohup bash "$0" cora 0 > "$TASK_RESULTS/cora.log" 2>&1 < /dev/null &
    echo "cora_pid=$!"
    nohup bash "$0" citeseer 1 > "$TASK_RESULTS/citeseer.log" 2>&1 < /dev/null &
    echo "citeseer_pid=$!"
    exit 0
fi
TASK_DATASET="$1"
TASK_GPU="$2"
trap 'TASK_CODE=$?; printf "%s\n" "$TASK_CODE" > "$TASK_RESULTS/${TASK_DATASET}_exit_code.txt"' EXIT
cd "$TASK_SOURCE"
export TOKENIZERS_PARALLELISM=false
"$TASK_PY" -u run_frozen_baseline_joint_kernel.py --dataset "$TASK_DATASET" \
    --device "cuda:$TASK_GPU" --phase baseline --output "$TASK_RESULTS/baseline/$TASK_DATASET"
"$TASK_PY" -u run_frozen_baseline_joint_kernel.py --dataset "$TASK_DATASET" \
    --device "cuda:$TASK_GPU" --phase matrix --output "$TASK_RESULTS/matrix/$TASK_DATASET"
