#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/pc/research/TAG_condensation
PY=/home/pc/miniconda3/envs/tagsam/bin/python
OUT=$ROOT/results/cotag_q_four_gap_20261015_v1
MODE=${1:?text,fullhistory,cost}
GPU=${2:-0}
if [ "$#" -gt 2 ]; then
  TEXT_SEEDS=("${@:3}")
else
  TEXT_SEEDS=(271006 271108 271109)
fi
RUN=$(date +%Y%m%dT%H%M%S)
LOGS="$OUT/logs/$MODE-$RUN"
mkdir -p "$LOGS"
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$ROOT/experiments/uq_final_evidence_20261012/code:$ROOT/code/v2_llm_fusion:$ROOT"
failures=0
run_cell() {
  task=$1 dataset=$2 arm=$3 seed=$4
  cell="${dataset}_o9101_c${seed}_${arm}"
  output="$OUT/$task/$cell"
  if [ -f "$output/results.json" ] && grep -q '"status": "complete"' "$output/results.json"; then
    return
  fi
  rc=0
  "$PY" -u "$ROOT/tools/run_cotag_four_gap_20261015.py" --task "$task" --arm "$arm" \
    --dataset "$dataset" --construction-seed "$seed" --output "$output" \
    --reference "$ROOT/experiments/uq_final_evidence_20261012/${dataset}_reference.json" \
    --text-cache "$ROOT/results/public_baselines_text_cache_20261001/$dataset" \
    --minilm /home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip \
    --device cuda:0 > "$LOGS/${task}_${cell}.log" 2>&1 || rc=$?
  printf '%s %s exit=%s %s\n' "$task" "$cell" "$rc" "$(date -Is)" >> "$LOGS/status"
  if [ "$rc" -ne 0 ]; then failures=$((failures+1)); fi
}
printf 'running %s GPU=%s\n' "$(date -Is)" "$GPU" > "$LOGS/status"
case "$MODE" in
  text)
    for seed in "${TEXT_SEEDS[@]}"; do
      for dataset in bookhis arxiv sportsfit; do
        for arm in no_text_init no_pair_feedback; do run_cell text "$dataset" "$arm" "$seed"; done
      done
    done ;;
  fullhistory)
    for dataset in bookhis arxiv sportsfit; do
      for arm in full_ef full_topology; do run_cell fullhistory "$dataset" "$arm" 271006; done
    done ;;
  cost)
    for dataset in bookhis arxiv sportsfit; do run_cell cost "$dataset" dense_vs_sparse 271006; done ;;
  *) exit 2 ;;
esac
printf 'complete failures=%s %s\n' "$failures" "$(date -Is)" >> "$LOGS/status"
test "$failures" -eq 0
