#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/pc/research/TAG_condensation
PY=/home/pc/miniconda3/envs/tagsam/bin/python
SCRIPT=$ROOT/tools/run_uq_priority_controls_20261014.py
OUT=$ROOT/results/uq_priority_controls_20261014
MODE=${1:?mode payload or no_graph}
GPU=${2:?physical GPU id}

mkdir -p "$OUT/$MODE" "$OUT/logs"
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$ROOT/tools:$ROOT/experiments/uq_final_evidence_20261012/code:$ROOT/experiments/uq_component_ablations_20261014/code:$ROOT/experiments/uq_native_retry_20261013/code:$ROOT/code/v2_llm_fusion"

status="$OUT/${MODE}.status"
printf 'running %s mode=%s gpu=%s\n' "$(date -Is)" "$MODE" "$GPU" > "$status"
failures=0
for seed in 271006 271108 271109; do
  for dataset in arxiv sportsfit bookhis; do
    output="$OUT/$MODE/${dataset}_o9101_c${seed}.json"
    if [ -s "$output" ]; then
      printf 'skip %s existing\n' "$output" >> "$status"
      continue
    fi
    reference="$ROOT/experiments/uq_final_evidence_20261012/${dataset}_reference.json"
    cache="$ROOT/results/public_baselines_text_cache_20261001/$dataset"
    log="$OUT/logs/${MODE}_${dataset}_o9101_c${seed}.log"
    printf 'start %s %s %s\n' "$dataset" "$seed" "$(date -Is)" >> "$status"
    rc=0
    "$PY" -u "$SCRIPT" --dataset "$dataset" --reference "$reference" \
      --text-cache "$cache" --minilm /home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip \
      --output "$output" --device cuda:0 --construction-seed "$seed" --mode "$MODE" > "$log" 2>&1 || rc=$?
    printf 'done %s %s rc=%s %s\n' "$dataset" "$seed" "$rc" "$(date -Is)" >> "$status"
    if [ "$rc" -ne 0 ]; then failures=$((failures+1)); fi
  done
done
printf 'complete failures=%s %s\n' "$failures" "$(date -Is)" > "$status"
test "$failures" -eq 0
