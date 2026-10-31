#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/pc/research/TAG_condensation
PY=/home/pc/miniconda3/envs/tagsam/bin/python
SCRIPT=$ROOT/tools/run_uq_exact5_no_node_text_20261015.py
OUT=$ROOT/results/uq_exact5_no_node_text_20261015
GPU=${1:-0}
mkdir -p "$OUT" "$OUT/logs"
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$ROOT/experiments/uq_final_evidence_20261012/code:$ROOT/code/v2_llm_fusion:$ROOT"

failures=0
printf 'running %s gpu=%s\n' "$(date -Is)" "$GPU" > "$OUT/status"
for seed in 271006 271108 271109; do
  for dataset in arxiv sportsfit bookhis; do
    cell="${dataset}_o9101_c${seed}"
    output="$OUT/$cell"
    mkdir -p "$output"
    if [ -s "$output/results.json" ] && grep -q '"status": "complete"' "$output/results.json"; then
      printf 'skip %s existing\n' "$cell" >> "$OUT/status"
      continue
    fi
    reference="$ROOT/experiments/uq_final_evidence_20261012/${dataset}_reference.json"
    cache="$ROOT/results/public_baselines_text_cache_20261001/$dataset"
    log="$OUT/logs/${cell}.log"
    rc=0
    "$PY" -u "$SCRIPT" --dataset "$dataset" --reference "$reference" \
      --text-cache "$cache" --minilm /home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip \
      --output "$output" --device cuda:0 --construction-seed "$seed" > "$log" 2>&1 || rc=$?
    printf '%s exit=%s %s\n' "$cell" "$rc" "$(date -Is)" >> "$OUT/status"
    if [ "$rc" -ne 0 ]; then failures=$((failures+1)); fi
  done
done
printf 'complete failures=%s %s\n' "$failures" "$(date -Is)" > "$OUT/status"
test "$failures" -eq 0
