#!/usr/bin/env bash
set -u

ROOT=/home/pc/research/TAG_condensation
EXP=$ROOT/experiments/uq_rate_sensitivity_20261012
OUT=$ROOT/results/uq_rate_sensitivity_20261012
PY=/home/pc/miniconda3/envs/tagsam/bin/python
CODE=$ROOT/experiments/uq_q_seed_attribution_20261010/code
MINILM=/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip
mkdir -p "$OUT/logs"

run_cell() {
  local dataset=$1 rate=$2 gpu=$3
  local tag=${dataset}_r${rate}
  env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
    PYTHONPATH="$CODE:$ROOT/code/v2_llm_fusion:$ROOT/tools:$ROOT" \
    "$PY" -u "$EXP/run_uq_rate_sensitivity_20261012.py" \
    --dataset "$dataset" --rate "$rate" --reference "$EXP/${dataset}_reference.json" \
    --text-cache "$ROOT/results/public_baselines_text_cache_20261001/$dataset" \
    --minilm "$MINILM" --output "$OUT/$tag" --device cuda:0 \
    > "$OUT/logs/$tag.log" 2>&1
  local rc=$?
  printf '%s %s exit=%s\n' "$tag" "$(date -Is)" "$rc" >> "$OUT/cell_status.txt"
  return "$rc"
}

lane_arxiv() {
  run_cell arxiv 0.01 0
  run_cell arxiv 0.10 0
}

lane_amazon() {
  run_cell bookhis 0.01 1
  run_cell bookhis 0.10 1
  run_cell sportsfit 0.01 1
  run_cell sportsfit 0.10 1
}

lane_arxiv & pid_arxiv=$!
lane_amazon & pid_amazon=$!
wait "$pid_arxiv"
wait "$pid_amazon"
printf 'terminal %s; inspect all six cell statuses\n' "$(date -Is)" > "$OUT/queue.status"
