#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/pc/research/TAG_condensation
PY=/home/pc/miniconda3/envs/tagsam/bin/python
OUT=$ROOT/results/cotag_q_four_gap_20261015_v1
GPU=${1:-1}
LOGS="$OUT/logs/tagsam-budget-$(date +%Y%m%dT%H%M%S)"
mkdir -p "$LOGS"
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$ROOT/experiments/uq_final_evidence_20261012/code:$ROOT/code/v2_llm_fusion:$ROOT"
printf 'running %s GPU=%s\n' "$(date -Is)" "$GPU" > "$LOGS/status"
failures=0
for dataset in bookhis arxiv sportsfit; do
  output="$OUT/tagsam_budget/${dataset}_o9101_c271006"
  if [ -f "$output/results.json" ] && grep -q '"status": "complete"' "$output/results.json"; then
    continue
  fi
  rc=0
  "$PY" -u "$ROOT/tools/run_cotag_tagsam_budget_20261015.py" \
    --dataset "$dataset" --output "$output" \
    --reference "$ROOT/experiments/uq_final_evidence_20261012/${dataset}_reference.json" \
    --text-cache "$ROOT/results/public_baselines_text_cache_20261001/$dataset" \
    --minilm /home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip \
    --device cuda:0 > "$LOGS/${dataset}.log" 2>&1 || rc=$?
  printf '%s exit=%s %s\n' "$dataset" "$rc" "$(date -Is)" >> "$LOGS/status"
  if [ "$rc" -ne 0 ]; then failures=$((failures+1)); fi
done
printf 'complete failures=%s %s\n' "$failures" "$(date -Is)" >> "$LOGS/status"
test "$failures" -eq 0
