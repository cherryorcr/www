#!/usr/bin/env bash
set -u

ROOT=/home/pc/research/TAG_condensation
EXP=$ROOT/experiments/uq_rate_sensitivity_20261012
OUT=$ROOT/results/uq_rate_sensitivity_20261012
CODE=$ROOT/experiments/uq_q_seed_attribution_20261010/code
PY=/home/pc/miniconda3/envs/tagsam/bin/python
mkdir -p "$OUT/logs"

# The archived ArXiv budget used the pool before reserving report-dev rows.
# Rebuild all three rates against the current legal construction pool.
for rate in 0.01 0.05 0.10; do
  tag=arxiv_r${rate}_fixedpool
  env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
    PYTHONPATH="$CODE:$ROOT/code/v2_llm_fusion:$ROOT/tools:$ROOT" \
    "$PY" -u "$EXP/run_uq_rate_sensitivity_20261012.py" \
    --dataset arxiv --rate "$rate" --reference "$EXP/arxiv_reference.json" \
    --text-cache "$ROOT/results/public_baselines_text_cache_20261001/arxiv" \
    --minilm /home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip \
    --output "$OUT/$tag" --device cuda:0 --allow-legacy-budget-reference \
    > "$OUT/logs/$tag.log" 2>&1
  rc=$?
  printf '%s %s exit=%s\n' "$tag" "$(date -Is)" "$rc" >> "$OUT/arxiv_fixedpool_status.txt"
done
printf 'terminal %s; inspect all three statuses\n' "$(date -Is)" > "$OUT/arxiv_fixedpool_queue.status"
