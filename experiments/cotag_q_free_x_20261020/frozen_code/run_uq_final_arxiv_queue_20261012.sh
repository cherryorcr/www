#!/usr/bin/env bash
set -u
ROOT=/home/pc/research/TAG_condensation
EXP=$ROOT/experiments/uq_final_evidence_20261012
OUT=$ROOT/results/uq_final_evidence_20261012
PY=/home/pc/miniconda3/envs/tagsam/bin/python
mkdir -p "$OUT/logs"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$EXP/code:$ROOT/code/v2_llm_fusion:$ROOT/tools:$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PY" -c 'import run_unified_q_matrix_20261008' || exit 1
for order in 9101 9102 9103; do
  for seed in 271006 271108 271109; do
    tag=arxiv_o${order}_c${seed}
    "$PY" -u "$EXP/code/run_uq_final_confirmation_20261012.py" \
      --dataset arxiv --reference "$EXP/arxiv_reference.json" \
      --text-cache "$ROOT/results/public_baselines_text_cache_20261001/arxiv" \
      --minilm /home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip \
      --output "$OUT/confirmation/$tag" --order-seed "$order" \
      --construction-seed "$seed" --save-final \
      > "$OUT/logs/$tag.log" 2>&1
    rc=$?
    printf '%s exit=%s %s\n' "$tag" "$rc" "$(date -Is)" >> "$OUT/confirmation_status.txt"
  done
done
printf 'complete %s\n' "$(date -Is)" > "$OUT/confirmation_queue.status"
