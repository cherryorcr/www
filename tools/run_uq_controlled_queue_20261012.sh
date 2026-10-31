#!/usr/bin/env bash
set -u
ROOT=/home/pc/research/TAG_condensation
EXP=$ROOT/experiments/uq_final_evidence_20261012
OUT=$ROOT/results/uq_final_evidence_20261012
PY=/home/pc/miniconda3/envs/tagsam/bin/python
mkdir -p "$OUT/logs"
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$EXP/code:$ROOT/code/v2_llm_fusion:$ROOT/tools:$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PY" -u "$EXP/code/audit_uq_controls_20261012.py" > "$OUT/control_numerical_audit.json" 2> "$OUT/logs/control_numerical_audit.log" || exit 1
for seed in 271006 271108 271109; do
  for dataset in bookhis sportsfit arxiv; do
    arms=(control_shared control_independent control_shared_no_residual control_independent_no_residual)
    if [ "$seed" = 271006 ]; then
      arms+=(control_shared_equal control_independent_equal control_soft_stored)
    fi
    tag=${dataset}_o9101_c${seed}
    "$PY" -u "$EXP/code/run_uq_final_confirmation_20261012.py" \
      --dataset "$dataset" --reference "$EXP/${dataset}_reference.json" \
      --text-cache "$ROOT/results/public_baselines_text_cache_20261001/$dataset" \
      --minilm /home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip \
      --output "$OUT/controls/$tag" --order-seed 9101 \
      --construction-seed "$seed" --arms "${arms[@]}" \
      > "$OUT/logs/control_$tag.log" 2>&1
    rc=$?
    printf '%s exit=%s %s\n' "$tag" "$rc" "$(date -Is)" >> "$OUT/control_status.txt"
  done
done
printf 'complete %s\n' "$(date -Is)" > "$OUT/control_queue.status"
