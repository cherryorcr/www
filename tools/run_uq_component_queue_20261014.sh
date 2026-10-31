#!/usr/bin/env bash
set -eu
UQROOT=/home/pc/research/TAG_condensation
UQCODE=$UQROOT/experiments/uq_component_ablations_20261014/code
UQOUT=$UQROOT/results/uq_component_ablations_20261014
UQPY=/home/pc/miniconda3/envs/tagsam/bin/python
mkdir -p "$UQOUT/logs" "$UQOUT/cells"
exec 9>"$UQOUT/queue.lock"
flock -n 9 || exit 1
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$UQCODE:$UQROOT/experiments/uq_final_evidence_20261012/code:$UQROOT/code/v2_llm_fusion:$UQROOT"
failures=0
printf 'running %s\n' "$(date -Is)" > "$UQOUT/queue.status"
for seed in 271006 271108 271109; do
  for arm in no_gm no_dm fixed_q random_init_learned random_init_fixed payload_controls; do
    for dataset in arxiv sportsfit bookhis; do
      tag=${dataset}_${arm}_o9101_c${seed}
      available=$(df --output=avail "$UQROOT" | tail -n 1 | tr -d ' ')
      if [ "$available" -lt 8388608 ]; then
        printf 'blocked_disk %s %s\n' "$tag" "$(date -Is)" > "$UQOUT/queue.status"
        exit 2
      fi
      printf '%s started %s\n' "$tag" "$(date -Is)" >> "$UQOUT/progress.txt"
      rc=0
      "$UQPY" -u "$UQCODE/run_uq_component_ablations_20261014.py" \
        --dataset "$dataset" --arm "$arm" --construction-seed "$seed" \
        --reference "$UQROOT/experiments/uq_final_evidence_20261012/${dataset}_reference.json" \
        --text-cache "$UQROOT/results/public_baselines_text_cache_20261001/$dataset" \
        --minilm /home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip \
        --output "$UQOUT/cells/$tag" > "$UQOUT/logs/$tag.log" 2>&1 || rc=$?
      printf '%s exit=%s %s\n' "$tag" "$rc" "$(date -Is)" >> "$UQOUT/progress.txt"
      if [ "$rc" -ne 0 ]; then failures=$((failures+1)); fi
    done
  done
done
printf 'complete failures=%s %s\n' "$failures" "$(date -Is)" > "$UQOUT/queue.status"
test "$failures" -eq 0
