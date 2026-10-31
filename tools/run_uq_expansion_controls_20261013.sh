#!/usr/bin/env bash
set -u
ROOT=/home/pc/research/TAG_condensation
EXP=$ROOT/experiments/uq_baseline_expansion_20261013
OUT=$ROOT/results/uq_baseline_expansion_20261013
PY=/home/pc/miniconda3/envs/tagsam/bin/python
mkdir -p "$OUT/logs"
exec 9>"$OUT/controls.lock"
flock -n 9 || exit 1
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$EXP/code:$ROOT/experiments/uq_final_evidence_20261012/code:$ROOT/code/v2_llm_fusion:$ROOT/tools:$ROOT"
failures=0
for dataset in cora citeseer bookhis sportsfit arxiv; do
  reference=()
  cache="$OUT/text_cache/$dataset"
  if [ "$dataset" != cora ] && [ "$dataset" != citeseer ]; then
    reference=(--reference "$ROOT/experiments/uq_final_evidence_20261012/${dataset}_reference.json")
    cache="$ROOT/results/public_baselines_text_cache_20261001/$dataset"
  fi
  arms=(random_coreset kcenter_coreset cluster_medoid herding_coreset puma_tag)
  if [ "$dataset" = cora ] || [ "$dataset" = citeseer ]; then
    arms+=(uq_dm_edgefree cat_cctp gecc_hard)
  fi
  for arm in "${arms[@]}"; do
    tag=${dataset}_${arm}_o9101_c271006
    "$PY" -u "$EXP/code/run_uq_baseline_expansion_20261013.py" \
      --dataset "$dataset" "${reference[@]}" \
      --text-cache "$cache" \
      --minilm /home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip \
      --output "$OUT/cells/$tag" --arms "$arm" > "$OUT/logs/$tag.log" 2>&1
    rc=$?
    printf '%s exit=%s %s\n' "$tag" "$rc" "$(date -Is)" >> "$OUT/status.txt"
    if [ "$rc" -ne 0 ]; then failures=$((failures+1)); fi
  done
done
printf 'complete failures=%s %s\n' "$failures" "$(date -Is)" > "$OUT/controls.status"
