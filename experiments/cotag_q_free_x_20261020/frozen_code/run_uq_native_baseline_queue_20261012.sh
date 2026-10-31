#!/usr/bin/env bash
set -u
ROOT=/home/pc/research/TAG_condensation
EXP=$ROOT/experiments/uq_final_evidence_20261012
OUT=$ROOT/results/uq_final_evidence_20261012
PY=/home/pc/miniconda3/envs/tagsam/bin/python
mkdir -p "$OUT/logs"
exec 9>"$OUT/native_baseline_queue.lock"
flock -n 9 || exit 1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$EXP/code:$ROOT/code/v2_llm_fusion:$ROOT/tools:$ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for arm in gecc_soft tagsam_selector_response stad_native_local; do
  smoke="$OUT/smoke/$arm"
  if [ -f "$smoke/audit.json" ]; then
    "$PY" -c 'import json,sys; a=json.load(open(sys.argv[1])); assert a["passed"] and a["arm"] == sys.argv[2]' \
      "$smoke/audit.json" "$arm" || exit 1
    printf 'smoke_%s reused_passed %s\n' "$arm" "$(date -Is)" >> "$OUT/baseline_status.txt"
    continue
  fi
  "$PY" -u "$EXP/code/smoke_uq_native_end_to_end_20261012.py" \
    --arm "$arm" --output "$smoke" > "$OUT/logs/smoke_${arm}_native_v2.log" 2>&1
  rc=$?
  printf 'smoke_%s exit=%s %s\n' "$arm" "$rc" "$(date -Is)" >> "$OUT/baseline_status.txt"
  if [ "$rc" -ne 0 ]; then
    printf 'failed smoke_%s %s\n' "$arm" "$(date -Is)" > "$OUT/native_baseline_queue.status"
    exit "$rc"
  fi
done
failures=0
for dataset in bookhis sportsfit arxiv; do
  for arm in gecc_soft tagsam_selector_response stad_native_local; do
    free_kb=$(df --output=avail "$ROOT" | tail -n 1 | tr -d ' ')
    if [ "$free_kb" -lt 8388608 ]; then
      printf 'blocked disk_free_kb=%s %s\n' "$free_kb" "$(date -Is)" > "$OUT/native_baseline_queue.status"
      exit 1
    fi
    tag=${arm}_${dataset}_o9101_c271006
    category=baselines
    if [ "$arm" = stad_native_local ]; then category=stad; fi
    "$PY" -u "$EXP/code/run_uq_native_baseline_batch_20261012.py" \
      --arm "$arm" --dataset "$dataset" --reference "$EXP/${dataset}_reference.json" \
      --text-cache "$ROOT/results/public_baselines_text_cache_20261001/$dataset" \
      --minilm /home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip \
      --output "$OUT/$category/$tag" --order-seed 9101 --construction-seed 271006 \
      > "$OUT/logs/$tag.log" 2>&1
    rc=$?
    printf '%s exit=%s %s\n' "$tag" "$rc" "$(date -Is)" >> "$OUT/baseline_status.txt"
    if [ "$rc" -ne 0 ]; then failures=$((failures + 1)); fi
  done
done
printf 'complete failures=%s %s\n' "$failures" "$(date -Is)" > "$OUT/native_baseline_queue.status"
