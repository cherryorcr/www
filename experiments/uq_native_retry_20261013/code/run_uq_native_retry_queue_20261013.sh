#!/usr/bin/env bash
set -u
UQROOT=/home/pc/research/TAG_condensation
UQEXP=$UQROOT/experiments/uq_final_evidence_20261012
UQRETRY=$UQROOT/experiments/uq_native_retry_20261013
UQOUT=$UQROOT/results/uq_final_evidence_20261012
UQPY=/home/pc/miniconda3/envs/tagsam/bin/python
mkdir -p "$UQOUT/logs"
exec 9>"$UQOUT/native_retry_queue.lock"
flock -n 9 || exit 1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$UQRETRY/code:$UQROOT/code/v2_llm_fusion:$UQROOT/tools:$UQROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$UQPY" -c 'import json,sys; a=json.load(open(sys.argv[1])); assert len(a["records"]) == 13 and all(r["evaluation_nodes_used"] is False for r in a["records"])' \
  "$UQRETRY/legal_sparse_class_audit.json" || exit 1
"$UQPY" "$UQRETRY/code/test_uq_native_retry_20261013.py" > "$UQOUT/logs/native_retry_regression_20261013.log" 2>&1 || exit 1
failures=0
run_cell() {
  local dataset=$1 arm=$2 category=baselines tag original free_kb rc
  if [ "$arm" = stad_native_local ]; then category=stad; fi
  original=$UQOUT/$category/${arm}_${dataset}_o9101_c271006/results.json
  if "$UQPY" -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); sys.exit(0 if p.exists() and json.load(open(p))["status"] == "complete" else 1)' "$original"; then
    printf '%s_%s original_complete_reused %s\n' "$arm" "$dataset" "$(date -Is)" >> "$UQOUT/native_retry_status.txt"
    return 0
  fi
  free_kb=$(df --output=avail "$UQROOT" | tail -n 1 | tr -d ' ')
  if [ "$free_kb" -lt 8388608 ]; then
    printf 'blocked disk_free_kb=%s %s\n' "$free_kb" "$(date -Is)" > "$UQOUT/native_retry_queue.status"
    exit 1
  fi
  tag=${arm}_${dataset}_o9101_c271006_retry20261013
  printf '%s started %s\n' "$tag" "$(date -Is)" >> "$UQOUT/native_retry_status.txt"
  "$UQPY" -u "$UQRETRY/code/run_uq_native_baseline_batch_20261012.py" \
    --arm "$arm" --dataset "$dataset" --reference "$UQEXP/${dataset}_reference.json" \
    --text-cache "$UQROOT/results/public_baselines_text_cache_20261001/$dataset" \
    --minilm /home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip \
    --output "$UQOUT/$category/$tag" --order-seed 9101 --construction-seed 271006 \
    > "$UQOUT/logs/$tag.log" 2>&1
  rc=$?
  printf '%s exit=%s %s\n' "$tag" "$rc" "$(date -Is)" >> "$UQOUT/native_retry_status.txt"
  if [ "$rc" -ne 0 ]; then failures=$((failures + 1)); fi
}
# The two BookHis failures are known; their measured peaks fit beside the
# currently CPU-bound GECC job. Later cells wait for the original queue.
printf 'running bookhis_retries %s\n' "$(date -Is)" > "$UQOUT/native_retry_queue.status"
run_cell bookhis tagsam_selector_response
run_cell bookhis stad_native_local
printf 'waiting original_native_queue %s\n' "$(date -Is)" > "$UQOUT/native_retry_queue.status"
while pgrep -f '^bash /home/pc/research/TAG_condensation/experiments/uq_final_evidence_20261012/code/run_uq_native_baseline_queue_20261012.sh$' > /dev/null; do
  sleep 30
done
printf 'running remaining_retries %s\n' "$(date -Is)" > "$UQOUT/native_retry_queue.status"
for dataset in sportsfit arxiv; do
  for arm in tagsam_selector_response stad_native_local; do
    run_cell "$dataset" "$arm"
  done
done
printf 'complete failures=%s %s\n' "$failures" "$(date -Is)" > "$UQOUT/native_retry_queue.status"
