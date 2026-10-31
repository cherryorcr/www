#!/usr/bin/env bash
set -u
UQROOT=/home/pc/research/TAG_condensation
UQORIGINAL=$UQROOT/experiments/uq_final_evidence_20261012
UQRETRY=$UQROOT/experiments/uq_native_retry_20261013
UQOUT=$UQROOT/results/uq_final_evidence_20261012
UQPY=/home/pc/miniconda3/envs/tagsam/bin/python
UQGECCPID=${1:?existing SportsFit GECC PID is required}
mkdir -p "$UQOUT/logs"
exec 9>"$UQOUT/native_parallel_references.lock"
flock -n 9 || exit 1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$UQRETRY/code:$UQROOT/code/v2_llm_fusion:$UQROOT/tools:$UQROOT"
failures=0
run_cell() {
  local dataset=$1 arm=$2 category=baselines tag path rc code_dir=$UQRETRY/code
  if [ "$arm" = stad_native_local ]; then category=stad; fi
  tag=${arm}_${dataset}_o9101_c271006_retry20261013
  if [ "$arm" = gecc_soft ]; then
    tag=${arm}_${dataset}_o9101_c271006
    code_dir=$UQORIGINAL/code
  fi
  for path in "$UQOUT/$category/${arm}_${dataset}_o9101_c271006/results.json" \
              "$UQOUT/$category/${arm}_${dataset}_o9101_c271006_retry20261013/results.json"; do
    if "$UQPY" -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); sys.exit(0 if p.exists() and json.load(open(p))["status"] == "complete" else 1)' "$path"; then
      printf '%s_%s complete_reused %s\n' "$arm" "$dataset" "$(date -Is)" >> "$UQOUT/native_parallel_status.txt"
      return 0
    fi
  done
  if pgrep -f "run_uq_native_baseline_batch_20261012.py.*--output $UQOUT/$category/${arm}_${dataset}_o9101_c271006(_retry20261013)?( |$)" > /dev/null; then
    printf '%s active_not_duplicated %s\n' "$tag" "$(date -Is)" >> "$UQOUT/native_parallel_status.txt"
    failures=$((failures+1))
    return 1
  fi
  local free_kb
  free_kb=$(df --output=avail "$UQROOT" | tail -n 1 | tr -d ' ')
  if [ "$free_kb" -lt 8388608 ]; then
    printf '%s blocked_disk_under_8GiB %s\n' "$tag" "$(date -Is)" >> "$UQOUT/native_parallel_status.txt"
    failures=$((failures+1))
    return 1
  fi
  printf '%s started %s\n' "$tag" "$(date -Is)" >> "$UQOUT/native_parallel_status.txt"
  PYTHONPATH="$code_dir:$UQROOT/code/v2_llm_fusion:$UQROOT/tools:$UQROOT" \
    "$UQPY" -u "$code_dir/run_uq_native_baseline_batch_20261012.py" \
      --arm "$arm" --dataset "$dataset" --reference "$UQORIGINAL/${dataset}_reference.json" \
      --text-cache "$UQROOT/results/public_baselines_text_cache_20261001/$dataset" \
      --minilm /home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip \
      --output "$UQOUT/$category/$tag" --order-seed 9101 --construction-seed 271006 \
      > "$UQOUT/logs/$tag.log" 2>&1
  rc=$?
  printf '%s exit=%s %s\n' "$tag" "$rc" "$(date -Is)" >> "$UQOUT/native_parallel_status.txt"
  if [ "$rc" -ne 0 ]; then failures=$((failures+1)); fi
}
gecc_worker() {
  printf 'waiting_existing_gecc_pid=%s %s\n' "$UQGECCPID" "$(date -Is)" > "$UQOUT/native_parallel_gecc.status"
  while [ -r "/proc/$UQGECCPID/cmdline" ] &&
        tr '\0' ' ' < "/proc/$UQGECCPID/cmdline" | grep -q 'run_uq_native_baseline_batch_20261012.py.*--arm gecc_soft.*--dataset sportsfit'; do
    sleep 30
  done
  if ! "$UQPY" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "complete"' \
      "$UQOUT/baselines/gecc_soft_sportsfit_o9101_c271006/results.json"; then
    printf 'blocked_previous_gecc_incomplete %s\n' "$(date -Is)" > "$UQOUT/native_parallel_gecc.status"
    return 1
  fi
  printf 'running_gecc_arxiv %s\n' "$(date -Is)" > "$UQOUT/native_parallel_gecc.status"
  if run_cell arxiv gecc_soft; then
    printf 'complete failures=%s %s\n' "$failures" "$(date -Is)" > "$UQOUT/native_parallel_gecc.status"
  else
    printf 'failed failures=%s %s\n' "$failures" "$(date -Is)" > "$UQOUT/native_parallel_gecc.status"
    return 1
  fi
}
gecc_worker &
gecc_watcher=$!
for dataset in sportsfit arxiv; do
  for arm in tagsam_selector_response stad_native_local; do
    run_cell "$dataset" "$arm"
  done
done
printf 'references_complete failures=%s %s\n' "$failures" "$(date -Is)" > "$UQOUT/native_parallel_references.status"
wait "$gecc_watcher"
