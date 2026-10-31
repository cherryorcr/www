#!/usr/bin/env bash
set -u
ROOT=/home/pc/research/TAG_condensation
EXP=$ROOT/experiments/uq_baseline_expansion_20261013
OUT=$ROOT/results/uq_baseline_expansion_20261013
PY=/home/pc/miniconda3/envs/tagsam/bin/python
mkdir -p "$OUT/logs"
exec 9>"$OUT/native.lock"
flock -n 9 || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$EXP/code_v2:$ROOT/experiments/uq_final_evidence_20261012/code:$ROOT/code/v2_llm_fusion:$ROOT/tools:$ROOT"
failures=0
run_cell() {
  local dataset=$1 arm=$2
  local tag=${dataset}_${arm}_o9101_c271006_v2
  local reference=()
  local cache="$OUT/text_cache/$dataset"
  if [ "$dataset" != cora ] && [ "$dataset" != citeseer ]; then
    reference=(--reference "$ROOT/experiments/uq_final_evidence_20261012/${dataset}_reference.json")
    cache="$ROOT/results/public_baselines_text_cache_20261001/$dataset"
  fi
  local path="$OUT/cells/$tag/results.json"
  if [ -f "$path" ] && "$PY" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])).get("status") != "complete")' "$path"; then
    printf '%s reused_complete %s\n' "$tag" "$(date -Is)" >> "$OUT/native_status.txt"
    return
  fi
  local available
  available=$(df --output=avail -B1 "$OUT" | tail -1)
  if [ "$available" -lt 5368709120 ]; then
    printf '%s blocked_disk_under_5GiB %s\n' "$tag" "$(date -Is)" >> "$OUT/native_status.txt"
    failures=$((failures+1))
    return
  fi
  "$PY" -u "$EXP/code_v2/run_uq_baseline_expansion_20261013.py" \
    --dataset "$dataset" "${reference[@]}" --text-cache "$cache" \
    --minilm /home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip \
    --output "$OUT/cells/$tag" --arms "$arm" > "$OUT/logs/$tag.log" 2>&1
  local rc=$?
  printf '%s exit=%s %s\n' "$tag" "$rc" "$(date -Is)" >> "$OUT/native_status.txt"
  if [ "$rc" -ne 0 ]; then failures=$((failures+1)); fi
}
export CUDA_VISIBLE_DEVICES=0
for dataset in cora citeseer; do
  for arm in uq_dm_edgefree cat_cctp gecc_hard; do
    run_cell "$dataset" "$arm"
  done
done
# The original coreset queue owns GPU 1 until its lock is released.
printf 'waiting_for_controls %s\n' "$(date -Is)" >> "$OUT/native_status.txt"
exec 8>"$OUT/controls.lock"
flock 8
export CUDA_VISIBLE_DEVICES=1
for arm in gcond_x_core sfgc_core; do
  for dataset in cora citeseer bookhis sportsfit arxiv; do
    run_cell "$dataset" "$arm"
  done
done
printf 'complete failures=%s %s\n' "$failures" "$(date -Is)" > "$OUT/native.status"
