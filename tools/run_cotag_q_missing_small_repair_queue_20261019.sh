#!/usr/bin/env bash
set -u
set -o pipefail

arm=${1:?expected tagsam_native_bounded or stad_native_local}
gpu=${2:?expected physical GPU index}
case "$arm" in
  tagsam_native_bounded|stad_native_local) ;;
  *) exit 2 ;;
esac
root=/home/pc/research/TAG_condensation
python=/home/pc/miniconda3/envs/tagsam/bin/python
original=$root/results/cotag_q_missing_baselines_20261019
output=$root/results/cotag_q_missing_baselines_20261019_repair_v2
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$root" || exit 1
mkdir -p "$output/logs"
failed=0
for dataset in cora citeseer; do
  cell=$output/$dataset/$arm
  if [ -f "$cell/results.json" ] && "$python" -c 'import json,sys;sys.exit(json.load(open(sys.argv[1])).get("status")!="complete")' "$cell/results.json"; then
    printf '%s SKIP %s %s\n' "$(date -Is)" "$dataset" "$arm"
    continue
  fi
  resume=()
  if [ "$arm" = tagsam_native_bounded ]; then
    resume=(--resume-from "$original/$dataset/$arm")
  fi
  stamp=$(date -u +%Y%m%dT%H%M%S)
  log=$output/logs/${dataset}_${arm}_${stamp}.log
  printf '%s START %s %s gpu=%s log=%s\n' "$(date -Is)" "$dataset" "$arm" "$gpu" "$log"
  "$python" -u tools/run_cotag_q_missing_small_baselines_20261019.py \
    --dataset "$dataset" --arm "$arm" --output "$cell" "${resume[@]}" \
    --text-cache "$original/text_cache_$dataset" --device cuda:0 >"$log" 2>&1
  code=$?
  printf '%s END %s %s exit=%s\n' "$(date -Is)" "$dataset" "$arm" "$code"
  if [ "$code" -ne 0 ]; then failed=$((failed + 1)); fi
done
printf '%s FINISHED arm=%s failed=%s\n' "$(date -Is)" "$arm" "$failed"
test "$failed" -eq 0
