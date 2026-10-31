#!/usr/bin/env bash
set -u
set -o pipefail

lane=${1:?expected native or references}
gpu=${2:?expected physical GPU index}
root=/home/pc/research/TAG_condensation
python=/home/pc/miniconda3/envs/tagsam/bin/python
output=$root/results/cotag_q_missing_baselines_20261019
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$root" || exit 1
mkdir -p "$output/logs"

case "$lane" in
  native) arms=(tagsam_native_bounded) ;;
  references) arms=(dance_adapter sr_gm_adapter stad_adapter stad_native_local) ;;
  *) printf 'Unknown lane: %s\n' "$lane" >&2; exit 2 ;;
esac

failed=0
for arm in "${arms[@]}"; do
  for dataset in cora citeseer; do
    cell=$output/$dataset/$arm
    if [ -f "$cell/results.json" ] && "$python" -c 'import json,sys;sys.exit(json.load(open(sys.argv[1])).get("status")!="complete")' "$cell/results.json"; then
      printf '%s SKIP %s %s\n' "$(date -Is)" "$dataset" "$arm"
      continue
    fi
    stamp=$(date -u +%Y%m%dT%H%M%S)
    log=$output/logs/${dataset}_${arm}_${stamp}.log
    printf '%s START %s %s gpu=%s log=%s\n' "$(date -Is)" "$dataset" "$arm" "$gpu" "$log"
    "$python" -u tools/run_cotag_q_missing_small_baselines_20261019.py \
      --dataset "$dataset" --arm "$arm" --output "$cell" \
      --text-cache "$output/text_cache_$dataset" --device cuda:0 >"$log" 2>&1
    code=$?
    printf '%s END %s %s exit=%s\n' "$(date -Is)" "$dataset" "$arm" "$code"
    if [ "$code" -ne 0 ]; then failed=$((failed + 1)); fi
  done
done
printf '%s FINISHED lane=%s failed=%s\n' "$(date -Is)" "$lane" "$failed"
test "$failed" -eq 0
