#!/usr/bin/env bash
set -u

ROOT=/home/pc/research/TAG_condensation
EXP=$ROOT/experiments/uq_baseline_expansion_20261013
OUT=$ROOT/results/cotag_q_small_9cell_20261019
PY=/home/pc/miniconda3/envs/tagsam/bin/python
RUNNER=$EXP/code_v2/run_uq_baseline_expansion_20261013.py
MINILM=/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip

mkdir -p "$OUT/logs"
exec 9>"$OUT/queue.lock"
flock -n 9 || exit 1

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$EXP/code_v2:$ROOT/experiments/uq_final_evidence_20261012/code:$ROOT/code/v2_llm_fusion:$ROOT/tools:$ROOT"

ARMS=(uq_dm_edgefree cat_cctp gecc_hard)
ORDER_SEEDS=(9101 9102 9103)
CONSTRUCTION_SEEDS=(271006 271108 271109)

run_cell() {
  local dataset=$1
  local order_seed=$2
  local construction_seed=$3
  local tag="${dataset}_o${order_seed}_c${construction_seed}"
  local output="$OUT/cells/$tag"
  local log="$OUT/logs/$tag.log"
  local result="$output/results.json"
  local cache="$ROOT/results/uq_baseline_expansion_20261013/text_cache/$dataset"

  if [ -f "$result" ] && "$PY" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])).get("status") != "complete")' "$result"; then
    printf '%s reused_complete %s\n' "$tag" "$(date -Is)" >> "$OUT/status.log"
    return 0
  fi

  local available
  available=$(df --output=avail -B1 "$OUT" | tail -1)
  if [ "$available" -lt 5368709120 ]; then
    printf '%s blocked_disk_under_5GiB %s\n' "$tag" "$(date -Is)" >> "$OUT/status.log"
    return 1
  fi

  "$PY" -u "$RUNNER" \
    --dataset "$dataset" \
    --text-cache "$cache" \
    --minilm "$MINILM" \
    --output "$output" \
    --order-seed "$order_seed" \
    --construction-seed "$construction_seed" \
    --arms "${ARMS[@]}" \
    > "$log" 2>&1
  local rc=$?
  printf '%s exit=%s %s\n' "$tag" "$rc" "$(date -Is)" >> "$OUT/status.log"
  return "$rc"
}

failures=0
for dataset in cora citeseer; do
  for order_seed in "${ORDER_SEEDS[@]}"; do
    for construction_seed in "${CONSTRUCTION_SEEDS[@]}"; do
      run_cell "$dataset" "$order_seed" "$construction_seed" || failures=$((failures + 1))
    done
  done
done

printf 'complete failures=%s %s\n' "$failures" "$(date -Is)" > "$OUT/status"
