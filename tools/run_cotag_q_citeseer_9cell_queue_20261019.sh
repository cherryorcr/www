#!/usr/bin/env bash
set -u

ROOT=/home/pc/research/TAG_condensation
EXP=$ROOT/experiments/uq_baseline_expansion_20261013
OUT=$ROOT/results/cotag_q_small_9cell_20261019
PY=/home/pc/miniconda3/envs/tagsam/bin/python
RUNNER=$EXP/code_v2/run_uq_baseline_expansion_20261013.py
MINILM=/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip

mkdir -p "$OUT/logs"
exec 9>"$OUT/citeseer.queue.lock"
flock -n 9 || exit 1

export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$EXP/code_v2:$ROOT/experiments/uq_final_evidence_20261012/code:$ROOT/code/v2_llm_fusion:$ROOT/tools:$ROOT"

for order_seed in 9101 9102 9103; do
  for construction_seed in 271006 271108 271109; do
    tag="citeseer_o${order_seed}_c${construction_seed}"
    output="$OUT/cells/$tag"
    result="$output/results.json"
    if [ -f "$result" ] && "$PY" -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])).get("status") != "complete")' "$result"; then
      printf '%s reused_complete %s\n' "$tag" "$(date -Is)" >> "$OUT/status.log"
      continue
    fi
    available=$(df --output=avail -B1 "$OUT" | tail -1)
    if [ "$available" -lt 5368709120 ]; then
      printf '%s blocked_disk_under_5GiB %s\n' "$tag" "$(date -Is)" >> "$OUT/status.log"
      exit 1
    fi
    "$PY" -u "$RUNNER" \
      --dataset citeseer \
      --text-cache "$ROOT/results/uq_baseline_expansion_20261013/text_cache/citeseer" \
      --minilm "$MINILM" \
      --output "$output" \
      --order-seed "$order_seed" \
      --construction-seed "$construction_seed" \
      --arms uq_dm_edgefree cat_cctp gecc_hard \
      > "$OUT/logs/$tag.log" 2>&1
    rc=$?
    printf '%s exit=%s %s\n' "$tag" "$rc" "$(date -Is)" >> "$OUT/status.log"
    [ "$rc" -eq 0 ] || exit "$rc"
  done
done
