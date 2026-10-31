#!/usr/bin/env bash
set -u

ROOT=/home/pc/research/TAG_condensation
TOOLS=$ROOT/tools
CODE=$ROOT/experiments/uq_q_seed_attribution_20261010/code
PY=/home/pc/miniconda3/envs/tagsam/bin/python
OUT=/dev/shm/baseline_seed_attribution_20261010
mkdir -p "$OUT"

run_one() {
  local gpu="$1" dataset="$2" order="$3"
  local output="$OUT/${dataset}_o${order}.json"
  local log="$OUT/${dataset}_o${order}.log"
  if [[ -s "$output" ]]; then
    echo "reuse $dataset $order" >> "$OUT/queue.log"
    return 0
  fi
  echo "start gpu=$gpu dataset=$dataset order=$order" >> "$OUT/queue.log"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$CODE:$TOOLS:$ROOT/code/v2_llm_fusion" \
    "$PY" -u "$TOOLS/run_baseline_seed_attribution_20261010.py" \
    --dataset "$dataset" --order "$order" --device cuda:0 \
    --output "$output" > "$log" 2>&1
  local rc=$?
  echo "finish gpu=$gpu dataset=$dataset order=$order rc=$rc" >> "$OUT/queue.log"
  return "$rc"
}

lane0() {
  run_one 0 wikics 9101 && run_one 0 wikics 9102 && run_one 0 wikics 9103 && \
  run_one 0 sportsfit 9101 && run_one 0 sportsfit 9102
}

lane1() {
  run_one 1 bookhis 9101 && run_one 1 bookhis 9102 && run_one 1 bookhis 9103 && \
  run_one 1 sportsfit 9103
}

echo "started $(date -Is)" > "$OUT/queue.status"
lane0 > "$OUT/lane0.log" 2>&1 & pid0=$!
lane1 > "$OUT/lane1.log" 2>&1 & pid1=$!
wait "$pid0"; rc0=$?
wait "$pid1"; rc1=$?
printf 'done %s rc0=%s rc1=%s\n' "$(date -Is)" "$rc0" "$rc1" > "$OUT/queue.status"
exit $((rc0 != 0 || rc1 != 0))
