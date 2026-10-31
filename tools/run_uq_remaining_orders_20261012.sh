#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/pc/research/TAG_condensation
EXP="$ROOT/experiments/uq_final_evidence_20261012"
OUT="$ROOT/results/uq_final_evidence_20261012"
PY=/home/pc/miniconda3/envs/tagsam/bin/python
WAIT_PID=${1:?Pass the first-order queue PID}

mkdir -p "$OUT/logs"
exec 9>"$OUT/remaining_orders.lock"
flock -n 9 || exit 1
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$EXP/code:$ROOT/code/v2_llm_fusion:$ROOT/tools:$ROOT"

while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done

if [ "$(cat "$OUT/remaining_confirmation.status" 2>/dev/null | cut -d' ' -f1)" != complete ]; then
  printf 'blocked first_order_incomplete %s\n' "$(date -Is)" > "$OUT/remaining_orders.status"
  exit 1
fi

for dataset in bookhis sportsfit; do
  for order in 9102 9103; do
    for seed in 271006 271108 271109; do
      tag=${dataset}_o${order}_c${seed}
      output="$OUT/confirmation/$tag"
      free_kb=$(df --output=avail "$ROOT" | tail -n 1 | tr -d ' ')
      if [ "$free_kb" -lt 8388608 ]; then
        printf 'blocked disk_free_kb=%s %s\n' "$free_kb" "$(date -Is)" > "$OUT/remaining_orders.status"
        exit 1
      fi
      while [ "$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 4096 ]; do
        sleep 60
      done
      if [ -e "$output/results.json" ]; then
        "$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); p=r["protocol"]; assert r["status"] == "complete" and p["dataset"] == sys.argv[2] and p["order_seed"] == int(sys.argv[3]) and p["construction_seed"] == int(sys.argv[4]) and p["arms"] == ["uq_dm_edgefree", "cat_cctp", "gecc_hard"] and p["budget_mode"] == "fixed_rate_current_legal_pool"' \
          "$output/results.json" "$dataset" "$order" "$seed"
        printf '%s reused_complete %s\n' "$tag" "$(date -Is)" >> "$OUT/remaining_orders_status.txt"
        continue
      fi
      if [ -d "$output" ]; then
        printf 'blocked partial_output=%s %s\n' "$output" "$(date -Is)" > "$OUT/remaining_orders.status"
        exit 1
      fi
      "$PY" -u "$EXP/code/run_uq_final_confirmation_20261012.py" \
        --dataset "$dataset" --reference "$EXP/${dataset}_reference.json" \
        --text-cache "$ROOT/results/public_baselines_text_cache_20261001/$dataset" \
        --minilm /home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip \
        --output "$output" --order-seed "$order" --construction-seed "$seed" \
        > "$OUT/logs/confirmation_$tag.log" 2>&1 || {
          printf 'failed %s %s\n' "$tag" "$(date -Is)" > "$OUT/remaining_orders.status"
          exit 1
        }
      printf '%s exit=0 %s\n' "$tag" "$(date -Is)" >> "$OUT/remaining_orders_status.txt"
    done
  done
done
printf 'complete orders=9102,9103 %s\n' "$(date -Is)" > "$OUT/remaining_orders.status"
