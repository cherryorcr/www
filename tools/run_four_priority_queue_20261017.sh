#!/usr/bin/env bash
set -u

ROOT=/home/pc/research/TAG_condensation
PY=/home/pc/miniconda3/envs/tagsam/bin/python
SCRIPT=$ROOT/tools/run_four_priority_experiments_20261017.py
OUT=$ROOT/results/uq_four_priority_20261017
GPU=${1:?GPU index}
TRACK=${2:?track: fresh|text_budget|resource}
QUEUE=${3:?queue label}

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$OUT/logs"

reference() {
  echo "$ROOT/results/uq_final_evidence_20261012/confirmation/$1_o9101_c271006/results.json"
}

text_cache() {
  case "$1" in
    arxiv|sportsfit) echo "$ROOT/results/tgc_large_full_history_20260926/${1}_base4" ;;
    bookhis) echo "$ROOT/results/public_baselines_text_cache_20261001/bookhis" ;;
  esac
}

run_fresh() {
  dataset=$1
  seed=$2
  output="$OUT/fresh/$dataset/o9101_c${seed}"
  log="$OUT/logs/${QUEUE}_fresh_${dataset}_c${seed}.log"
  if [ -f "$output/results.json" ] && grep -q '"status": "complete"' "$output/results.json"; then
    echo "skip complete fresh $dataset $seed" >> "$OUT/logs/queue_status"
    return 0
  fi
  "$PY" -u "$SCRIPT" --mode fresh --dataset "$dataset" \
    --reference "$(reference "$dataset")" \
    --text-cache "$(text_cache "$dataset")" \
    --minilm /home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip \
    --output "$output" --construction-seed "$seed" --device cuda:0 \
    > "$log" 2>&1
  rc=$?
  echo "fresh $dataset $seed rc=$rc" >> "$OUT/logs/queue_status"
  return "$rc"
}

run_text() {
  dataset=$1
  log="$OUT/logs/${QUEUE}_text_${dataset}.log"
  if [ -f "$OUT/text_budget/$dataset/results.json" ] && grep -q '"status": "complete"' "$OUT/text_budget/$dataset/results.json"; then
    echo "skip complete text $dataset" >> "$OUT/logs/queue_status"
    return 0
  fi
  "$PY" -u "$SCRIPT" --mode text_budget --dataset "$dataset" \
    --reference "$(reference "$dataset")" \
    --text-cache "$(text_cache "$dataset")" \
    --minilm /home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip \
    --output "$OUT/text_budget/$dataset" --device cuda:0 \
    > "$log" 2>&1
  rc=$?
  echo "text $dataset rc=$rc" >> "$OUT/logs/queue_status"
  return "$rc"
}

run_resource() {
  dataset=$1
  log="$OUT/logs/${QUEUE}_resource_${dataset}.log"
  if [ -f "$OUT/resource/$dataset/results.json" ] && grep -q '"status": "complete"' "$OUT/resource/$dataset/results.json"; then
    echo "skip complete resource $dataset" >> "$OUT/logs/queue_status"
    return 0
  fi
  "$PY" -u "$SCRIPT" --mode resource --dataset "$dataset" \
    --reference "$(reference "$dataset")" \
    --text-cache "$(text_cache "$dataset")" \
    --minilm /home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip \
    --output "$OUT/resource/$dataset" --device cuda:0 \
    > "$log" 2>&1
  rc=$?
  echo "resource $dataset rc=$rc" >> "$OUT/logs/queue_status"
  return "$rc"
}

failures=0
case "$TRACK" in
  fresh)
    # Keep each queue on one GPU and skip cells already marked complete.
    if [ "$QUEUE" = "fresh0" ]; then
      cells=("arxiv 271006" "arxiv 271108" "arxiv 271109")
    elif [ "$QUEUE" = "fresh1" ]; then
      cells=("sportsfit 271006" "sportsfit 271108" "sportsfit 271109")
    else
      cells=("bookhis 271108" "bookhis 271109")
    fi
    for cell in "${cells[@]}"; do
      read -r dataset seed <<< "$cell"
      run_fresh "$dataset" "$seed" || failures=$((failures + 1))
    done
    ;;
  text_budget)
    for dataset in "$QUEUE"; do
      run_text "$dataset" || failures=$((failures + 1))
    done
    ;;
  resource)
    for dataset in arxiv sportsfit bookhis; do
      run_resource "$dataset" || failures=$((failures + 1))
    done
    ;;
  *)
    echo "unknown track: $TRACK" >&2
    exit 2
    ;;
esac
echo "queue=$QUEUE track=$TRACK gpu=$GPU failures=$failures finished=$(date -Is)" >> "$OUT/logs/queue_status"
exit "$failures"
