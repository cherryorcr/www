#!/usr/bin/env bash
set -u

ROOT=/home/pc/research/TAG_condensation
PY=/home/pc/miniconda3/envs/tagsam/bin/python
OUT=$ROOT/results/cotag_q_step_sensitivity_20261016
LOGS=$OUT/logs
GPU=${1:?GPU index}
QUEUE=${2:?queue index}
mkdir -p "$LOGS"

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT/experiments/uq_final_evidence_20261012/code:$ROOT/code/v2_llm_fusion:$ROOT"

run_cell() {
  dataset=$1
  steps=$2
  seed=$3
  output="$OUT/${dataset}/steps${steps}/c${seed}"
  log="$LOGS/q${QUEUE}_${dataset}_steps${steps}_c${seed}.log"
  if [ -f "$output/results.json" ] && grep -q '"status": "complete"' "$output/results.json"; then
    printf 'skip complete %s %s %s\n' "$dataset" "$steps" "$seed"
    return 0
  fi
  available=$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')
  if [ "$available" -lt 8388608 ]; then
    printf 'pause: less than 8 GiB free before %s %s %s\n' "$dataset" "$steps" "$seed" >&2
    return 3
  fi
  printf 'start %s %s %s %s\n' "$(date -Is)" "$dataset" "$steps" "$seed" >> "$LOGS/status"
  "$PY" -u "$ROOT/tools/run_cotag_q_step_sensitivity_20261016.py" \
    --dataset "$dataset" \
    --steps "$steps" \
    --construction-seed "$seed" \
    --reference "$ROOT/experiments/uq_final_evidence_20261012/${dataset}_reference.json" \
    --text-cache "$ROOT/results/public_baselines_text_cache_20261001/$dataset" \
    --minilm /home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip \
    --output "$output" \
    --device cuda:0 > "$log" 2>&1
  rc=$?
  printf 'finish %s %s %s %s rc=%s\n' "$(date -Is)" "$dataset" "$steps" "$seed" "$rc" >> "$LOGS/status"
  return "$rc"
}

printf 'queue %s gpu %s started %s\n' "$QUEUE" "$GPU" "$(date -Is)" >> "$LOGS/status"
failures=0

# Keep the two workers on disjoint cells.  A cell is always reconstructed from
# stage zero, so changing the order below cannot change its protocol.
if [ "$QUEUE" = "0" ]; then
  cells=(
    "arxiv 10 271006" "arxiv 60 271006" "arxiv 100 271006"
    "sportsfit 10 271006" "sportsfit 60 271006" "sportsfit 100 271006"
    "bookhis 10 271006" "bookhis 60 271006" "bookhis 100 271006"
    "arxiv 10 271108" "arxiv 60 271108" "arxiv 100 271108"
    "sportsfit 10 271108" "sportsfit 60 271108" "sportsfit 100 271108"
  )
else
  cells=(
    "arxiv 10 271109" "arxiv 60 271109" "arxiv 100 271109"
    "sportsfit 10 271109" "sportsfit 60 271109" "sportsfit 100 271109"
    "bookhis 10 271109" "bookhis 60 271109" "bookhis 100 271109"
    "bookhis 10 271108" "bookhis 60 271108" "bookhis 100 271108"
  )
fi

for cell in "${cells[@]}"; do
  read -r dataset steps seed <<< "$cell"
  run_cell "$dataset" "$steps" "$seed" || failures=$((failures + 1))
done

printf 'queue %s gpu %s complete failures=%s %s\n' "$QUEUE" "$GPU" "$failures" "$(date -Is)" >> "$LOGS/status"
exit "$failures"
