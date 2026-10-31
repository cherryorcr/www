#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/pc/research/TAG_condensation
EXP="$ROOT/experiments/uq_q_coverage_20261012"
OUT="$ROOT/results/uq_q_coverage_20261012"
PY=/home/pc/miniconda3/envs/tagsam/bin/python
GPU_INDEX="${1:-1}"
shift "$(( $# > 0 ? 1 : 0 ))"
DATASETS=("$@")
if [ "${#DATASETS[@]}" -eq 0 ]; then
  DATASETS=(wikics sportsfit bookhis)
fi
mkdir -p "$OUT"
exec 9>"$OUT/launcher.lock"
flock -n 9 || exit 1
export PYTHONPATH="$EXP/code:$ROOT/code/v2_llm_fusion:$ROOT/code/frozen_v1_2026_09_09"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

# GPU1 confirmation wrappers remain alive between children. GPU0's native
# baseline queue is independent and need not delay this GPU1 batch.
if [ "$GPU_INDEX" = 1 ]; then
  printf 'waiting GPU1 confirmation %s\n' "$(date -Is)" > "$OUT/launcher.status"
  while pgrep -af 'uq_final_evidence_20261012/(code/)?(run_uq_remaining_confirmation_batch|run_uq_remaining_orders|run_uq_final_confirmation)' >/dev/null; do
    sleep 60
  done
fi
for dataset in "${DATASETS[@]}"; do
  if [ "$(df --output=avail "$ROOT" | tail -n 1 | tr -d ' ')" -lt 8388608 ]; then
    printf 'blocked disk %s\n' "$(date -Is)" > "$OUT/launcher.status"
    exit 1
  fi
  while [ "$(nvidia-smi --id="$GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 4096 ]; do
    sleep 60
  done
  tag="${dataset}_o9101_c271108"
  printf 'running %s %s\n' "$tag" "$(date -Is)" > "$OUT/launcher.status"
  "$PY" -u "$EXP/code/run_uq_coverage_confirmation_20261012.py" \
    --dataset "$dataset" --reference "$ROOT/results/unified_q_inloop_cat_20261007/$dataset/protocol.json" \
    --text-cache "$ROOT/results/public_baselines_text_cache_20261001/$dataset" \
    --minilm /home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip \
    --output "$OUT/$tag" --order-seed 9101 --construction-seed 271108 \
    --device "cuda:$GPU_INDEX" --q-steps 30 --sinkhorn-max-steps 4096 \
    --student-steps 2048 --student-seeds 150 151 152 \
    > "$OUT/$tag.log" 2>&1 || {
      printf 'failed %s %s\n' "$tag" "$(date -Is)" > "$OUT/launcher.status"
      exit 1
    }
  printf '%s complete %s\n' "$tag" "$(date -Is)" >> "$OUT/launcher_progress.txt"
  "$PY" "$EXP/code/summarize_uq_coverage_20261012.py" --root "$OUT" \
    --output "$OUT/report_${dataset}_$(date +%H%M%S)" --no-plots
done
printf 'complete %s\n' "$(date -Is)" > "$OUT/launcher.status"
