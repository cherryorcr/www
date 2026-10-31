#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/pc/miniconda3/envs/tagsam/bin/python
ROOT=/home/pc/research/TAG_condensation
WORK=$ROOT/work_runs/phase12_20260920_snapshot
SOURCE=$ROOT/results/tgc_fresh_r10_source_20260926
OUT=$ROOT/results/tgc_fresh_semantic_recursive_20260926
CACHE=$ROOT/results/p5_rate_curve_small_20260919
MINILM=/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip
PYTHONPATH_VALUE=$ROOT/code/v2_llm_fusion:$WORK/code/v2_llm_fusion

mkdir -p "$OUT"
for order in order19611 order19612; do
  for stage in 1 2; do
    PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON" -u \
      "$WORK/code/v2_llm_fusion/run_tgc_semantic_recursive_confirmation_20260926.py" \
      --dataset "$1" --order "$order" --stage "$stage" \
      --source-root "$SOURCE" --parent-root "$OUT" --output-root "$OUT" \
      --cache-root "$CACHE" --minilm "$MINILM" --device cuda:0 \
      --parent-arm semantic_perm \
      > "$OUT/${1}_${order}_s${stage}.log" 2>&1
  done
done
