#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/pc/miniconda3/envs/tagsam/bin/python
ROOT=/home/pc/research/TAG_condensation
SNAPSHOT=$ROOT/work_runs/phase12_20260920_snapshot
CACHE_ROOT=$ROOT/results/p5_rate_curve_small_20260919
SEMANTIC_ROOT=$ROOT/results/tgc_fresh_semantic_recursive_20260926
SOURCE=$ROOT/results/tgr_attribution_topology_only_mix10_small_20260927
OUT=$ROOT/results/tgr_attribution_topology_only_mix10_small_eval_20260927
EVAL=$SNAPSHOT/code/v2_llm_fusion/run_gc_sr_stage_eval.py
PYTHONPATH_VALUE=$SNAPSHOT/code/v2_llm_fusion:$SNAPSHOT/code/frozen_v1_2026_09_09

for dataset in cora citeseer; do
  for order in order19611 order19612; do
    for stage in 1 2; do
      candidate="$SOURCE/${dataset}_${order}_s${stage}/tgr_message_memory.pt"
      base="$SEMANTIC_ROOT/${dataset}_${order}_s${stage}/semantic_perm_memory.pt"
      output="$OUT/${dataset}_${order}_s${stage}.json"
      mkdir -p "$OUT"
      if [[ -f "$output" ]]; then
        continue
      fi
      PYTHONPATH="$PYTHONPATH_VALUE" CUDA_VISIBLE_DEVICES=0 \
        "$PYTHON" -u "$EVAL" \
        --dataset "$dataset" --order "$order" --stage "$stage" \
        --base "$base" --candidate "$candidate" --output "$output" \
        --minilm /home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip \
        --cache-root "$CACHE_ROOT" --device cuda:0 \
        --seeds 150 151 152 153 154 155 \
        > "$OUT/${dataset}_${order}_s${stage}.log" 2>&1
    done
  done
done
echo TGR_TOPOLOGY_MIX10_EVAL_20260927_COMPLETE
