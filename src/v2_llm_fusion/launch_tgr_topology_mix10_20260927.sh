#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/pc/miniconda3/envs/tagsam/bin/python
ROOT=/home/pc/research/TAG_condensation
SNAPSHOT=$ROOT/work_runs/phase12_20260920_snapshot
DATA_ROOT=/home/pc/research/GraphCLIP/processed_data
CACHE_ROOT=$ROOT/results/p5_rate_curve_small_20260919
C0_ROOT=$ROOT/results/tgc_fresh_r10_source_20260926
SEMANTIC_ROOT=$ROOT/results/tgc_fresh_semantic_recursive_20260926
RUNNER=$SNAPSHOT/code/v2_llm_fusion/run_teacher_guided_message_revision.py
PYTHONPATH_VALUE=$SNAPSHOT/code/v2_llm_fusion:$SNAPSHOT/code/frozen_v1_2026_09_09
OUT=$ROOT/results/tgr_attribution_topology_only_mix10_small_20260927

for dataset in cora citeseer; do
  for order in order19611 order19612; do
    for stage in 1 2; do
      cell="$OUT/${dataset}_${order}_s${stage}"
      mkdir -p "$OUT/logs"
      if [[ -f "$cell/audit.json" ]]; then
        continue
      fi
      if [[ -e "$cell" ]]; then
        echo "ERROR incomplete output exists: $cell" >&2
        exit 2
      fi
      PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON" -u "$RUNNER" \
        --dataset "$dataset" --order "$order" --stage "$stage" \
        --data "$DATA_ROOT/$dataset.pt" \
        --text-cache "$CACHE_ROOT/${dataset}_r005/text_emb_pool_only.pt" \
        --c0-root "$C0_ROOT" --semantic-root "$SEMANTIC_ROOT" \
        --output "$cell" --mode topology_only --device cpu \
        --teacher-steps 4 --inner-steps 2 --outer-steps 4 \
        --batch 16 --top-r 8 --message-mix 0.10 --swap-fraction 0.10 \
        --old-radius 0.05 --new-radius 0.25 \
        --teacher-seeds 9300 9301 --validation-teacher-seeds 9310 \
        > "$OUT/logs/${dataset}_${order}_s${stage}.log" 2>&1
    done
  done
done
echo TGR_TOPOLOGY_MIX10_20260927_COMPLETE
