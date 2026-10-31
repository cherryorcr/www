#!/usr/bin/env bash
set -euo pipefail
cd /home/pc/research/TAG_condensation
PY=/home/pc/miniconda3/envs/tagsam/bin/python
ROOT=/home/pc/research/TAG_condensation/results/direct_student_outer_20260928
DATA=/home/pc/research/GraphCLIP/processed_data
CACHE=/home/pc/research/TAG_condensation/results/p5_rate_curve_small_20260919
MINILM=/home/pc/research/TAGSAM/graphclip_native_work/minilm_from_graphclip

run_one() {
  local dataset="$1" name="$2" old="$3"
  "$PY" code/v2_llm_fusion/prepare_direct_student_prompt_support.py \
    --dataset "$dataset" --order "${name#*_}" --stage 1 \
    --data "$DATA/${dataset}.pt" \
    --text-cache "$CACHE/${dataset}_r005/text_emb_pool_only.pt" \
    --support "$ROOT/prep/${name}/support.pt" --old-rows "$old" \
    --minilm "$MINILM" --output "$ROOT/prompt_support_v4_${name}"
}

run_one cora cora_canonical 25
run_one cora cora_order19102 35
run_one citeseer citeseer_canonical 38
run_one citeseer citeseer_order19102 40
