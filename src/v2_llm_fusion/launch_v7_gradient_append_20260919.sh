#!/usr/bin/env bash
set -euo pipefail
snapshot=/home/pc/research/TAG_condensation/experiments/v7a_gradient_append_20260919
output=/home/pc/research/TAG_condensation/results/v7a_gradient_append_20260919
python=/home/pc/miniconda3/envs/tagsam/bin/python
runner="$snapshot/code/v2_llm_fusion/run_v7_gradient_append.py"
protocol="$snapshot/docs/current/V7A_GRADIENT_COMPATIBLE_APPEND_FROZEN_PROTOCOL_ZH_2026-09-19.md"
mkdir -p "$output"
cd "$snapshot"
"$python" "$runner" --dataset cora --output "$output/cora" --device cuda:0 --protocol "$protocol" "$@" >> "$output/cora.log" 2>&1 &
cora_pid=$!
"$python" "$runner" --dataset citeseer --output "$output/citeseer" --device cuda:1 --protocol "$protocol" "$@" >> "$output/citeseer.log" 2>&1 &
citeseer_pid=$!
status=0
wait "$cora_pid" || status=1
wait "$citeseer_pid" || status=1
if [[ "$status" -eq 0 && "$#" -eq 0 ]]; then
    "$python" "$snapshot/code/v2_llm_fusion/audit_v7_gradient_append.py" \
        --root "$output" --protocol "$protocol" --reload-final --reload-gradients \
        --device cuda:0 --output-name audit_summary_reload.json >> "$output/audit.log" 2>&1 || status=1
fi
exit "$status"
