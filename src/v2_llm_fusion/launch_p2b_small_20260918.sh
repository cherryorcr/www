#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/pc/research/TAG_condensation
CODE="$ROOT/code/graphclip_unified_v3_2026_09_13"
PY=/home/pc/miniconda3/envs/tagsam/bin/python
OUT="$ROOT/results/p2b_student_attention_20260918"
mkdir -p "$OUT"
"$PY" -u "$CODE/run_p2b_student_attention.py" --dataset cora --device cuda:0 \
    --output "$OUT/cora" > "$OUT/cora.log" 2>&1 &
P0=$!
"$PY" -u "$CODE/run_p2b_student_attention.py" --dataset citeseer --device cuda:1 \
    --output "$OUT/citeseer" > "$OUT/citeseer.log" 2>&1 &
P1=$!
S0=0
S1=0
wait "$P0" || S0=$?
wait "$P1" || S1=$?
printf 'P2B exit codes: cora=%s citeseer=%s\n' "$S0" "$S1"
test "$S0" -eq 0 && test "$S1" -eq 0
