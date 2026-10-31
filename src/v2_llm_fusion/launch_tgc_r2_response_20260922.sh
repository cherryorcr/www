#!/usr/bin/env bash
set -euo pipefail
cd /home/pc/research/TAG_condensation/work_runs/tgc_r2_response_20260922_snapshot/code/v2_llm_fusion
ROOT=/home/pc/research/TAG_condensation/results/tgc_r2_response_20260922
REF=/home/pc/research/TAG_condensation/results/tgc_r_phase_a_20260922
CAP=/home/pc/research/TAG_condensation/results/tgc_r2_capacity_20260922
PY=/home/pc/miniconda3/envs/tagsam/bin/python
mkdir -p "$ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
"$PY" -u run_tgc_r2_response_diagnostic.py --dataset cora --reference "$REF" --capacity "$CAP" --output "$ROOT" --device cuda:0 > "$ROOT/cora.log" 2>&1 &
P0=$!
"$PY" -u run_tgc_r2_response_diagnostic.py --dataset citeseer --reference "$REF" --capacity "$CAP" --output "$ROOT" --device cuda:1 > "$ROOT/citeseer.log" 2>&1 &
P1=$!
printf 'cora=%s citeseer=%s\n' "$P0" "$P1"
S0=0
S1=0
wait "$P0" || S0=$?
wait "$P1" || S1=$?
printf 'cora_exit=%s citeseer_exit=%s\n' "$S0" "$S1"
test "$S0" -eq 0
test "$S1" -eq 0
"$PY" audit_tgc_r2_response.py --root "$ROOT"
printf 'TGC_R2_ALL_RESPONSE_JOBS_COMPLETE\n'
