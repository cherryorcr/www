#!/usr/bin/env bash
set -euo pipefail
cd /home/pc/research/TAG_condensation/work_runs/tgc_r3_selector_20260922_snapshot/code/v2_llm_fusion
ROOT=/home/pc/research/TAG_condensation/results/tgc_r3_selector_confirmation_20260922
PY=/home/pc/miniconda3/envs/tagsam/bin/python
mkdir -p "$ROOT"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
"$PY" -m pytest -q test_tgc_selector_confirmation.py test_tgc_response_alignment.py test_tgc_student_revision.py test_tgc_r3_reporting.py > "$ROOT/preflight.log" 2>&1
"$PY" -u run_tgc_selector_confirmation.py --dataset cora --phase prepare --output "$ROOT" --device cuda:0 > "$ROOT/cora_prepare.log" 2>&1
"$PY" -u run_tgc_selector_confirmation.py --dataset citeseer --phase prepare --output "$ROOT" --device cuda:1 > "$ROOT/citeseer_prepare.log" 2>&1
"$PY" -u run_tgc_selector_confirmation.py --phase seal --output "$ROOT"
"$PY" -u run_tgc_selector_confirmation.py --dataset cora --phase evaluate --output "$ROOT" --device cuda:0 > "$ROOT/cora.log" 2>&1 &
P0=$!
"$PY" -u run_tgc_selector_confirmation.py --dataset citeseer --phase evaluate --output "$ROOT" --device cuda:1 > "$ROOT/citeseer.log" 2>&1 &
P1=$!
S0=0
S1=0
wait "$P0" || S0=$?
wait "$P1" || S1=$?
printf 'cora_exit=%s citeseer_exit=%s\n' "$S0" "$S1"
test "$S0" -eq 0
test "$S1" -eq 0
"$PY" -u audit_tgc_selector_confirmation.py --root "$ROOT" --device cuda:0 > "$ROOT/audit.log" 2>&1
PYTHONPATH=. "$PY" -u summarize_tgc_selector_confirmation.py --root "$ROOT"
printf 'TGC_R3_SELECTOR_CONFIRMATION_COMPLETE\n'
