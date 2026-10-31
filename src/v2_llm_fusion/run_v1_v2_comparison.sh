#!/bin/bash
# Experimental script to run v1 vs v2 comparison on Cora
# Run this on the server: bash run_v1_v2_comparison.sh

set -e

# Configuration
DATA_PATH="/home/pc/research/GraphCLIP/processed_data/cora.pt"
MINILM_PATH="/home/pc/models/all-MiniLM-L6-v2"
OUTPUT_BASE="/home/pc/research/problem_validation_results/v2_llm_fusion_$(date +%Y_%m_%d)"
DEVICE="cuda:0"

# Experiment settings
RATE=0.05
FUSE_G=5
DEGREE=1.0
STUDENT_SEEDS="100 101 102 103 104 105"
STUDENT_STEPS=2048

# LLM backends to test
LLM_BACKENDS=("gpt4o" "claude-sonnet")
TARGET_WORDS=200

echo "========================================"
echo "V1 vs V2 Comparison Experiment"
echo "========================================"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_BASE"
echo "Rate: $RATE, g=$FUSE_G, deg=$DEGREE"
echo "Student: ${STUDENT_STEPS} steps, seeds: $STUDENT_SEEDS"
echo ""

mkdir -p "$OUTPUT_BASE"

# ============================================================
# Baseline: V1 (concatenation)
# ============================================================
echo "=== Running V1 Baseline (concatenation) ==="
python run_v2_comparison.py \
    --data "$DATA_PATH" \
    --minilm "$MINILM_PATH" \
    --output "$OUTPUT_BASE/v1_concat" \
    --device "$DEVICE" \
    --rate "$RATE" \
    --fuse-g "$FUSE_G" \
    --degree "$DEGREE" \
    --student-seeds $STUDENT_SEEDS \
    --student-steps "$STUDENT_STEPS" \
    --method v1_concat

echo ""
echo "V1 baseline complete!"
echo ""

# ============================================================
# V2: LLM fusion (test multiple backends)
# ============================================================
for backend in "${LLM_BACKENDS[@]}"; do
    echo "=== Running V2 with LLM backend: $backend ==="

    python run_v2_comparison.py \
        --data "$DATA_PATH" \
        --minilm "$MINILM_PATH" \
        --output "$OUTPUT_BASE/v2_llm_${backend}" \
        --device "$DEVICE" \
        --rate "$RATE" \
        --fuse-g "$FUSE_G" \
        --degree "$DEGREE" \
        --student-seeds $STUDENT_SEEDS \
        --student-steps "$STUDENT_STEPS" \
        --method v2_llm \
        --llm-backend "$backend" \
        --target-words "$TARGET_WORDS"

    echo ""
    echo "V2 with $backend complete!"
    echo ""
done

# ============================================================
# Generate comparison report
# ============================================================
echo "=== Generating comparison report ==="
python -c "
import json
from pathlib import Path

base = Path('$OUTPUT_BASE')
results = {}

# Load all results
for result_file in base.glob('*/results.json'):
    with open(result_file) as f:
        data = json.load(f)
        arm_name = result_file.parent.name
        results[arm_name] = {
            'method': data['method'],
            'llm_backend': data.get('llm_backend'),
            'dev_f1_mean': data['dev_macro_f1_mean'],
            'dev_f1_std': data['dev_macro_f1_std']
        }

# Print comparison table
print('\n' + '='*60)
print('COMPARISON RESULTS (Cora Dev Macro-F1)')
print('='*60)
print(f'{'Arm':<30} {'Mean':<10} {'Std':<10}')
print('-'*60)

for arm, res in sorted(results.items()):
    backend_str = f\"({res['llm_backend']})\" if res['llm_backend'] else ''
    arm_display = f\"{res['method']} {backend_str}\"
    print(f'{arm_display:<30} {res[\"dev_f1_mean\"]:<10.2f} {res[\"dev_f1_std\"]:<10.2f}')

print('='*60)

# Save summary
summary = {
    'experiment': 'v1_concat vs v2_llm_fusion',
    'dataset': 'Cora',
    'rate': $RATE,
    'fuse_g': $FUSE_G,
    'degree': $DEGREE,
    'student_steps': $STUDENT_STEPS,
    'results': results
}

with open(base / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'\nSummary saved to {base / \"summary.json\"}')
"

echo ""
echo "========================================"
echo "All experiments complete!"
echo "Results saved to: $OUTPUT_BASE"
echo "========================================"
