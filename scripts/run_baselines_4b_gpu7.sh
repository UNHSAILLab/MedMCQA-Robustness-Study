#!/bin/bash
# Run all baseline experiments on GPU 7 with --limit 1000
# cloze_score 4B already done on full dataset.
set -e
export CUDA_VISIBLE_DEVICES=7
cd /home/bsada1/coderepo/MedMCQA-Robustness-Study

echo "=========================================="
echo "=== Permutation Vote (4B, 1000 items) ==="
echo "Start: $(date)"
python main.py -e permutation_vote -m 4b --limit 1000
echo "Done:  $(date)"

echo ""
echo "============================================="
echo "=== CoT Self-Consistency (4B, 1000 items) ==="
echo "Start: $(date)"
python main.py -e cot_self_consistency -m 4b --limit 1000
echo "Done:  $(date)"

echo ""
echo "======================================="
echo "=== Cloze Score (27B, 1000 items)   ==="
echo "Start: $(date)"
python main.py -e cloze_score -m 27b --limit 1000
echo "Done:  $(date)"

echo ""
echo "==========================================="
echo "=== Permutation Vote (27B, 1000 items) ==="
echo "Start: $(date)"
python main.py -e permutation_vote -m 27b --limit 1000
echo "Done:  $(date)"

echo ""
echo "=============================================="
echo "=== CoT Self-Consistency (27B, 1000 items) ==="
echo "Start: $(date)"
python main.py -e cot_self_consistency -m 27b --limit 1000
echo "Done:  $(date)"

echo ""
echo "=========================================="
echo "=== ALL BASELINE EXPERIMENTS COMPLETE ==="
echo "Finish: $(date)"
