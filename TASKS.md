# Experiment Task List

## Overview

This study runs 4 experiments on medical MCQ robustness using MedGemma models.

| Experiment | Dataset | Items | Conditions | Est. Time (4B) | Est. Time (27B) |
|------------|---------|-------|------------|----------------|-----------------|
| Exp 1: Prompt Ablation | MedMCQA | 4,183 | 5 | 3-4 hrs | 8-10 hrs |
| Exp 2: Option Order | MedMCQA | 4,183 | 5 | 4-5 hrs | 10-12 hrs |
| Exp 3: Evidence Conditioning | PubMedQA | 1,000 | 6 | 1-2 hrs | 3-4 hrs |
| Exp 4: Self-Consistency | Both | 5,183 | N=1,3,5,10 | 6-8 hrs | 15-20 hrs |

**Total estimated time (sequential):** ~15-19 hrs (4B) or ~36-46 hrs (27B)

---

## Parallel Execution Plan (8 GPUs)

### Recommended GPU Assignment

```
GPU 0: exp1_4b  (Prompt Ablation, MedGemma 4B)
GPU 1: exp1_27b (Prompt Ablation, MedGemma 27B)
GPU 2: exp2_4b  (Option Order, MedGemma 4B)
GPU 3: exp2_27b (Option Order, MedGemma 27B)
GPU 4: exp3_4b  (Evidence Conditioning, MedGemma 4B)
GPU 5: exp3_27b (Evidence Conditioning, MedGemma 27B)
GPU 6: exp4_4b  (Self-Consistency, MedGemma 4B)
GPU 7: exp4_27b (Self-Consistency, MedGemma 27B)
```

### Commands

```bash
# Run all 8 experiments in parallel
python scripts/run_parallel.py --gpus 8

# Or run manually on specific GPUs:
CUDA_VISIBLE_DEVICES=0 python main.py -e prompt_ablation -m 4b &
CUDA_VISIBLE_DEVICES=1 python main.py -e prompt_ablation -m 27b &
CUDA_VISIBLE_DEVICES=2 python main.py -e option_order -m 4b &
CUDA_VISIBLE_DEVICES=3 python main.py -e option_order -m 27b &
CUDA_VISIBLE_DEVICES=4 python main.py -e evidence_conditioning -m 4b &
CUDA_VISIBLE_DEVICES=5 python main.py -e evidence_conditioning -m 27b &
CUDA_VISIBLE_DEVICES=6 python main.py -e self_consistency -m 4b &
CUDA_VISIBLE_DEVICES=7 python main.py -e self_consistency -m 27b &
wait
```

---

## Detailed Task Descriptions

### Experiment 1: Prompt Recipe Ablation

**Research Question:** Which prompt choices actually move accuracy on medical MCQ tasks?

**Conditions:**
1. `zero_shot_direct` - Simple question + options, direct answer
2. `zero_shot_cot` - Chain-of-thought reasoning before answering
3. `few_shot_3_direct` - 3 examples + direct answer
4. `few_shot_3_cot` - 3 examples + chain-of-thought
5. `answer_only` - Minimal prompt, just the answer

**Metrics:**
- Overall accuracy per condition
- Per-subject accuracy breakdown (21 medical subjects)
- CoT gain: `acc(cot) - acc(direct)`
- Few-shot gain: `acc(few_shot) - acc(zero_shot)`

**Command:**
```bash
CUDA_VISIBLE_DEVICES=0 python main.py -e prompt_ablation -m 4b
CUDA_VISIBLE_DEVICES=1 python main.py -e prompt_ablation -m 27b
```

---

### Experiment 2: Option Order Sensitivity

**Research Question:** Do models change answers when options are reordered?

**Perturbations:**
1. `original` - Canonical A-B-C-D order
2. `random_shuffle` - Random permutation of options
3. `rotate_1` - Cyclic rotation by 1 position
4. `rotate_2` - Cyclic rotation by 2 positions
5. `distractor_swap` - Swap positions of two wrong answers

**Metrics:**
- Accuracy on original vs perturbed
- Flip rate: proportion of predictions that changed
- Consistency breakdown:
  - `consistent_correct`: both correct
  - `flip_to_wrong`: was correct, now wrong
  - `flip_to_correct`: was wrong, now correct
- Position bias score

**Command:**
```bash
CUDA_VISIBLE_DEVICES=2 python main.py -e option_order -m 4b
CUDA_VISIBLE_DEVICES=3 python main.py -e option_order -m 27b
```

---

### Experiment 3: Evidence Conditioning (PubMedQA)

**Research Question:** With and without the abstract context, how much does performance change?

**Conditions:**
1. `question_only` - No context provided
2. `full_context` - Complete abstract
3. `truncated_50` - First 50% of context
4. `truncated_25` - First 25% of context
5. `background_only` - Only BACKGROUND section
6. `results_only` - Only RESULTS section

**Metrics:**
- 3-class accuracy (yes/no/maybe)
- Context importance: `acc(full) - acc(none)`
- Truncation loss: `acc(full) - acc(truncated)`
- Section sufficiency scores

**Command:**
```bash
CUDA_VISIBLE_DEVICES=4 python main.py -e evidence_conditioning -m 4b
CUDA_VISIBLE_DEVICES=5 python main.py -e evidence_conditioning -m 27b
```

---

### Experiment 4: Self-Consistency Voting

**Research Question:** Can you trade extra sampling for better accuracy and confidence?

**Configuration:**
- Temperature: 0.7
- Sample counts: N = 1, 3, 5, 10
- Prompting: Chain-of-thought
- Aggregation: Majority vote
- Confidence: vote proportion for majority

**Metrics:**
- Accuracy@N for each sample count
- ECE (Expected Calibration Error)
- Brier score
- Mean confidence
- Reliability diagrams

**Command:**
```bash
CUDA_VISIBLE_DEVICES=6 python main.py -e self_consistency -m 4b
CUDA_VISIBLE_DEVICES=7 python main.py -e self_consistency -m 27b
```

---

## Quick Start

### 1. Setup Environment

```bash
# Clone and setup
git clone <repository>
cd MedMCQA-Robustness-Study
bash scripts/setup.sh
```

### 2. Test with Limited Data

```bash
# Quick test (50 items per experiment)
python scripts/run_parallel.py --gpus 8 --limit 50

# Or single experiment test
python main.py -e prompt_ablation -m 4b -l 100
```

### 3. Run Full Experiments

```bash
# All experiments, all models, 8 GPUs
python scripts/run_parallel.py --gpus 8

# Just 4B model (faster)
python scripts/run_parallel.py --gpus 4 --models 4b

# Specific experiments
python scripts/run_parallel.py --gpus 2 --experiments exp1 exp2 --models 4b
```

### 4. Monitor Progress

```bash
# Check running processes
ps aux | grep "python main.py"

# Check GPU usage
nvidia-smi

# Check output logs
tail -f outputs/results/*.log
```

### 5. Results Location

```
outputs/
├── results/          # JSON results for each experiment
├── figures/          # Generated plots
├── cache/            # Cached model responses (speeds up reruns)
└── checkpoints/      # Resume checkpoints
```

---

## Resource Requirements

| Model | VRAM | Recommended GPU |
|-------|------|-----------------|
| MedGemma 4B | ~10GB | RTX 3080, RTX 4080, A100 |
| MedGemma 27B (4-bit) | ~15GB | RTX 3090, RTX 4090, A100 |

**Note:** With 8x GPUs, you can run all 8 task combinations (4 experiments × 2 models) in parallel. Total wall-clock time will be determined by the longest task (~15-20 hrs for exp4_27b).

---

## Expected Outputs

After all experiments complete, you'll have:

1. **Accuracy tables** comparing prompt conditions
2. **Robustness metrics** showing flip rates and consistency
3. **Context importance scores** for PubMedQA
4. **Calibration analysis** with reliability diagrams
5. **Per-subject heatmaps** for MedMCQA

These provide clean ablation tables and concrete reliability findings suitable for publication.
