# Experiment Log and Project Status

**Last updated:** 2026-02-11
**Authors:** Binesh Sadanandan, Vahid Behzadan (SAIL Lab, University of New Haven)

This document provides a comprehensive record of every experiment, result file, codebase component, and running process in this project. Use it to resume work after a break.

---

## Table of Contents

1. [Quick Resume Checklist](#quick-resume-checklist)
2. [Hardware and Environment](#hardware-and-environment)
3. [Project Structure](#project-structure)
4. [Configuration](#configuration)
5. [Experiment Definitions](#experiment-definitions)
6. [Completed Experiment Results](#completed-experiment-results)
7. [In-Progress Experiments](#in-progress-experiments)
8. [Result Files Inventory](#result-files-inventory)
9. [Cache and Checkpointing](#cache-and-checkpointing)
10. [How to Resume Experiments](#how-to-resume-experiments)
11. [Paper Status](#paper-status)
12. [Known Issues and Notes](#known-issues-and-notes)

---

## Quick Resume Checklist

When you come back to this project:

1. **Check if experiments are still running:** `nvidia-smi` and `ps aux | grep run_experiment`
2. **Check cache size:** `python3 -c "import sqlite3; db=sqlite3.connect('outputs/cache/responses.db'); print(db.execute('SELECT COUNT(*) FROM cache').fetchone())"`
3. **Check checkpoint files:** `ls -lh outputs/checkpoints/`
4. **Review latest result files:** `ls -lt outputs/results/*.json | head -10`
5. **Resume any killed experiments** using the commands in [How to Resume Experiments](#how-to-resume-experiments)

---

## Hardware and Environment

| Component | Details |
|-----------|---------|
| GPUs | 8x NVIDIA A100 80GB PCIe |
| Python | 3.9+ |
| PyTorch | 2.0+ with CUDA |
| Transformers | 4.40+ |
| Key packages | `bitsandbytes`, `accelerate`, `pydantic>=2.0`, `datasets` |
| HuggingFace | Logged in with access to `google/medgemma-4b-it` and `google/medgemma-27b-text-it` |

**VRAM requirements:**
- MedGemma-4B: ~9 GB (fits on any single GPU)
- MedGemma-27B: ~55 GB (full bfloat16 only; quantization produces NaN outputs)

---

## Project Structure

```
MedMCQA-Robustness-Study/
├── main.py                              # CLI entry point (alternative to scripts/run_experiment.py)
├── requirements.txt                     # Python dependencies
├── configs/
│   └── base.yaml                        # Single source of truth for all inference parameters
│
├── src/                                 # Core library
│   ├── __init__.py
│   ├── data/
│   │   ├── loaders.py                   # MedMCQALoader, PubMedQALoader (HuggingFace datasets)
│   │   └── schemas.py                   # MCQItem, PubMedQAItem (Pydantic models)
│   ├── models/
│   │   ├── base.py                      # Abstract BaseModel (generate, generate_with_logprobs)
│   │   ├── medgemma.py                  # MedGemmaModel (4B and 27B, with quantization options)
│   │   └── medical_llms.py              # MedicalLLM (BioMistral, Meditron)
│   ├── experiments/
│   │   ├── base.py                      # BaseExperiment (caching, checkpointing, result saving)
│   │   ├── exp1_prompt_ablation.py      # PromptAblationExperiment
│   │   ├── exp2_option_order.py         # OptionOrderExperiment
│   │   ├── exp3_evidence_conditioning.py# EvidenceConditioningExperiment
│   │   ├── exp4_self_consistency.py     # SelfConsistencyExperiment
│   │   ├── exp5_robust_baselines.py     # CoTSelfConsistencyExperiment, PermutationVoteExperiment, ClozeScoreExperiment
│   │   └── multi_seed_runner.py         # Multi-seed aggregation support
│   ├── prompts/
│   │   ├── templates.py                 # PromptStyle enum, MedMCQAPromptTemplate, PubMedQAPromptTemplate
│   │   ├── few_shot_examples.py         # 5 curated MedMCQA examples (Medicine, Biochem, Anatomy, Pharm, Path)
│   │   └── few_shot_selector.py         # FewShotSelector: random, label_balanced, subject_matched
│   ├── perturbations/
│   │   ├── option_shuffle.py            # OptionShuffler: random, rotate_1, rotate_2, distractor_swap
│   │   └── context_truncation.py        # ContextTruncator: remove, truncate_ratio, extract_sections, etc.
│   ├── evaluation/
│   │   ├── metrics.py                   # MCQMetrics, PubMedQAMetrics (accuracy, CI, position bias)
│   │   ├── calibration.py              # ECE, SelfConsistencyCalibration
│   │   ├── significance.py             # McNemar's test, Bonferroni correction
│   │   ├── quality_controls.py         # Label remapping, parser failure rates, audit
│   │   └── visualization.py            # Plotting utilities
│   └── utils/
│       ├── caching.py                   # ResponseCache (SQLite-based, keyed by hash of prompt+model+experiment)
│       └── checkpointing.py            # ExperimentCheckpoint (JSON, saves every 100 items)
│
├── scripts/                             # Runner scripts
│   ├── run_experiment.py                # Main CLI runner (preferred over main.py)
│   ├── run_multi_seed.py               # Multi-seed runner with aggregation
│   ├── run_parallel.py                 # Distributed multi-GPU execution
│   ├── run_baselines_4b_gpu7.sh        # Shell script for baseline experiments
│   ├── run_quality_checks.py           # Validates result file integrity
│   ├── error_analysis.py               # Post-hoc error analysis
│   ├── monitor_progress.sh             # SQLite cache progress checker
│   ├── setup.sh                        # Environment setup
│   ├── debug_4b.py                     # Debug script for 4B model
│   └── debug_27b.py                    # Debug script for 27B model
│
├── notebooks/
│   └── colab_demo.ipynb                # Google Colab demo notebook
│
├── paper/                              # Research paper
│   ├── main.tex                        # Original article-class paper
│   ├── main.pdf                        # Compiled PDF
│   ├── references.bib                  # Bibliography (22 references)
│   ├── figures/                        # 6 figures (PDF + PNG, ~300 DPI)
│   ├── generate_figures.py             # Figure generation script
│   └── springer/                       # LNCS conference submission
│       ├── main.tex                    # Springer LNCS formatted paper (current working version)
│       ├── references.bib              # Copy of bibliography
│       ├── llncs.cls                   # Springer LNCS class file (v2.26)
│       ├── splncs04.bst                # BibTeX style
│       └── samplepaper.tex             # Template reference
│
├── outputs/                            # All output artifacts (gitignored)
│   ├── results/                        # Experiment result JSON files (~215 MB total)
│   ├── cache/                          # SQLite response cache (responses.db, ~160 MB)
│   ├── checkpoints/                    # Experiment checkpoints for resume
│   ├── audit/                          # CSV audit files for evidence conditioning
│   └── other/                          # Misc outputs
│
└── experiment.log                      # Recent experiment log
```

---

## Configuration

All experiments use `configs/base.yaml` as the single source of truth. Key parameters:

### Inference Parameters (Frozen)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `temperature` | 0.0 | Greedy decoding for deterministic, reproducible output |
| `max_new_tokens` | 256 | Sufficient for MCQ answer + reasoning; prevents runaway generation |
| `top_p` | 1.0 | Disabled (not needed with greedy decoding) |
| `top_k` | 0 | Disabled (not needed with greedy decoding) |
| `do_sample` | false | Deterministic decoding |
| `batch_size` | 4 | Balances throughput and VRAM usage on A100 |
| `checkpoint_interval` | 100 | Saves progress every 100 items |
| `seed` | 42 | Primary seed |
| `seeds` | [42, 123, 456, 789, 1337] | For multi-seed experiments |

### Self-Consistency Sampling (exp4, exp5)

| Parameter | Value |
|-----------|-------|
| `temperature` | 0.7 |
| `top_p` | 0.95 |
| `top_k` | 50 |
| `do_sample` | true |
| `sample_counts` (exp4) | [1, 3, 5, 10] |
| `sample_counts` (CoT SC) | [3, 5] |

### Permutation Vote (exp5)

| Parameter | Value |
|-----------|-------|
| `k` | 4 (number of option permutations per item) |

### Models

| Model | HuggingFace ID | Quantization | VRAM |
|-------|---------------|--------------|------|
| MedGemma-4B | `google/medgemma-4b-it` | None (bf16) | ~9 GB |
| MedGemma-27B | `google/medgemma-27b-text-it` | None (bf16) | ~55 GB |
| BioMistral-7B | `BioMistral/BioMistral-7B` | None (bf16) | ~14 GB |

### Datasets

| Dataset | HuggingFace ID | Split | N |
|---------|---------------|-------|---|
| MedMCQA | `openlifescienceai/medmcqa` | validation | 4,183 |
| PubMedQA | `qiaojin/PubMedQA` (pqa_labeled) | train | 1,000 |

---

## Experiment Definitions

### Experiment 1: Prompt Ablation (`exp1_prompt_ablation`)

**Goal:** Measure how prompt formatting affects accuracy on medical MCQs.

**Dataset:** MedMCQA validation (4,183 items)

**Conditions tested:**

| Condition | Description |
|-----------|-------------|
| `zero_shot_direct` | "Answer the following question. Reply with the letter only." |
| `zero_shot_cot` | "Think step by step, then answer." |
| `few_shot_3_direct` | 3 curated examples + direct answer |
| `few_shot_3_cot` | 3 curated examples + CoT reasoning |
| `answer_only` | Question + options, no instruction, no system prompt |
| `few_shot_3_random` | 3 randomly-selected examples (from training pool of 5,000) |
| `few_shot_3_balanced` | 3 label-balanced examples |
| `few_shot_3_subject` | 3 subject-matched examples |
| `few_shot_3_order_2` | Same examples, order permutation 2 |
| `few_shot_3_order_3` | Same examples, order permutation 3 |

**Run command:**
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py prompt_ablation -m 4b --seed 42
CUDA_VISIBLE_DEVICES=2 python scripts/run_experiment.py prompt_ablation -m 27b --seed 42
```

---

### Experiment 2: Option Order Sensitivity (`exp2_option_order`)

**Goal:** Measure how answer option ordering affects model predictions.

**Dataset:** MedMCQA validation (4,183 items)

**Perturbation types:**

| Perturbation | Description |
|--------------|-------------|
| `original` | Options in dataset order (A, B, C, D) |
| `random_shuffle` | Random reordering of all 4 options |
| `rotate_1` | Cyclic rotation by 1 position (B, C, D, A) |
| `rotate_2` | Cyclic rotation by 2 positions (C, D, A, B) |
| `distractor_swap` | Swap correct answer with a random distractor |

**Key metrics:** Accuracy, flip rate (% of items where answer changes), position bias score.

**Run command:**
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_experiment.py option_order -m 4b --seed 42
CUDA_VISIBLE_DEVICES=4 python scripts/run_experiment.py option_order -m 27b --seed 42
```

---

### Experiment 3: Evidence Conditioning (`exp3_evidence_conditioning`)

**Goal:** Measure how the amount and type of context affects PubMedQA performance.

**Dataset:** PubMedQA pqa_labeled (1,000 items)

**Context conditions:**

| Condition | Description |
|-----------|-------------|
| `question_only` | No context at all |
| `full_context` | Complete abstract text |
| `truncated_50` | First 50% of context (by character) |
| `truncated_25` | First 25% of context |
| `background_only` | Only BACKGROUND section |
| `results_only` | Only RESULTS/CONCLUSIONS section |
| `truncated_back_50` | Remove last 50% (keep beginning) |
| `truncated_middle_50` | Remove middle 50% (keep beginning + end) |
| `sentence_trunc_50` | Truncate at sentence boundaries to ~50% |
| `salient_top5` | Top 5 sentences by TF-IDF salience |
| `salient_top3` | Top 3 sentences by TF-IDF salience |

**Run command:**
```bash
CUDA_VISIBLE_DEVICES=6 python scripts/run_experiment.py evidence_conditioning -m 4b --seed 42
```

---

### Experiment 4: Self-Consistency (`exp4_self_consistency`)

**Goal:** Test if sampling multiple answers and majority-voting improves accuracy.

**Settings:** temperature=0.7, sample_counts=[1, 3, 5, 10]

---

### Experiment 5: Robust Baselines (`exp5_robust_baselines`)

Three mitigation strategies:

#### 5a: Cloze Scoring (`cloze_score`)
Bypasses generation entirely. Feeds prompt, measures log-probability of each option token (A/B/C/D), picks highest.

#### 5b: Permutation Vote (`permutation_vote`)
Runs the model K=4 times with different option orderings, majority-votes across answers (with inverse mapping back to original ordering).

#### 5c: CoT Self-Consistency (`cot_self_consistency`)
Combines CoT prompting with stochastic sampling (temp=0.7) and majority voting across N=[3, 5] samples.

**Run commands:**
```bash
# Baselines on GPU 7
CUDA_VISIBLE_DEVICES=7 python scripts/run_experiment.py cloze_score -m 4b
CUDA_VISIBLE_DEVICES=7 python scripts/run_experiment.py permutation_vote -m 4b -l 1000
CUDA_VISIBLE_DEVICES=7 python scripts/run_experiment.py cot_self_consistency -m 4b -l 500
```

---

## Completed Experiment Results

### Exp1: Prompt Ablation - MedGemma-4B (COMPLETE, n=4,183)

**File:** `outputs/results/exp1_prompt_ablation_medgemma_4b_20260129_075638.json` (33 MB)
**Date:** 2026-01-29 09:17

| Condition | Accuracy | 95% CI |
|-----------|----------|--------|
| **zero_shot_direct** | **47.6%** | [46.1%, 49.1%] |
| answer_only | 43.0% | [41.5%, 44.6%] |
| zero_shot_cot | 41.9% | [40.4%, 43.3%] |
| few_shot_3_cot | 40.8% | [39.4%, 42.3%] |
| few_shot_3_direct | 35.7% | [34.3%, 37.0%] |

**Key finding:** Zero-shot direct is the best strategy. CoT hurts by -5.7 pp. Few-shot hurts by -11.9 pp.

---

### Exp1: Prompt Ablation - BioMistral-7B (COMPLETE, n=4,183)

**File:** `outputs/results/exp1_prompt_ablation_biomistral_7b_20260129_112151.json` (22 MB)
**Date:** 2026-01-29 11:46

| Condition | Accuracy | 95% CI |
|-----------|----------|--------|
| answer_only | 38.6% | [37.1%, 40.0%] |
| zero_shot_direct | 0.65% | [0.4%, 0.9%] |
| zero_shot_cot | 0.0% | [0.0%, 0.0%] |
| few_shot_3_direct | 0.0% | [0.0%, 0.0%] |
| few_shot_3_cot | 0.0% | [0.0%, 0.0%] |

**Key finding:** BioMistral fails catastrophically on all structured prompting conditions except answer_only. It cannot follow MCQ formatting instructions.

---

### Exp2: Option Order - MedGemma-4B, Seed 42 (COMPLETE, n=4,183)

**File:** `outputs/results/exp2_option_order_medgemma_4b_20260210_003504.json` (34 MB)
**Date:** 2026-02-10 19:09

| Perturbation | Accuracy | Flip Rate |
|--------------|----------|-----------|
| Original | 48.5% | -- |
| Random shuffle | 27.7% | 60.4% |
| Rotate-1 | 19.9% | 73.0% |
| Rotate-2 | 20.7% | 72.5% |
| Distractor swap | 47.9% | 39.5% |

**Position bias score:** 0.111
**Predicted distribution:** A=43.3%, B=23.3%, C=16.1%, D=17.3%

---

### Exp2: Option Order - MedGemma-4B, Seed 42 (Earlier run, n=4,183)

**File:** `outputs/results/exp2_option_order_medgemma_4b_20260129_075639.json` (23 MB)
**Date:** 2026-01-29 09:13

| Perturbation | Accuracy | Flip Rate |
|--------------|----------|-----------|
| Original | 47.6% | -- |
| Random shuffle | 27.3% | 57.8% |
| Rotate-1 | 20.2% | 73.0% |
| Rotate-2 | 21.8% | 69.7% |
| Distractor swap | 47.6% | 36.1% |

**Position bias score:** 0.137

---

### Exp2: Option Order - BioMistral-7B (COMPLETE, n=4,183)

**File:** `outputs/results/exp2_option_order_biomistral_7b_20260129_160719.json` (12 MB)
**Date:** 2026-01-29 16:17

All conditions near 0% accuracy (BioMistral cannot follow MCQ formatting).

---

### Exp3: Evidence Conditioning - MedGemma-4B (COMPLETE, n=1,000, 11 conditions)

**File:** `outputs/results/exp3_evidence_conditioning_medgemma_4b_20260209_202031.json` (23 MB)
**Date:** 2026-02-10 06:22

| Condition | Accuracy | Yes acc | No acc | Maybe acc |
|-----------|----------|---------|--------|-----------|
| **full_context** | **45.8%** | 35.7% | 58.0% | 59.1% |
| truncated_back_50 | 44.5% | 32.1% | 61.5% | 54.5% |
| results_only | 41.9% | 28.1% | 60.1% | 55.5% |
| salient_top5 | 40.1% | 35.1% | 44.7% | 50.9% |
| salient_top3 | 36.4% | 29.0% | 41.1% | 59.1% |
| question_only | 34.5% | 39.3% | 25.4% | 38.2% |
| truncated_middle_50 | 33.9% | 23.9% | 45.6% | 48.2% |
| sentence_trunc_50 | 30.3% | 28.8% | 24.6% | 55.5% |
| background_only | 28.9% | 22.8% | 28.7% | 60.0% |
| truncated_50 | 13.8% | 3.6% | 4.4% | 93.6% |
| truncated_25 | 13.7% | 2.7% | 6.5% | 90.9% |

**Key finding:** Naive character-level truncation at 50% drops accuracy to 13.8% (below the 34.5% no-context baseline). Truncation from the back (keep beginning) or salience-based selection perform much better.

---

### Exp3: Evidence Conditioning - MedGemma-4B (Earlier run, 6 conditions, n=1,000)

**File:** `outputs/results/exp3_evidence_conditioning_medgemma_4b_20260129_075639.json` (9 MB)
**Date:** 2026-01-29 08:19

| Condition | Accuracy |
|-----------|----------|
| full_context | 45.0% |
| results_only | 41.7% |
| question_only | 36.7% |
| background_only | 26.5% |
| truncated_50 | 14.1% |
| truncated_25 | 13.1% |

---

### Exp3: Evidence Conditioning - MedGemma-27B (COMPLETE, 6 conditions, n=1,000)

**File:** `outputs/results/exp3_evidence_conditioning_medgemma_27b_20260131_132552.json` (13 MB)
**Date:** 2026-02-01 00:36

| Condition | Accuracy | Yes acc | No acc | Maybe acc |
|-----------|----------|---------|--------|-----------|
| results_only | **40.0%** | 24.8% | 71.3% | 20.0% |
| full_context | 38.2% | 17.8% | 80.5% | 10.9% |
| question_only | 31.0% | 1.8% | 81.7% | 21.8% |
| truncated_50 | 23.4% | 6.7% | 44.7% | 41.8% |
| background_only | 19.8% | 4.7% | 34.9% | 49.1% |
| truncated_25 | 18.6% | 3.4% | 36.4% | 40.0% |

**Key finding:** 27B model shows heavy "no" bias (81.7% of answers on question-only). Results-only context actually outperforms full context for 27B.

---

### Exp3: Evidence Conditioning - BioMistral-7B (COMPLETE, 6 conditions, n=1,000)

**File:** `outputs/results/exp3_evidence_conditioning_biomistral_7b_20260129_160720.json` (5.7 MB)
**Date:** 2026-01-29 16:11

All conditions near 0-1.8% accuracy (BioMistral cannot follow PubMedQA formatting).

---

### Exp5: Cloze Scoring - MedGemma-4B (COMPLETE, n=4,183)

**File:** `outputs/results/exp5_cloze_score_medgemma_4b_20260209_211452.json` (1.4 MB)
**Date:** 2026-02-09 21:17

| Metric | Value |
|--------|-------|
| Accuracy | 51.8% |
| 95% CI | [50.3%, 53.3%] |
| Position bias score | 0.013 |
| Mean logprob margin | 4.07 +/- 2.38 |
| Predicted dist | A=32.0%, B=25.7%, C=23.5%, D=18.9% |

**Key finding:** Cloze scoring recovers +4.2 pp over zero-shot direct (47.6% to 51.8%) and dramatically reduces position bias from 0.137 to 0.013.

---

### Exp5: Cloze Scoring - MedGemma-27B (COMPLETE, n=1,000)

**File:** `outputs/results/exp5_cloze_score_medgemma_27b_20260210_103334.json` (333 KB)
**Date:** 2026-02-10 10:34

| Metric | Value |
|--------|-------|
| **Accuracy** | **64.5%** |
| 95% CI | [61.4%, 67.1%] |
| Position bias score | 0.054 |
| Mean logprob margin | 3.14 +/- 2.09 |
| Predicted dist | A=36.8%, B=22.3%, C=19.4%, D=21.5% |

**Key finding:** 27B cloze scoring achieves the highest accuracy of any method tested (64.5%).

---

### Exp5: Permutation Vote - MedGemma-4B (COMPLETE, K=4, n=1,000)

**File:** `outputs/results/exp5_permutation_vote_medgemma_4b_20260209_220342.json` (517 KB)
**Date:** 2026-02-10 08:47

| Metric | Value |
|--------|-------|
| Aggregated accuracy (vote) | 49.0% |
| 95% CI | [46.0%, 52.1%] |
| Per-permutation mean | 45.1% +/- 1.7% |
| Per-permutation accuracies | [47.7%, 43.6%, 43.5%, 45.4%] |
| Mean agreement rate | 70.0% +/- 22.1% |
| N permutations | 4 |

**Key finding:** Permutation voting modestly improves accuracy (+1.4 pp over the per-permutation mean) and the 70% agreement rate shows substantial cross-permutation consistency.

---

### Exp5: CoT Self-Consistency - MedGemma-4B (PARTIAL, n=10 pilot)

**File:** `outputs/results/exp5_cot_self_consistency_medgemma_4b_20260209_201255.json` (419 KB)
**Date:** 2026-02-09 21:09

| N samples | Accuracy | Mean confidence |
|-----------|----------|-----------------|
| 3 | 50.0% | 0.800 +/- 0.163 |
| 5 | 60.0% | 0.720 +/- 0.223 |
| 10 | 80.0% | 0.750 +/- 0.186 |

**Note:** This is only a 10-item pilot. Full run (n=500+) was in progress on GPU 7 as of 2026-02-11.

---

### Error Analysis (from exp1 MedGemma-4B)

**File:** `outputs/results/error_analysis.json` (18 KB)

| Metric | Value |
|--------|-------|
| CoT hurt (items where CoT flipped correct to wrong) | 750 |
| CoT helped (items where CoT flipped wrong to correct) | 512 |
| CoT net effect | -238 items |
| Few-shot hurt | 979 |
| Few-shot helped | 481 |
| Few-shot net effect | -498 items |

**CoT failure patterns:**
- Verbose reasoning (wrong due to overthinking): 680 items
- Self-contradiction (contradicts own reasoning): 192 items
- Wrong conclusion (correct reasoning, wrong final answer): 83 items
- Expressed uncertainty: 1 item

---

## In-Progress Experiments

As of 2026-02-11 03:52 UTC, the following experiments were actively running:

### GPU Allocation

| GPU | Experiment | Model | Seed | Status |
|-----|-----------|-------|------|--------|
| 0 | exp1 prompt_ablation | MedGemma-4B | 42 | Running (multi-seed, extended conditions) |
| 1 | exp2 option_order | MedGemma-4B | 123 | Running (started 2026-02-10 19:18) |
| 2 | exp1 prompt_ablation | MedGemma-27B | 42 | Running (multi-seed) |
| 3 | exp2 option_order | MedGemma-27B | 456 | Running |
| 4 | exp2 option_order | MedGemma-27B | 42 | Running |
| 5 | exp2 option_order | MedGemma-27B | 123 | Running |
| 6 | exp5 permutation_vote | MedGemma-27B | -- | Running (n=1,000) |
| 7 | exp5 cot_self_consistency | MedGemma-4B | -- | Running (n=500, was on N=5 samples) |

### Cache Progress (as of 2026-02-11 03:52 UTC)

| Experiment | Model | Cached Responses |
|------------|-------|-----------------|
| exp1_prompt_ablation | medgemma_4b | 28,282 |
| exp1_prompt_ablation | medgemma_27b | 17,372 |
| exp2_option_order | medgemma_4b | 19,129 |
| exp2_option_order | medgemma_27b | 21,466 |
| exp3_evidence_conditioning | medgemma_4b | 10,974 |
| **Total** | | **97,223** |

### What Still Needs to Run

1. **Exp1 prompt_ablation - MedGemma-27B** (full n=4,183 with all 10 conditions) -- was running on GPU 2
2. **Exp1 prompt_ablation - MedGemma-4B** (multi-seed extended conditions) -- was running on GPU 0
3. **Exp2 option_order - MedGemma-4B** (seeds 123, 456) -- seed 123 running on GPU 1; seed 456 not yet started
4. **Exp2 option_order - MedGemma-27B** (seeds 42, 123, 456) -- running on GPUs 3, 4, 5
5. **Exp5 permutation_vote - MedGemma-27B** (n=1,000) -- running on GPU 6
6. **Exp5 cot_self_consistency - MedGemma-4B** (n=500, N=[3,5] samples) -- running on GPU 7
7. **Exp5 cot_self_consistency - MedGemma-27B** -- not yet started
8. **Exp5 permutation_vote results need to be updated in paper** when complete
9. **Exp3 evidence_conditioning - MedGemma-27B with extended conditions** (truncated_back, truncated_middle, salient) -- not yet started

---

## Result Files Inventory

### Full Runs (Primary Results for Paper)

| File | Model | Experiment | N | Size | Date |
|------|-------|-----------|---|------|------|
| `exp1_prompt_ablation_medgemma_4b_20260129_075638.json` | 4B | Prompt ablation (5 conds) | 4,183 | 33 MB | Jan 29 |
| `exp2_option_order_medgemma_4b_20260210_003504.json` | 4B | Option order | 4,183 | 34 MB | Feb 10 |
| `exp2_option_order_medgemma_4b_20260129_075639.json` | 4B | Option order (earlier) | 4,183 | 23 MB | Jan 29 |
| `exp3_evidence_conditioning_medgemma_4b_20260209_202031.json` | 4B | Evidence cond (11 conds) | 1,000 | 23 MB | Feb 10 |
| `exp3_evidence_conditioning_medgemma_4b_20260129_075639.json` | 4B | Evidence cond (6 conds) | 1,000 | 9 MB | Jan 29 |
| `exp3_evidence_conditioning_medgemma_27b_20260131_132552.json` | 27B | Evidence cond (6 conds) | 1,000 | 13 MB | Feb 1 |
| `exp5_cloze_score_medgemma_4b_20260209_211452.json` | 4B | Cloze scoring | 4,183 | 1.4 MB | Feb 9 |
| `exp5_cloze_score_medgemma_27b_20260210_103334.json` | 27B | Cloze scoring | 1,000 | 333 KB | Feb 10 |
| `exp5_permutation_vote_medgemma_4b_20260209_220342.json` | 4B | Permutation vote (K=4) | 1,000 | 517 KB | Feb 10 |
| `exp1_prompt_ablation_biomistral_7b_20260129_112151.json` | BioMistral | Prompt ablation | 4,183 | 22 MB | Jan 29 |
| `exp2_option_order_biomistral_7b_20260129_160719.json` | BioMistral | Option order | 4,183 | 12 MB | Jan 29 |
| `exp3_evidence_conditioning_biomistral_7b_20260129_160720.json` | BioMistral | Evidence cond | 1,000 | 5.7 MB | Jan 29 |
| `error_analysis.json` | 4B | Error analysis | -- | 18 KB | Jan 29 |

### Pilot / Debug Runs (Small N)

| File | N | Notes |
|------|---|-------|
| `exp1_prompt_ablation_medgemma_4b_20260209_201715.json` | 10 | 10 conditions, pilot |
| `exp1_prompt_ablation_medgemma_27b_20260209_201530.json` | 10 | 10 conditions, pilot |
| `exp2_option_order_medgemma_4b_20260209_202353.json` | 10 | Pilot |
| `exp3_evidence_conditioning_medgemma_4b_20260209_201251.json` | 10 | 11 conditions, pilot |
| `exp5_cot_self_consistency_medgemma_4b_20260209_201255.json` | 10 | CoT SC pilot |
| `exp1_prompt_ablation_biomistral_7b_20260129_112215.json` | 10 | BioMistral pilot |
| `exp2_option_order_biomistral_7b_20260129_165952.json` | 5 | BioMistral pilot |
| `exp1_prompt_ablation_biomistral_7b_20260129_112620.json` | 4,183 | Duplicate of earlier run |

---

## Cache and Checkpointing

### Response Cache

**Location:** `outputs/cache/responses.db` (SQLite, ~160 MB)

The cache stores every model response keyed by `hash(model_name + experiment + prompt_text)`. This means:
- **Restarting an experiment will reuse cached responses** (no re-inference needed)
- **Changing the prompt template invalidates the cache entry** for that item
- The cache is shared across all experiments and seeds

**Schema:** Table `cache` with columns: `key` (TEXT PRIMARY KEY), `response` (TEXT), `model_name` (TEXT), `experiment` (TEXT), `created_at` (TIMESTAMP)

**Querying the cache:**
```bash
# Count total cached responses
python3 -c "import sqlite3; db=sqlite3.connect('outputs/cache/responses.db'); print(db.execute('SELECT COUNT(*) FROM cache').fetchone())"

# Count by experiment and model
python3 -c "
import sqlite3
db = sqlite3.connect('outputs/cache/responses.db')
for row in db.execute('SELECT experiment, model_name, COUNT(*) FROM cache GROUP BY experiment, model_name ORDER BY experiment'):
    print(f'{row[0]:30s} | {row[1]:20s} | {row[2]:6d}')
"
```

### Checkpoints

**Location:** `outputs/checkpoints/`

Active checkpoints (as of 2026-02-11):

| File | Size | Last Modified | Status |
|------|------|---------------|--------|
| `exp1_prompt_ablation_medgemma_4b_checkpoint.json` | 7.2 MB | Feb 10 22:48 | Active |
| `exp1_prompt_ablation_medgemma_27b_checkpoint.json` | 1.1 MB | Feb 10 22:50 | Active |
| `exp2_option_order_medgemma_4b_checkpoint.json` | 6.6 MB | Feb 10 22:50 | Active |
| `exp2_option_order_medgemma_27b_checkpoint.json` | 5.2 MB | Feb 10 22:44 | Active |
| `exp1_prompt_ablation_medgemma_27b_4bit_checkpoint.json` | 4.7 MB | Feb 1 15:34 | Stale (4bit deprecated) |
| `exp2_option_order_medgemma_27b_4bit_checkpoint.json` | 1.8 MB | Feb 1 15:24 | Stale (4bit deprecated) |

Checkpoints are automatically cleared when an experiment completes successfully.

---

## How to Resume Experiments

The system is designed for safe kill-and-resume:

1. **Kill the process:** `kill <PID>` or Ctrl+C
2. **Model loading time:** ~1-2 min (4B), ~5 min (27B) on each restart
3. **No lost work:** The SQLite cache retains all completed responses. On restart, the experiment checks the cache and skips items that already have responses.
4. **Checkpoints:** If a checkpoint file exists, the experiment resumes from the last checkpoint.

### Resume Commands

```bash
# Resume exp1 prompt ablation (4B)
CUDA_VISIBLE_DEVICES=0 python scripts/run_experiment.py prompt_ablation -m 4b --seed 42

# Resume exp2 option order (4B, seed 123)
CUDA_VISIBLE_DEVICES=1 python scripts/run_experiment.py option_order -m 4b --seed 123

# Resume exp1 prompt ablation (27B)
CUDA_VISIBLE_DEVICES=2 python scripts/run_experiment.py prompt_ablation -m 27b --seed 42

# Resume exp2 option order (27B, seeds 42/123/456)
CUDA_VISIBLE_DEVICES=4 python scripts/run_experiment.py option_order -m 27b --seed 42
CUDA_VISIBLE_DEVICES=5 python scripts/run_experiment.py option_order -m 27b --seed 123
CUDA_VISIBLE_DEVICES=3 python scripts/run_experiment.py option_order -m 27b --seed 456

# Resume exp5 permutation vote (27B)
CUDA_VISIBLE_DEVICES=6 python main.py -e permutation_vote -m 27b -l 1000

# Resume exp5 CoT self-consistency (4B)
CUDA_VISIBLE_DEVICES=7 python main.py -e cot_self_consistency -m 4b -l 500
```

### Monitoring

```bash
# GPU usage
nvidia-smi

# Running experiment processes
ps aux | grep run_experiment | grep -v grep

# Cache growth (run periodically)
bash scripts/monitor_progress.sh

# Tail the experiment log
tail -f experiment.log
```

---

## Paper Status

### Current Version
The active paper is at `paper/springer/main.tex` (Springer LNCS format for conference submission).

### What's in the Paper
- 6 main sections + 8 appendix sections
- 15 tables, 6 figures
- ~10 LNCS pages (limit: 15)
- Style: cleaned for academic writing (no em dashes, no banned words, all abbreviations expanded)

### Paper Data Sources

| Paper Table | Source Result File | Status |
|-------------|-------------------|--------|
| Table 1 (Inference params) | `configs/base.yaml` | Complete |
| Table 2 (Experiment overview) | Manual | Complete |
| Table 3 (Prompt ablation) | `exp1_..._medgemma_4b_20260129_075638.json` | Complete |
| Table 4 (Option order) | `exp2_..._medgemma_4b_20260129_075639.json` | Complete |
| Table 5 (Evidence conditioning) | `exp3_..._medgemma_4b_20260129_075639.json` + `exp3_..._medgemma_27b_20260131_132552.json` | Complete |
| Table 6 (Sample responses) | Extracted from exp1 and exp2 result files | Complete |
| Table 7 (Baselines) | `exp5_cloze_*.json` + `exp5_permutation_vote_*.json` | Partial (27B perm vote and CoT SC still running) |
| Appendix tables | Various | Partial |

### What the Paper Still Needs
1. **27B prompt ablation results** (exp1, running on GPU 2)
2. **27B option order results** (exp2, running on GPUs 3-5)
3. **27B permutation vote results** (exp5, running on GPU 6)
4. **CoT self-consistency full results** (exp5, running on GPU 7)
5. **Confidence intervals and significance tests** for multi-seed experiments
6. **Reproducibility appendix** with all prompt templates and parsing rules
7. Final compilation and proofreading

### Compilation

The LNCS paper requires:
```bash
cd paper/springer
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

**Note:** There may be a texlive/conda conflict on the current machine. If `pdflatex` fails with `Can't locate mktexlsr.pl`, try: `export PATH=/usr/bin:$PATH` before compiling, or compile on a different machine.

---

## Known Issues and Notes

1. **MedGemma-27B 4-bit quantization produces NaN logits.** Always use full precision (bfloat16). The `medgemma_27b_4bit` config is marked deprecated in `base.yaml`.

2. **BioMistral-7B cannot follow MCQ formatting instructions.** It achieves <1% accuracy on all conditions except `answer_only` (38.6%). This is documented in the paper as a finding about instruction-following capability.

3. **Conda/texlive conflict.** The conda environment may shadow system texlive binaries, causing `pdflatex` to fail. Workaround: prepend `/usr/bin` to PATH or compile the paper outside conda.

4. **Large result files.** Some result files are 30+ MB because they store the full prompt and response for every item. The cache DB is ~160 MB. These are gitignored.

5. **GPU 4 runs hot.** Observed at 78C during 27B inference. Within safe limits but monitor if running extended sessions.

6. **The two exp2 result files for MedGemma-4B** (`20260129` and `20260210`) differ slightly in accuracy (47.6% vs 48.5%) because the newer run uses the updated multi-seed infrastructure and extended conditions. Use the `20260210` version as the primary result.

7. **Multi-seed experiments use 3 seeds** (42, 123, 456) for option order. The full seed list is [42, 123, 456, 789, 1337] but only 3 were planned for exp2 to save compute.

8. **PubMedQA labeled split** has only a `train` split (no validation/test). All 1,000 items in `pqa_labeled` are used.

---

## Contact

**Binesh Sadanandan** - bsada1@unh.newhaven.edu
**Vahid Behzadan** - vbehzadan@newhaven.edu
SAIL Lab, University of New Haven
