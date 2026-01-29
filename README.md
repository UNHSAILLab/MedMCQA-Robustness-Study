# MedMCQA Robustness Study

A comprehensive research framework for evaluating the robustness of large language models on medical multiple-choice question (MCQ) tasks. This project systematically tests how prompt configurations, option ordering, evidence context, and self-consistency voting impact model performance on medical knowledge benchmarks.

## Overview

This study focuses on Google's MedGemma models (4B and 27B variants) and includes four main experiments:

1. **Prompt Recipe Ablation** - Evaluates different prompt styles (zero-shot, few-shot, chain-of-thought)
2. **Option Order Sensitivity** - Measures position bias when answer options are shuffled
3. **Evidence Conditioning** - Tests context dependency using PubMedQA with varying context levels
4. **Self-Consistency Voting** - Analyzes majority voting with confidence calibration

## Project Structure

```
MedMCQA-Robustness-Study/
├── main.py                     # Main CLI entry point
├── requirements.txt            # Python dependencies
├── configs/
│   └── base.yaml               # Base configuration
├── src/
│   ├── data/                   # Data loading and schemas
│   │   ├── loaders.py          # Dataset loaders (MedMCQA, PubMedQA)
│   │   └── schemas.py          # Pydantic data models
│   ├── models/                 # Model implementations
│   │   ├── base.py             # Abstract base model class
│   │   └── medgemma.py         # MedGemma wrapper
│   ├── experiments/            # Experiment implementations
│   │   ├── base.py             # Base experiment class
│   │   ├── exp1_prompt_ablation.py
│   │   ├── exp2_option_order.py
│   │   ├── exp3_evidence_conditioning.py
│   │   └── exp4_self_consistency.py
│   ├── prompts/                # Prompt templates
│   │   ├── templates.py        # Template definitions
│   │   └── few_shot_examples.py
│   ├── perturbations/          # Data perturbation modules
│   │   ├── option_shuffle.py   # Option order shuffling
│   │   └── context_truncation.py
│   ├── evaluation/             # Metrics and visualization
│   │   ├── metrics.py          # Accuracy metrics
│   │   ├── calibration.py      # Calibration metrics (ECE, MCE)
│   │   └── visualization.py    # Plotting utilities
│   └── utils/                  # Utilities
│       ├── caching.py          # SQLite response caching
│       └── checkpointing.py    # Experiment checkpointing
├── scripts/
│   └── run_experiment.py       # Experiment runner script
├── tests/                      # Test suite
└── outputs/                    # Results directory
    ├── results/
    ├── cache/
    ├── checkpoints/
    └── figures/
```

## Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended for model inference)
- HuggingFace account with access to MedGemma models

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd MedMCQA-Robustness-Study
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure HuggingFace authentication:
```bash
huggingface-cli login
```

## Usage

### Quick Start

Verify your setup:
```bash
python main.py --test
python main.py --verify
```

### Running Experiments

Run a specific experiment:
```bash
python main.py -e prompt_ablation -m 4b
```

Run all experiments with limited data (for testing):
```bash
python main.py -e all -m 4b -l 100
```

Use custom configuration:
```bash
python main.py -c configs/base.yaml -o outputs/results
```

### Command-Line Options

| Option | Description |
|--------|-------------|
| `-e, --experiment` | Experiment to run: `prompt_ablation`, `option_order`, `evidence_conditioning`, `self_consistency`, `all` |
| `-m, --model` | Model variant: `4b`, `27b`, `27b-8bit` |
| `-l, --limit` | Limit number of items to process (useful for testing) |
| `-c, --config` | Path to configuration file |
| `-o, --output-dir` | Output directory for results |
| `--test` | Quick setup verification |
| `--verify` | Check all dependencies |

## Experiments

### Experiment 1: Prompt Recipe Ablation

Tests five prompt styles on MedMCQA:
- Zero-shot direct
- Zero-shot chain-of-thought (CoT)
- Few-shot (3 examples) direct
- Few-shot (3 examples) CoT
- Answer-only (minimal prompt)

### Experiment 2: Option Order Sensitivity

Measures position bias using:
- Random shuffle perturbations
- Rotation-based perturbations
- Distractor swapping

Tracks accuracy drop and answer flip rates.

### Experiment 3: Evidence Conditioning

Tests on PubMedQA with varying context:
- Question-only mode
- Full context
- Truncated context (25%, 50%)
- Section-specific (background vs. results)

### Experiment 4: Self-Consistency Voting

Implements majority voting with temperature sampling:
- Samples N answers per question
- Majority vote for final prediction
- Confidence as vote proportion
- Expected Calibration Error (ECE) analysis

## Datasets

- **MedMCQA**: Medical entrance exam questions with 4 answer options (validation split: 4,183 items)
- **PubMedQA**: Biomedical literature QA with yes/no/maybe answers (labeled subset)

Both datasets are loaded from HuggingFace Hub.

## Configuration

The `configs/base.yaml` file contains all configurable parameters:

```yaml
seed: 42

paths:
  cache_dir: ./outputs/cache
  results_dir: ./outputs/results
  figures_dir: ./outputs/figures
  checkpoints_dir: ./outputs/checkpoints

inference:
  batch_size: 4
  checkpoint_interval: 100
  use_cache: true

generation:
  max_new_tokens: 256

models:
  medgemma_4b:
    model_id: google/medgemma-4b-it
    quantization: null
  medgemma_27b:
    model_id: google/medgemma-27b-text-it
    quantization: 4bit
```

## Key Features

- **Caching**: SQLite-based response caching to avoid redundant inference
- **Checkpointing**: Resumable experiments with periodic progress saves
- **Batch Processing**: Efficient GPU utilization with configurable batch sizes
- **Comprehensive Metrics**: Accuracy, calibration (ECE/MCE), and confidence analysis
- **Modular Design**: Easily extensible with new models, experiments, or datasets

## Dependencies

- PyTorch >= 2.0
- Transformers >= 4.40
- Datasets (HuggingFace)
- BitsAndBytes (quantization)
- Accelerate (distributed inference)
- Pydantic >= 2.0
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn

## Output Format

Results are saved as JSON files with the naming convention:
```
experiment_{name}_{model}_{timestamp}.json
```

Each result file contains:
- Experiment configuration
- Per-condition metrics
- Individual predictions
- Calibration analysis (where applicable)

## License

[Add license information]

## Citation

[Add citation information if applicable]

## Acknowledgments

This project uses the MedGemma models from Google and datasets from HuggingFace Hub.
