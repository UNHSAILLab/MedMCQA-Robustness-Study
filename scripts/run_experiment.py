#!/usr/bin/env python3
"""CLI runner for medical MCQ robustness experiments."""

import argparse
import logging
import sys
import os
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.medgemma import MedGemmaModel
from src.models.medical_llms import MedicalLLM
from src.experiments.exp1_prompt_ablation import PromptAblationExperiment
from src.experiments.exp2_option_order import OptionOrderExperiment
from src.experiments.exp3_evidence_conditioning import EvidenceConditioningExperiment
from src.experiments.exp4_self_consistency import SelfConsistencyExperiment
from src.experiments.exp5_robust_baselines import (
    CoTSelfConsistencyExperiment,
    PermutationVoteExperiment,
    ClozeScoreExperiment,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


EXPERIMENTS = {
    'prompt_ablation': PromptAblationExperiment,
    'option_order': OptionOrderExperiment,
    'evidence_conditioning': EvidenceConditioningExperiment,
    'self_consistency': SelfConsistencyExperiment,
    'cot_self_consistency': CoTSelfConsistencyExperiment,
    'permutation_vote': PermutationVoteExperiment,
    'cloze_score': ClozeScoreExperiment,
    'all': None  # Special case
}

MODELS = {
    # MedGemma variants
    '4b': {'type': 'medgemma', 'variant': '4b', 'quantization': None},
    '27b': {'type': 'medgemma', 'variant': '27b', 'quantization': None},  # Full precision
    '27b-4bit': {'type': 'medgemma', 'variant': '27b', 'quantization': '4bit'},
    '27b-8bit': {'type': 'medgemma', 'variant': '27b', 'quantization': '8bit'},
    # Medical LLMs
    'biomistral-7b': {'type': 'medical_llm', 'model_name': 'biomistral-7b', 'quantization': None},
    'biomistral-7b-4bit': {'type': 'medical_llm', 'model_name': 'biomistral-7b', 'quantization': '4bit'},
    'meditron-7b': {'type': 'medical_llm', 'model_name': 'meditron-7b', 'quantization': None},
    'meditron-7b-4bit': {'type': 'medical_llm', 'model_name': 'meditron-7b', 'quantization': '4bit'},
}


def load_config(config_path: str = None) -> dict:
    """Load configuration from file or use defaults."""
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)

    # Default config — must stay in sync with configs/base.yaml
    return {
        'seed': 42,
        'seeds': [42, 123, 456, 789, 1337],
        'inference': {
            'batch_size': 4,
            'checkpoint_interval': 100,
            'use_cache': True
        },
        'generation': {
            'max_new_tokens': 256,
            'temperature': 0.0,
            'top_p': 1.0,
            'top_k': 0,
            'do_sample': False
        },
        'self_consistency': {
            'temperature': 0.7,
            'top_p': 0.95,
            'top_k': 50,
            'do_sample': True,
            'max_new_tokens': 256,
            'sample_counts': [1, 3, 5, 10]
        }
    }


def run_experiment(
    experiment_name: str,
    model_name: str,
    config: dict,
    limit: int = None,
    output_dir: str = "outputs/results"
):
    """Run a single experiment."""
    logger.info(f"Running experiment: {experiment_name}")
    logger.info(f"Model: {model_name}")

    # Load model
    model_config = MODELS.get(model_name)
    if model_config is None:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    # Copy config to avoid modifying the original
    model_config = model_config.copy()
    model_type = model_config.pop('type', 'medgemma')

    if model_type == 'medgemma':
        model = MedGemmaModel(**model_config)
    elif model_type == 'medical_llm':
        model = MedicalLLM(**model_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load()

    # Set limit in config
    if limit:
        config.setdefault('dataset', {})['limit'] = limit

    # Get experiment class
    exp_class = EXPERIMENTS.get(experiment_name)
    if exp_class is None:
        raise ValueError(f"Unknown experiment: {experiment_name}")

    # Run experiment
    experiment = exp_class(
        model=model,
        config=config,
        output_dir=output_dir
    )

    result = experiment.full_run()

    logger.info(f"Experiment completed. Results saved to: {result['output_path']}")

    # Print summary metrics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    metrics = result['metrics']
    if experiment_name == 'prompt_ablation':
        print("\nAccuracy by prompt condition:")
        for cond, data in metrics.items():
            if isinstance(data, dict) and 'accuracy' in data:
                print(f"  {cond}: {data['accuracy']:.1%}")

    elif experiment_name == 'option_order':
        print("\nOriginal accuracy:", f"{metrics['original']['accuracy']:.1%}")
        print("\nPerturbation effects:")
        for pert, data in metrics.get('perturbations', {}).items():
            print(f"  {pert}:")
            print(f"    Accuracy drop: {data.get('accuracy_drop', 0):.1%}")
            print(f"    Flip rate: {data.get('flip_rate', 0):.1%}")

    elif experiment_name == 'evidence_conditioning':
        print("\nAccuracy by context condition:")
        for cond, data in metrics.items():
            if isinstance(data, dict) and 'accuracy' in data:
                print(f"  {cond}: {data['accuracy']:.1%}")

    elif experiment_name == 'self_consistency':
        for dataset in ['medmcqa', 'pubmedqa']:
            if dataset in metrics:
                print(f"\n{dataset.upper()}:")
                for key, data in metrics[dataset].items():
                    if key.startswith('n_'):
                        print(f"  N={data.get('n_samples', key)}: "
                              f"acc={data.get('accuracy', 0):.1%}, "
                              f"ECE={data.get('ece', 0):.3f}")

    elif experiment_name == 'cot_self_consistency':
        print("\nCoT Self-Consistency (MedMCQA):")
        for key, data in metrics.items():
            if key.startswith('n_'):
                print(f"  N={data.get('n_samples', key)}: "
                      f"acc={data.get('accuracy', 0):.1%}, "
                      f"confidence={data.get('mean_confidence', 0):.3f}")

    elif experiment_name == 'permutation_vote':
        print("\nPermutation Vote (MedMCQA):")
        print(f"  Aggregated accuracy: {metrics.get('aggregated_accuracy', 0):.1%}")
        print(f"  Per-perm accuracy mean: {metrics.get('per_permutation_accuracy_mean', 0):.1%}")
        print(f"  Per-perm accuracy std:  {metrics.get('per_permutation_accuracy_std', 0):.3f}")
        print(f"  Mean agreement rate:    {metrics.get('mean_agreement_rate', 0):.3f}")

    elif experiment_name == 'cloze_score':
        print("\nCloze Score (MedMCQA):")
        print(f"  Accuracy: {metrics.get('accuracy', 0):.1%}")
        print(f"  Mean logprob margin: {metrics.get('mean_logprob_margin', 0):.3f}")

    print("=" * 60)

    # Unload model
    model.unload()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run medical MCQ robustness experiments"
    )

    parser.add_argument(
        'experiment',
        choices=list(EXPERIMENTS.keys()),
        help="Experiment to run"
    )

    parser.add_argument(
        '--model', '-m',
        choices=list(MODELS.keys()),
        default='4b',
        help="Model variant (default: 4b)"
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help="Path to config file"
    )

    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help="Limit number of items (for testing)"
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default="outputs/results",
        help="Output directory for results"
    )

    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=None,
        help="Random seed (overrides config seed)"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override seed if provided
    if args.seed is not None:
        config['seed'] = args.seed

    if args.experiment == 'all':
        # Run all experiments
        for exp_name in ['prompt_ablation', 'option_order', 'evidence_conditioning', 'self_consistency']:
            try:
                run_experiment(
                    experiment_name=exp_name,
                    model_name=args.model,
                    config=config,
                    limit=args.limit,
                    output_dir=args.output_dir
                )
            except Exception as e:
                logger.error(f"Experiment {exp_name} failed: {e}")
                continue
    else:
        run_experiment(
            experiment_name=args.experiment,
            model_name=args.model,
            config=config,
            limit=args.limit,
            output_dir=args.output_dir
        )


if __name__ == '__main__':
    main()
