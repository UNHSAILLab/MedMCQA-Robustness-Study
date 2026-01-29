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
from src.experiments.exp1_prompt_ablation import PromptAblationExperiment
from src.experiments.exp2_option_order import OptionOrderExperiment
from src.experiments.exp3_evidence_conditioning import EvidenceConditioningExperiment
from src.experiments.exp4_self_consistency import SelfConsistencyExperiment

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
    'all': None  # Special case
}

MODELS = {
    '4b': {'variant': '4b', 'quantization': None},
    '27b': {'variant': '27b', 'quantization': '4bit'},
    '27b-8bit': {'variant': '27b', 'quantization': '8bit'},
}


def load_config(config_path: str = None) -> dict:
    """Load configuration from file or use defaults."""
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)

    # Default config
    return {
        'seed': 42,
        'inference': {
            'batch_size': 4,
            'checkpoint_interval': 100
        },
        'generation': {
            'max_new_tokens': 256,
            'temperature': 0.0,
            'do_sample': False
        },
        'self_consistency': {
            'temperature': 0.7,
            'sample_counts': [1, 3, 5]
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

    model = MedGemmaModel(**model_config)
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

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

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
