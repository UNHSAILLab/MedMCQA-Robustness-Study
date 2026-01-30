#!/usr/bin/env python3
"""
Medical MCQ Robustness Study

A comprehensive robustness study combining 4 experiments on medical MCQ tasks:
1. Prompt Recipe Ablation (MedMCQA)
2. Option Order Sensitivity (MedMCQA)
3. Evidence Conditioning (PubMedQA)
4. Self-Consistency Voting (Both datasets)

Usage:
    # Run a specific experiment
    python main.py --experiment prompt_ablation --model 4b

    # Run with limited data for testing
    python main.py --experiment option_order --model 4b --limit 100

    # Run all experiments
    python main.py --experiment all --model 4b
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('experiment.log')
    ]
)
logger = logging.getLogger(__name__)


def verify_setup():
    """Verify that all dependencies are installed."""
    missing = []

    try:
        import torch
    except ImportError:
        missing.append('torch')

    try:
        import transformers
    except ImportError:
        missing.append('transformers')

    try:
        import datasets
    except ImportError:
        missing.append('datasets')

    try:
        import bitsandbytes
    except ImportError:
        missing.append('bitsandbytes')

    if missing:
        print("Missing dependencies:", missing)
        print("Install with: pip install -r requirements.txt")
        return False

    return True


def quick_test():
    """Run a quick test to verify everything works."""
    print("Running quick test...")

    # Test data loading
    print("1. Testing data loaders...")
    from src.data.loaders import load_medmcqa, load_pubmedqa

    medmcqa = load_medmcqa(limit=5)
    print(f"   MedMCQA: loaded {len(medmcqa)} items")
    print(f"   Sample question: {medmcqa[0].question[:80]}...")

    pubmedqa = load_pubmedqa(limit=5)
    print(f"   PubMedQA: loaded {len(pubmedqa)} items")
    print(f"   Sample question: {pubmedqa[0].question[:80]}...")

    # Test prompt templates
    print("\n2. Testing prompt templates...")
    from src.prompts.templates import (
        MedMCQAPromptTemplate, PubMedQAPromptTemplate,
        PromptConfig, PromptStyle
    )

    mcq_template = MedMCQAPromptTemplate()
    config = PromptConfig(style=PromptStyle.ZERO_SHOT_COT)
    prompt = mcq_template.format(medmcqa[0], config)
    print(f"   Generated prompt length: {len(prompt)} chars")

    # Test perturbations
    print("\n3. Testing perturbations...")
    from src.perturbations.option_shuffle import OptionShuffler

    shuffler = OptionShuffler(seed=42)
    perturbed, mapping = shuffler.shuffle_random(medmcqa[0])
    print(f"   Original answer: {medmcqa[0].correct_answer}")
    print(f"   Perturbed answer: {perturbed.correct_answer}")
    print(f"   Mapping: {mapping}")

    # Test metrics
    print("\n4. Testing metrics...")
    from src.evaluation.metrics import MCQMetrics

    preds = ['A', 'B', 'C', 'A']
    labels = ['A', 'B', 'A', 'A']
    acc = MCQMetrics.accuracy(preds, labels)
    print(f"   Test accuracy: {acc:.1%}")

    print("\n" + "=" * 40)
    print("Quick test PASSED!")
    print("=" * 40)


def main():
    parser = argparse.ArgumentParser(
        description="Medical MCQ Robustness Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--experiment', '-e',
        choices=['prompt_ablation', 'option_order', 'evidence_conditioning',
                 'self_consistency', 'all'],
        help="Experiment to run"
    )

    parser.add_argument(
        '--model', '-m',
        choices=['4b', '27b', '27b-4bit', '27b-8bit'],
        default='4b',
        help="Model variant (default: 4b)"
    )

    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help="Limit number of items (for testing)"
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/base.yaml',
        help="Path to config file"
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='outputs/results',
        help="Output directory"
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help="Run quick test to verify setup"
    )

    parser.add_argument(
        '--verify',
        action='store_true',
        help="Verify dependencies are installed"
    )

    args = parser.parse_args()

    if args.verify:
        if verify_setup():
            print("All dependencies installed correctly!")
        sys.exit(0)

    if args.test:
        if not verify_setup():
            sys.exit(1)
        quick_test()
        sys.exit(0)

    if args.experiment is None:
        parser.print_help()
        print("\nExamples:")
        print("  python main.py --test                    # Run quick test")
        print("  python main.py -e prompt_ablation -m 4b  # Run experiment")
        print("  python main.py -e all -m 4b -l 100       # All experiments, limited data")
        sys.exit(0)

    # Run experiment via the CLI script
    from scripts.run_experiment import run_experiment, load_config

    config = load_config(args.config)

    if args.experiment == 'all':
        experiments = ['prompt_ablation', 'option_order',
                      'evidence_conditioning', 'self_consistency']
    else:
        experiments = [args.experiment]

    for exp_name in experiments:
        try:
            run_experiment(
                experiment_name=exp_name,
                model_name=args.model,
                config=config,
                limit=args.limit,
                output_dir=args.output_dir
            )
        except Exception as e:
            logger.exception(f"Experiment {exp_name} failed")
            continue


if __name__ == '__main__':
    main()
