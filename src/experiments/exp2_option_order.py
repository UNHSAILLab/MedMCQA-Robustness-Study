"""Experiment 2: Option Order Sensitivity on MedMCQA."""

from typing import Dict, Any, List
import logging

from .base import BaseExperiment
from ..data.loaders import load_medmcqa
from ..data.schemas import MCQItem
from ..prompts.templates import MedMCQAPromptTemplate, PromptConfig, PromptStyle
from ..perturbations.option_shuffle import OptionShuffler, DistractorSwapper
from ..evaluation.metrics import MCQMetrics, parse_mcq_answer

logger = logging.getLogger(__name__)


class OptionOrderExperiment(BaseExperiment):
    """Experiment 2: Option order sensitivity analysis.

    Measures:
    - Accuracy drop when options are shuffled
    - Flip rate (how often predictions change)
    - Consistency analysis (correct->wrong, wrong->correct, etc.)
    - Position bias analysis
    """

    PERTURBATION_TYPES = [
        'random_shuffle',
        'rotate_1',
        'rotate_2',
        'distractor_swap'
    ]

    @property
    def name(self) -> str:
        return "exp2_option_order"

    def run(self) -> Dict[str, Any]:
        """Run original and perturbed versions."""
        # Load data (use validation split which has answers)
        limit = self.config.get('dataset', {}).get('limit')
        seed = self.config.get('seed', 42)
        data = load_medmcqa(split="validation", limit=limit)
        logger.info(f"Loaded {len(data)} MedMCQA items")

        results = {
            'original': None,
            'perturbations': {},
            'items': [
                {
                    'id': item.id,
                    'correct_answer': item.correct_answer,
                    'subject': item.subject
                }
                for item in data
            ]
        }

        template = MedMCQAPromptTemplate()
        prompt_config = PromptConfig(style=PromptStyle.ZERO_SHOT_DIRECT)

        def build_prompt(item: MCQItem) -> str:
            return template.format(item, prompt_config)

        # Run original (canonical) order
        logger.info("Running original order...")
        responses = self.run_inference(
            data,
            build_prompt,
            parse_fn=parse_mcq_answer
        )
        results['original'] = [r.model_dump() for r in responses]

        # Run each perturbation type
        shuffler = OptionShuffler(seed=seed)
        swapper = DistractorSwapper()

        for pert_type in self.PERTURBATION_TYPES:
            logger.info(f"Running perturbation: {pert_type}")

            # Generate perturbed data
            if pert_type == 'random_shuffle':
                perturbed_data = [shuffler.shuffle_random(item)[0] for item in data]
            elif pert_type == 'rotate_1':
                perturbed_data = [shuffler.rotate_options(item, 1)[0] for item in data]
            elif pert_type == 'rotate_2':
                perturbed_data = [shuffler.rotate_options(item, 2)[0] for item in data]
            elif pert_type == 'distractor_swap':
                perturbed_data = []
                for item in data:
                    swaps = swapper.get_all_distractor_swaps(item)
                    # Use first swap variant
                    perturbed_data.append(swaps[0][0] if swaps else item)

            # Run inference on perturbed data
            pert_responses = self.run_inference(
                perturbed_data,
                build_prompt,
                parse_fn=parse_mcq_answer
            )

            # Store perturbed results with updated correct answers
            results['perturbations'][pert_type] = {
                'responses': [r.model_dump() for r in pert_responses],
                'correct_answers': [item.correct_answer for item in perturbed_data]
            }

        return results

    def analyze(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute robustness metrics."""
        original_responses = results['original']
        items = results['items']

        original_preds = [r['parsed_answer'] or 'UNKNOWN' for r in original_responses]
        original_labels = [item['correct_answer'] for item in items]

        metrics = {
            'original': {
                'accuracy': MCQMetrics.accuracy(original_preds, original_labels),
                'accuracy_ci': MCQMetrics.accuracy_with_ci(original_preds, original_labels),
                'position_bias': MCQMetrics.position_bias(original_preds, original_labels)
            },
            'perturbations': {}
        }

        for pert_type, pert_data in results['perturbations'].items():
            pert_responses = pert_data['responses']
            pert_labels = pert_data['correct_answers']

            pert_preds = [r['parsed_answer'] or 'UNKNOWN' for r in pert_responses]

            # Compute robustness metrics
            robustness = MCQMetrics.robustness_score(
                original_preds, pert_preds, original_labels
            )

            # Add perturbed accuracy (against perturbed labels)
            pert_acc = MCQMetrics.accuracy(pert_preds, pert_labels)

            metrics['perturbations'][pert_type] = {
                'perturbed_accuracy': pert_acc,
                **robustness,
                'position_bias': MCQMetrics.position_bias(pert_preds, pert_labels)
            }

        # Summary statistics
        flip_rates = [m['flip_rate'] for m in metrics['perturbations'].values()]
        acc_drops = [m['accuracy_drop'] for m in metrics['perturbations'].values()]

        metrics['summary'] = {
            'mean_flip_rate': sum(flip_rates) / len(flip_rates) if flip_rates else 0,
            'max_flip_rate': max(flip_rates) if flip_rates else 0,
            'mean_accuracy_drop': sum(acc_drops) / len(acc_drops) if acc_drops else 0,
            'max_accuracy_drop': max(acc_drops) if acc_drops else 0
        }

        return metrics


def run_option_order_experiment(
    model,
    config: Dict = None,
    limit: int = None,
    output_dir: str = "outputs/results"
) -> Dict[str, Any]:
    """Convenience function to run option order experiment.

    Args:
        model: Model instance
        config: Configuration dict
        limit: Limit number of items (for testing)
        output_dir: Output directory

    Returns:
        Experiment results and metrics
    """
    if config is None:
        config = {}

    if limit:
        config.setdefault('dataset', {})['limit'] = limit

    experiment = OptionOrderExperiment(
        model=model,
        config=config,
        output_dir=output_dir
    )

    return experiment.full_run()
