"""Experiment 1: Prompt Recipe Ablation on MedMCQA."""

from typing import Dict, Any, List
import logging

from .base import BaseExperiment
from ..data.loaders import load_medmcqa
from ..data.schemas import MCQItem
from ..prompts.templates import MedMCQAPromptTemplate, PromptConfig, PromptStyle
from ..evaluation.metrics import MCQMetrics, parse_mcq_answer

logger = logging.getLogger(__name__)


class PromptAblationExperiment(BaseExperiment):
    """Experiment 1: Prompt recipe ablation on MedMCQA.

    Tests different prompt configurations:
    - Zero-shot direct
    - Zero-shot CoT
    - Few-shot (3 examples) direct
    - Few-shot (3 examples) CoT
    - Answer-only (minimal prompt)
    """

    CONDITIONS = [
        {
            'name': 'zero_shot_direct',
            'style': PromptStyle.ZERO_SHOT_DIRECT,
            'num_examples': 0
        },
        {
            'name': 'zero_shot_cot',
            'style': PromptStyle.ZERO_SHOT_COT,
            'num_examples': 0
        },
        {
            'name': 'few_shot_3_direct',
            'style': PromptStyle.FEW_SHOT_DIRECT,
            'num_examples': 3
        },
        {
            'name': 'few_shot_3_cot',
            'style': PromptStyle.FEW_SHOT_COT,
            'num_examples': 3
        },
        {
            'name': 'answer_only',
            'style': PromptStyle.ANSWER_ONLY,
            'num_examples': 0
        },
    ]

    @property
    def name(self) -> str:
        return "exp1_prompt_ablation"

    def run(self) -> Dict[str, Any]:
        """Run all prompt conditions on MedMCQA test set."""
        # Load data (use validation split which has answers)
        limit = self.config.get('dataset', {}).get('limit')
        data = load_medmcqa(split="validation", limit=limit)
        logger.info(f"Loaded {len(data)} MedMCQA items")

        results = {}
        template = MedMCQAPromptTemplate()

        for condition in self.CONDITIONS:
            cond_name = condition['name']
            logger.info(f"Running condition: {cond_name}")

            # Build prompt config
            prompt_config = PromptConfig(
                style=condition['style'],
                num_examples=condition['num_examples']
            )

            # Build prompt function
            def build_prompt(item: MCQItem, cfg=prompt_config) -> str:
                return template.format(item, cfg)

            # Run inference
            responses = self.run_inference(
                data,
                build_prompt,
                parse_fn=parse_mcq_answer
            )

            # Store results with item data
            results[cond_name] = {
                'responses': [r.model_dump() for r in responses],
                'items': [
                    {
                        'id': item.id,
                        'correct_answer': item.correct_answer,
                        'subject': item.subject
                    }
                    for item in data
                ]
            }

        return results

    def analyze(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute accuracy for each condition with subject breakdown."""
        metrics = {}

        for cond_name, cond_data in results.items():
            responses = cond_data['responses']
            items = cond_data['items']

            predictions = [r['parsed_answer'] or 'UNKNOWN' for r in responses]
            labels = [item['correct_answer'] for item in items]
            subjects = [item['subject'] for item in items]

            # Overall accuracy
            acc, lower, upper = MCQMetrics.accuracy_with_ci(predictions, labels)

            # Per-subject breakdown
            by_subject = MCQMetrics.accuracy_by_subject(predictions, labels, subjects)

            # Position bias
            position_bias = MCQMetrics.position_bias(predictions, labels)

            metrics[cond_name] = {
                'accuracy': acc,
                'accuracy_ci': (lower, upper),
                'n_samples': len(predictions),
                'by_subject': by_subject,
                'position_bias': position_bias
            }

        # Compute gains
        if 'zero_shot_direct' in metrics and 'zero_shot_cot' in metrics:
            metrics['cot_gain'] = (
                metrics['zero_shot_cot']['accuracy'] -
                metrics['zero_shot_direct']['accuracy']
            )

        if 'zero_shot_direct' in metrics and 'few_shot_3_direct' in metrics:
            metrics['few_shot_gain'] = (
                metrics['few_shot_3_direct']['accuracy'] -
                metrics['zero_shot_direct']['accuracy']
            )

        return metrics


def run_prompt_ablation(
    model,
    config: Dict = None,
    limit: int = None,
    output_dir: str = "outputs/results"
) -> Dict[str, Any]:
    """Convenience function to run prompt ablation experiment.

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

    experiment = PromptAblationExperiment(
        model=model,
        config=config,
        output_dir=output_dir
    )

    return experiment.full_run()
