"""Experiment 3: Evidence Conditioning on PubMedQA."""

from typing import Dict, Any, List
import logging

from .base import BaseExperiment
from ..data.loaders import load_pubmedqa
from ..data.schemas import PubMedQAItem
from ..prompts.templates import PubMedQAPromptTemplate
from ..perturbations.context_truncation import ContextTruncator
from ..evaluation.metrics import PubMedQAMetrics, parse_pubmedqa_answer

logger = logging.getLogger(__name__)


class EvidenceConditioningExperiment(BaseExperiment):
    """Experiment 3: Evidence conditioning on PubMedQA.

    Tests different context conditions:
    - Question only (no context)
    - Question + full context
    - Question + truncated context (50%)
    - Question + BACKGROUND only
    - Question + RESULTS only
    """

    CONTEXT_CONDITIONS = [
        {'name': 'question_only', 'mode': 'none'},
        {'name': 'full_context', 'mode': 'full'},
        {'name': 'truncated_50', 'mode': 'truncated', 'ratio': 0.5},
        {'name': 'truncated_25', 'mode': 'truncated', 'ratio': 0.25},
        {'name': 'background_only', 'mode': 'sections', 'sections': ['BACKGROUND', 'OBJECTIVE', 'OBJECTIVES']},
        {'name': 'results_only', 'mode': 'sections', 'sections': ['RESULTS', 'CONCLUSIONS', 'FINDINGS']},
        {'name': 'truncated_back_50', 'mode': 'truncated_back', 'ratio': 0.5},
        {'name': 'truncated_middle_50', 'mode': 'truncated_middle', 'ratio': 0.5},
        {'name': 'sentence_trunc_50', 'mode': 'sentence_truncated', 'ratio': 0.5},
        {'name': 'salient_top5', 'mode': 'salient', 'top_k': 5},
        {'name': 'salient_top3', 'mode': 'salient', 'top_k': 3},
    ]

    @property
    def name(self) -> str:
        return "exp3_evidence_conditioning"

    def run(self) -> Dict[str, Any]:
        """Run all context conditions on PubMedQA labeled set."""
        # Load data
        limit = self.config.get('dataset', {}).get('limit')
        data = load_pubmedqa(limit=limit)
        logger.info(f"Loaded {len(data)} PubMedQA items")

        results = {
            'items': [
                {
                    'id': item.id,
                    'correct_answer': item.correct_answer,
                    'section_labels': item.section_labels
                }
                for item in data
            ]
        }

        template = PubMedQAPromptTemplate()
        truncator = ContextTruncator()

        for condition in self.CONTEXT_CONDITIONS:
            cond_name = condition['name']
            logger.info(f"Running condition: {cond_name}")

            # Apply context manipulation
            if condition['mode'] == 'none':
                processed_data = [truncator.remove_context(item) for item in data]
            elif condition['mode'] == 'full':
                processed_data = data
            elif condition['mode'] == 'truncated':
                ratio = condition.get('ratio', 0.5)
                processed_data = [truncator.truncate_by_ratio(item, ratio) for item in data]
            elif condition['mode'] == 'sections':
                sections = condition.get('sections', [])
                processed_data = [truncator.keep_sections(item, sections) for item in data]
            elif condition['mode'] == 'truncated_back':
                ratio = condition.get('ratio', 0.5)
                processed_data = [truncator.truncate_back(item, ratio) for item in data]
            elif condition['mode'] == 'truncated_middle':
                ratio = condition.get('ratio', 0.5)
                processed_data = [truncator.truncate_middle(item, ratio) for item in data]
            elif condition['mode'] == 'sentence_truncated':
                ratio = condition.get('ratio', 0.5)
                processed_data = [truncator.truncate_by_sentences(item, ratio) for item in data]
            elif condition['mode'] == 'salient':
                top_k = condition.get('top_k', 5)
                processed_data = [truncator.extract_salient_sentences(item, top_k) for item in data]
            else:
                processed_data = data

            # Build prompt function based on condition
            context_mode = condition['mode']
            if context_mode in ('sections', 'truncated_back', 'truncated_middle',
                                'sentence_truncated', 'salient'):
                context_mode = 'full'  # context already manipulated

            def build_prompt(item: PubMedQAItem, mode=context_mode) -> str:
                return template.format(item, context_mode=mode)

            # Run inference
            responses = self.run_inference(
                processed_data,
                build_prompt,
                parse_fn=parse_pubmedqa_answer
            )

            results[cond_name] = {
                'responses': [r.model_dump() for r in responses]
            }

        return results

    def analyze(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how context affects performance."""
        items = results['items']
        labels = [item['correct_answer'] for item in items]

        metrics = {}

        for cond_name in [c['name'] for c in self.CONTEXT_CONDITIONS]:
            if cond_name not in results:
                continue

            responses = results[cond_name]['responses']
            predictions = [r['parsed_answer'] or 'unknown' for r in responses]

            # Overall accuracy
            accuracy = PubMedQAMetrics.accuracy(predictions, labels)

            # By answer type
            by_class = PubMedQAMetrics.accuracy_by_class(predictions, labels)

            metrics[cond_name] = {
                'accuracy': accuracy,
                'by_answer_type': by_class,
                'n_samples': len(predictions)
            }

        # Compute context importance metrics
        if 'full_context' in metrics and 'question_only' in metrics:
            context_importance = PubMedQAMetrics.context_importance(
                metrics['full_context']['accuracy'],
                metrics['question_only']['accuracy']
            )
            metrics['context_importance'] = context_importance

        # Information loss from truncation
        if 'full_context' in metrics and 'truncated_50' in metrics:
            metrics['truncation_loss_50'] = (
                metrics['full_context']['accuracy'] -
                metrics['truncated_50']['accuracy']
            )

        if 'full_context' in metrics and 'truncated_25' in metrics:
            metrics['truncation_loss_25'] = (
                metrics['full_context']['accuracy'] -
                metrics['truncated_25']['accuracy']
            )

        # Section importance
        if 'full_context' in metrics:
            full_acc = metrics['full_context']['accuracy']
            if 'background_only' in metrics:
                metrics['background_sufficiency'] = (
                    metrics['background_only']['accuracy'] / full_acc
                    if full_acc > 0 else 0
                )
            if 'results_only' in metrics:
                metrics['results_sufficiency'] = (
                    metrics['results_only']['accuracy'] / full_acc
                    if full_acc > 0 else 0
                )

        return metrics


def run_evidence_conditioning(
    model,
    config: Dict = None,
    limit: int = None,
    output_dir: str = "outputs/results"
) -> Dict[str, Any]:
    """Convenience function to run evidence conditioning experiment.

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

    experiment = EvidenceConditioningExperiment(
        model=model,
        config=config,
        output_dir=output_dir
    )

    return experiment.full_run()
