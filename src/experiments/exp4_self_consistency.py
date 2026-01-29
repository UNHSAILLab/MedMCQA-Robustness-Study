"""Experiment 4: Self-Consistency Voting."""

from typing import Dict, Any, List
from collections import Counter
import logging
import numpy as np

from .base import BaseExperiment
from ..data.loaders import load_medmcqa, load_pubmedqa
from ..data.schemas import MCQItem, PubMedQAItem
from ..prompts.templates import (
    MedMCQAPromptTemplate, PubMedQAPromptTemplate,
    PromptConfig, PromptStyle
)
from ..evaluation.metrics import parse_mcq_answer, parse_pubmedqa_answer
from ..evaluation.calibration import (
    CalibrationMetrics, SelfConsistencyCalibration
)

logger = logging.getLogger(__name__)


class SelfConsistencyExperiment(BaseExperiment):
    """Experiment 4: Self-consistency voting with confidence calibration.

    Strategy:
    - Sample N answers with temperature > 0
    - Majority vote for final answer
    - Use vote proportion as confidence
    - Analyze calibration of this confidence
    """

    SAMPLE_COUNTS = [1, 3, 5, 10]  # Can extend to 20 if resources allow

    @property
    def name(self) -> str:
        return "exp4_self_consistency"

    def run(self) -> Dict[str, Any]:
        """Run self-consistency sampling on both datasets."""
        results = {
            'medmcqa': {},
            'pubmedqa': {}
        }

        # Configuration
        temperature = self.config.get('self_consistency', {}).get('temperature', 0.7)
        sample_counts = self.config.get('self_consistency', {}).get(
            'sample_counts', self.SAMPLE_COUNTS
        )
        limit = self.config.get('dataset', {}).get('limit')

        # MedMCQA (use validation split which has answers)
        logger.info("Running self-consistency on MedMCQA...")
        medmcqa_data = load_medmcqa(split="validation", limit=limit)
        medmcqa_template = MedMCQAPromptTemplate()
        medmcqa_config = PromptConfig(style=PromptStyle.ZERO_SHOT_COT)

        results['medmcqa']['items'] = [
            {'id': item.id, 'correct_answer': item.correct_answer}
            for item in medmcqa_data
        ]

        for n_samples in sample_counts:
            logger.info(f"MedMCQA with N={n_samples} samples")
            results['medmcqa'][f'n_{n_samples}'] = self._run_self_consistency(
                data=medmcqa_data,
                prompt_builder=lambda item: medmcqa_template.format(item, medmcqa_config),
                parse_fn=parse_mcq_answer,
                n_samples=n_samples,
                temperature=temperature
            )

        # PubMedQA
        logger.info("Running self-consistency on PubMedQA...")
        pubmedqa_data = load_pubmedqa(limit=limit)
        pubmedqa_template = PubMedQAPromptTemplate()

        results['pubmedqa']['items'] = [
            {'id': item.id, 'correct_answer': item.correct_answer}
            for item in pubmedqa_data
        ]

        for n_samples in sample_counts:
            logger.info(f"PubMedQA with N={n_samples} samples")
            results['pubmedqa'][f'n_{n_samples}'] = self._run_self_consistency(
                data=pubmedqa_data,
                prompt_builder=lambda item: pubmedqa_template.format(
                    item, context_mode="full", include_cot=True
                ),
                parse_fn=parse_pubmedqa_answer,
                n_samples=n_samples,
                temperature=temperature
            )

        return results

    def _run_self_consistency(
        self,
        data: List,
        prompt_builder,
        parse_fn,
        n_samples: int,
        temperature: float
    ) -> List[Dict]:
        """Run SC sampling for a dataset.

        Args:
            data: List of items
            prompt_builder: Function to build prompt
            parse_fn: Function to parse answers
            n_samples: Number of samples per item
            temperature: Sampling temperature

        Returns:
            List of result dicts per item
        """
        generation_config = {
            'temperature': temperature,
            'do_sample': True,
            'max_new_tokens': 512  # Longer for CoT
        }

        results = []

        from tqdm import tqdm
        for item in tqdm(data, desc=f"SC (N={n_samples})"):
            prompt = prompt_builder(item)

            # Generate N samples
            # Note: We generate one at a time since num_return_sequences
            # may not be supported with all configurations
            samples = []
            for _ in range(n_samples):
                outputs = self.model.generate([prompt], **generation_config)
                samples.append(outputs[0])

            # Parse all samples
            parsed = [parse_fn(s) for s in samples]

            # Majority vote
            vote_counts = Counter(parsed)
            majority_answer = vote_counts.most_common(1)[0][0]

            # Confidence = proportion of votes for majority
            confidence = vote_counts[majority_answer] / n_samples

            # Entropy
            entropy = SelfConsistencyCalibration.compute_vote_entropy(vote_counts)

            results.append({
                'item_id': item.id,
                'samples': samples,
                'parsed_samples': parsed,
                'vote_counts': dict(vote_counts),
                'majority_answer': majority_answer,
                'confidence': confidence,
                'entropy': entropy
            })

        return results

    def analyze(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze SC performance and calibration."""
        metrics = {}

        for dataset in ['medmcqa', 'pubmedqa']:
            if dataset not in results:
                continue

            dataset_results = results[dataset]
            items = dataset_results.get('items', [])
            labels = [item['correct_answer'] for item in items]

            metrics[dataset] = {}

            # Analyze each N value
            for key in dataset_results:
                if not key.startswith('n_'):
                    continue

                n_value = int(key.split('_')[1])
                sc_results = dataset_results[key]

                predictions = [r['majority_answer'] for r in sc_results]
                confidences = [r['confidence'] for r in sc_results]
                entropies = [r['entropy'] for r in sc_results]

                # Accuracy
                correct = sum(p == l for p, l in zip(predictions, labels))
                accuracy = correct / len(labels) if labels else 0

                # Calibration
                ece, cal_data = CalibrationMetrics.compute_ece(
                    confidences, predictions, labels
                )
                brier = CalibrationMetrics.compute_brier_score(
                    confidences, predictions, labels
                )
                overconf = CalibrationMetrics.compute_overconfidence(
                    confidences, predictions, labels
                )

                metrics[dataset][key] = {
                    'n_samples': n_value,
                    'accuracy': accuracy,
                    'ece': ece,
                    'brier_score': brier,
                    'mean_confidence': float(np.mean(confidences)),
                    'std_confidence': float(np.std(confidences)),
                    'mean_entropy': float(np.mean(entropies)),
                    'overconfidence_metrics': overconf,
                    'calibration_data': cal_data
                }

            # Compute gains from self-consistency
            if 'n_1' in metrics[dataset] and 'n_5' in metrics[dataset]:
                metrics[dataset]['accuracy_gain_1_to_5'] = (
                    metrics[dataset]['n_5']['accuracy'] -
                    metrics[dataset]['n_1']['accuracy']
                )

            if 'n_1' in metrics[dataset] and 'n_10' in metrics[dataset]:
                metrics[dataset]['accuracy_gain_1_to_10'] = (
                    metrics[dataset]['n_10']['accuracy'] -
                    metrics[dataset]['n_1']['accuracy']
                )

        return metrics


def run_self_consistency(
    model,
    config: Dict = None,
    limit: int = None,
    sample_counts: List[int] = None,
    output_dir: str = "outputs/results"
) -> Dict[str, Any]:
    """Convenience function to run self-consistency experiment.

    Args:
        model: Model instance
        config: Configuration dict
        limit: Limit number of items (for testing)
        sample_counts: List of N values to test
        output_dir: Output directory

    Returns:
        Experiment results and metrics
    """
    if config is None:
        config = {}

    if limit:
        config.setdefault('dataset', {})['limit'] = limit

    if sample_counts:
        config.setdefault('self_consistency', {})['sample_counts'] = sample_counts

    experiment = SelfConsistencyExperiment(
        model=model,
        config=config,
        output_dir=output_dir
    )

    return experiment.full_run()
