"""Evaluation metrics for medical MCQ tasks."""

from typing import List, Dict, Tuple, Any, Optional
from collections import Counter
import numpy as np


class MCQMetrics:
    """Metrics for multiple choice question evaluation."""

    @staticmethod
    def accuracy(predictions: List[str], labels: List[str]) -> float:
        """Overall accuracy.

        Args:
            predictions: List of predicted answers
            labels: List of correct answers

        Returns:
            Accuracy as float (0-1)
        """
        if not predictions:
            return 0.0
        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        return correct / len(predictions)

    @staticmethod
    def accuracy_with_ci(
        predictions: List[str],
        labels: List[str],
        confidence: float = 0.95,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float, float]:
        """Accuracy with bootstrap confidence interval.

        Args:
            predictions: List of predicted answers
            labels: List of correct answers
            confidence: Confidence level (default 0.95)
            n_bootstrap: Number of bootstrap samples

        Returns:
            Tuple of (accuracy, lower_bound, upper_bound)
        """
        n = len(predictions)
        if n == 0:
            return 0.0, 0.0, 0.0

        correct = np.array([p == l for p, l in zip(predictions, labels)])
        acc = correct.mean()

        # Bootstrap
        rng = np.random.default_rng(42)
        boot_accs = []
        for _ in range(n_bootstrap):
            indices = rng.choice(n, size=n, replace=True)
            boot_accs.append(correct[indices].mean())

        alpha = 1 - confidence
        lower = np.percentile(boot_accs, 100 * alpha / 2)
        upper = np.percentile(boot_accs, 100 * (1 - alpha / 2))

        return float(acc), float(lower), float(upper)

    @staticmethod
    def accuracy_by_subject(
        predictions: List[str],
        labels: List[str],
        subjects: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Per-subject accuracy breakdown.

        Args:
            predictions: List of predicted answers
            labels: List of correct answers
            subjects: List of subject names

        Returns:
            Dict mapping subject to {accuracy, n, correct}
        """
        subject_results = {}

        unique_subjects = sorted(set(subjects))
        for subj in unique_subjects:
            mask = [s == subj for s in subjects]
            subj_preds = [p for p, m in zip(predictions, mask) if m]
            subj_labels = [l for l, m in zip(labels, mask) if m]

            correct = sum(1 for p, l in zip(subj_preds, subj_labels) if p == l)
            n = len(subj_preds)

            subject_results[subj] = {
                'accuracy': correct / n if n > 0 else 0.0,
                'n': n,
                'correct': correct
            }

        return subject_results

    @staticmethod
    def flip_rate(
        original_preds: List[str],
        perturbed_preds: List[str]
    ) -> float:
        """Proportion of predictions that changed after perturbation.

        Args:
            original_preds: Predictions on original data
            perturbed_preds: Predictions on perturbed data

        Returns:
            Flip rate (0-1)
        """
        if not original_preds:
            return 0.0
        flips = sum(1 for o, p in zip(original_preds, perturbed_preds) if o != p)
        return flips / len(original_preds)

    @staticmethod
    def consistency_breakdown(
        original_preds: List[str],
        perturbed_preds: List[str],
        labels: List[str]
    ) -> Dict[str, float]:
        """Detailed consistency analysis.

        Args:
            original_preds: Predictions on original data
            perturbed_preds: Predictions on perturbed data
            labels: Correct answers

        Returns:
            Dict with consistency categories as proportions:
            - consistent_correct: both original and perturbed correct
            - consistent_wrong: both wrong with same answer
            - flip_to_correct: was wrong, now correct
            - flip_to_wrong: was correct, now wrong
            - flip_wrong_to_wrong: wrong to different wrong answer
        """
        results = {
            'consistent_correct': 0,
            'consistent_wrong': 0,
            'flip_to_correct': 0,
            'flip_to_wrong': 0,
            'flip_wrong_to_wrong': 0
        }

        for orig, pert, label in zip(original_preds, perturbed_preds, labels):
            orig_correct = orig == label
            pert_correct = pert == label

            if orig == pert:
                if orig_correct:
                    results['consistent_correct'] += 1
                else:
                    results['consistent_wrong'] += 1
            else:
                if orig_correct and not pert_correct:
                    results['flip_to_wrong'] += 1
                elif not orig_correct and pert_correct:
                    results['flip_to_correct'] += 1
                else:
                    results['flip_wrong_to_wrong'] += 1

        n = len(labels)
        if n > 0:
            return {k: v / n for k, v in results.items()}
        return results

    @staticmethod
    def position_bias(
        predictions: List[str],
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Measure bias toward specific option positions.

        Args:
            predictions: List of predicted answers
            labels: Optional list of correct answers for comparison

        Returns:
            Dict with:
            - predicted_distribution: P(predict X) for each option
            - actual_distribution: P(label = X) for each option (if labels provided)
            - bias_score: Total variation distance from uniform/actual
        """
        n = len(predictions)
        if n == 0:
            return {'predicted_distribution': {}, 'bias_score': 0.0}

        pred_counts = Counter(predictions)
        options = ['A', 'B', 'C', 'D']

        pred_dist = {k: pred_counts.get(k, 0) / n for k in options}

        result = {'predicted_distribution': pred_dist}

        if labels:
            label_counts = Counter(labels)
            label_dist = {k: label_counts.get(k, 0) / n for k in options}
            result['actual_distribution'] = label_dist

            # Total variation distance from actual distribution
            tvd = sum(abs(pred_dist.get(k, 0) - label_dist.get(k, 0)) for k in options) / 2
            result['bias_score'] = tvd
        else:
            # Bias from uniform distribution
            uniform = 0.25
            tvd = sum(abs(pred_dist.get(k, 0) - uniform) for k in options) / 2
            result['bias_score'] = tvd

        return result

    @staticmethod
    def robustness_score(
        original_preds: List[str],
        perturbed_preds: List[str],
        labels: List[str]
    ) -> Dict[str, float]:
        """Compute overall robustness metrics.

        Args:
            original_preds: Predictions on original data
            perturbed_preds: Predictions on perturbed data
            labels: Correct answers

        Returns:
            Dict with robustness metrics
        """
        orig_acc = MCQMetrics.accuracy(original_preds, labels)
        pert_acc = MCQMetrics.accuracy(perturbed_preds, labels)
        flip = MCQMetrics.flip_rate(original_preds, perturbed_preds)
        consistency = MCQMetrics.consistency_breakdown(original_preds, perturbed_preds, labels)

        return {
            'original_accuracy': orig_acc,
            'perturbed_accuracy': pert_acc,
            'accuracy_drop': orig_acc - pert_acc,
            'flip_rate': flip,
            'stability': 1 - flip,
            'robust_accuracy': consistency['consistent_correct'],
            **consistency
        }


class PubMedQAMetrics:
    """Metrics specific to PubMedQA (3-class: yes/no/maybe)."""

    CLASSES = ['yes', 'no', 'maybe']

    @staticmethod
    def accuracy(predictions: List[str], labels: List[str]) -> float:
        """3-class accuracy."""
        if not predictions:
            return 0.0
        correct = sum(1 for p, l in zip(predictions, labels) if p.lower() == l.lower())
        return correct / len(predictions)

    @staticmethod
    def accuracy_by_class(
        predictions: List[str],
        labels: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Per-class accuracy breakdown."""
        results = {}

        for cls in PubMedQAMetrics.CLASSES:
            mask = [l.lower() == cls for l in labels]
            cls_preds = [p.lower() for p, m in zip(predictions, mask) if m]
            cls_labels = [l.lower() for l, m in zip(labels, mask) if m]

            if cls_labels:
                correct = sum(1 for p, l in zip(cls_preds, cls_labels) if p == l)
                results[cls] = {
                    'accuracy': correct / len(cls_labels),
                    'n': len(cls_labels),
                    'correct': correct
                }
            else:
                results[cls] = {'accuracy': 0.0, 'n': 0, 'correct': 0}

        return results

    @staticmethod
    def context_importance(
        acc_with_context: float,
        acc_without_context: float
    ) -> Dict[str, float]:
        """Compute context importance metrics.

        Args:
            acc_with_context: Accuracy with full context
            acc_without_context: Accuracy without context

        Returns:
            Dict with context importance metrics
        """
        delta = acc_with_context - acc_without_context
        relative_gain = delta / max(acc_without_context, 0.01)

        return {
            'absolute_gain': delta,
            'relative_gain': relative_gain,
            'with_context': acc_with_context,
            'without_context': acc_without_context
        }


def parse_mcq_answer(output: str) -> str:
    """Extract answer letter from model output.

    Args:
        output: Raw model output

    Returns:
        Parsed answer ('A', 'B', 'C', 'D', or 'UNKNOWN')
    """
    output = output.strip().upper()

    # Look for explicit "Answer: X" pattern
    if "ANSWER:" in output:
        after_answer = output.split("ANSWER:")[-1].strip()
        for char in after_answer:
            if char in 'ABCD':
                return char

    # Look for first occurrence of A, B, C, or D
    for char in output:
        if char in 'ABCD':
            return char

    return 'UNKNOWN'


def parse_pubmedqa_answer(output: str) -> str:
    """Extract yes/no/maybe from model output.

    Args:
        output: Raw model output

    Returns:
        Parsed answer ('yes', 'no', 'maybe', or 'unknown')
    """
    output_lower = output.lower().strip()

    # Check for explicit answers
    if output_lower.startswith('yes') or 'answer: yes' in output_lower or 'answer is yes' in output_lower:
        return 'yes'
    if output_lower.startswith('no') or 'answer: no' in output_lower or 'answer is no' in output_lower:
        return 'no'
    if output_lower.startswith('maybe') or 'answer: maybe' in output_lower or 'answer is maybe' in output_lower:
        return 'maybe'

    # Fallback: look for keywords
    if 'yes' in output_lower and 'no' not in output_lower:
        return 'yes'
    if 'no' in output_lower and 'yes' not in output_lower:
        return 'no'
    if 'maybe' in output_lower or 'uncertain' in output_lower:
        return 'maybe'

    return 'unknown'
