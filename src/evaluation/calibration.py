"""Confidence calibration metrics and visualization."""

from typing import List, Dict, Tuple, Any
import numpy as np


class CalibrationMetrics:
    """Expected Calibration Error and related metrics."""

    @staticmethod
    def compute_ece(
        confidences: List[float],
        predictions: List[str],
        labels: List[str],
        n_bins: int = 10
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute Expected Calibration Error.

        ECE = sum(|accuracy(bin) - confidence(bin)| * |bin| / N)

        Args:
            confidences: Model confidence scores
            predictions: Model predictions
            labels: True labels
            n_bins: Number of calibration bins

        Returns:
            Tuple of (ECE value, bin details dict)
        """
        confidences = np.array(confidences)
        correct = np.array([p == l for p, l in zip(predictions, labels)])
        n = len(confidences)

        if n == 0:
            return 0.0, {'bins': [], 'n_samples': 0}

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_data = []

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            bin_size = in_bin.sum()

            if bin_size > 0:
                bin_accuracy = correct[in_bin].mean()
                bin_confidence = confidences[in_bin].mean()
            else:
                bin_accuracy = 0.0
                bin_confidence = (bin_boundaries[i] + bin_boundaries[i + 1]) / 2

            bin_data.append({
                'bin_start': float(bin_boundaries[i]),
                'bin_end': float(bin_boundaries[i + 1]),
                'accuracy': float(bin_accuracy),
                'confidence': float(bin_confidence),
                'size': int(bin_size),
                'gap': float(abs(bin_accuracy - bin_confidence))
            })

        # Compute ECE
        ece = sum(b['gap'] * b['size'] / n for b in bin_data)

        return float(ece), {'bins': bin_data, 'n_samples': n}

    @staticmethod
    def compute_mce(
        confidences: List[float],
        predictions: List[str],
        labels: List[str],
        n_bins: int = 10
    ) -> float:
        """Compute Maximum Calibration Error.

        MCE = max(|accuracy(bin) - confidence(bin)|)

        Args:
            confidences: Model confidence scores
            predictions: Model predictions
            labels: True labels
            n_bins: Number of calibration bins

        Returns:
            MCE value
        """
        _, bin_data = CalibrationMetrics.compute_ece(
            confidences, predictions, labels, n_bins
        )

        if not bin_data['bins']:
            return 0.0

        # Only consider bins with samples
        gaps = [b['gap'] for b in bin_data['bins'] if b['size'] > 0]
        return max(gaps) if gaps else 0.0

    @staticmethod
    def compute_brier_score(
        confidences: List[float],
        predictions: List[str],
        labels: List[str]
    ) -> float:
        """Compute Brier score for binary correctness.

        Brier = mean((confidence - correct)^2)

        Args:
            confidences: Model confidence scores
            predictions: Model predictions
            labels: True labels

        Returns:
            Brier score (lower is better)
        """
        correct = np.array([p == l for p, l in zip(predictions, labels)], dtype=float)
        confidences = np.array(confidences)
        return float(np.mean((confidences - correct) ** 2))

    @staticmethod
    def compute_overconfidence(
        confidences: List[float],
        predictions: List[str],
        labels: List[str]
    ) -> Dict[str, float]:
        """Compute overconfidence metrics.

        Args:
            confidences: Model confidence scores
            predictions: Model predictions
            labels: True labels

        Returns:
            Dict with overconfidence metrics
        """
        correct = np.array([p == l for p, l in zip(predictions, labels)])
        confidences = np.array(confidences)

        accuracy = correct.mean()
        mean_confidence = confidences.mean()

        # Average confidence when wrong
        wrong_mask = ~correct
        avg_conf_when_wrong = confidences[wrong_mask].mean() if wrong_mask.any() else 0.0

        # Average confidence when correct
        avg_conf_when_correct = confidences[correct].mean() if correct.any() else 0.0

        return {
            'mean_confidence': float(mean_confidence),
            'accuracy': float(accuracy),
            'overconfidence': float(mean_confidence - accuracy),
            'avg_confidence_when_correct': float(avg_conf_when_correct),
            'avg_confidence_when_wrong': float(avg_conf_when_wrong),
            'confidence_gap': float(avg_conf_when_correct - avg_conf_when_wrong)
        }


class SelfConsistencyCalibration:
    """Calibration analysis for self-consistency voting."""

    @staticmethod
    def compute_vote_entropy(vote_counts: Dict[str, int]) -> float:
        """Compute entropy of vote distribution.

        Higher entropy = more disagreement among samples.

        Args:
            vote_counts: Dict mapping answers to counts

        Returns:
            Entropy value
        """
        total = sum(vote_counts.values())
        if total == 0:
            return 0.0

        probs = np.array(list(vote_counts.values())) / total
        probs = probs[probs > 0]  # Remove zeros for log
        return float(-np.sum(probs * np.log(probs)))

    @staticmethod
    def aggregate_votes(
        samples: List[str],
        parse_fn=None
    ) -> Tuple[str, float, Dict[str, int]]:
        """Aggregate samples using majority vote.

        Args:
            samples: List of model outputs
            parse_fn: Function to parse answers from outputs

        Returns:
            Tuple of (majority_answer, confidence, vote_counts)
        """
        if parse_fn is None:
            parsed = samples
        else:
            parsed = [parse_fn(s) for s in samples]

        # Count votes
        vote_counts: Dict[str, int] = {}
        for p in parsed:
            vote_counts[p] = vote_counts.get(p, 0) + 1

        # Find majority (break ties alphabetically for determinism)
        max_count = max(vote_counts.values())
        winners = sorted([k for k, v in vote_counts.items() if v == max_count])
        majority_answer = winners[0]

        # Confidence = proportion of votes for majority
        confidence = max_count / len(samples)

        return majority_answer, float(confidence), vote_counts

    @staticmethod
    def analyze_self_consistency(
        all_samples: List[List[str]],
        labels: List[str],
        parse_fn=None
    ) -> Dict[str, Any]:
        """Analyze self-consistency results across a dataset.

        Args:
            all_samples: List of sample lists (one per item)
            labels: Correct labels
            parse_fn: Function to parse answers

        Returns:
            Dict with aggregated metrics
        """
        predictions = []
        confidences = []
        entropies = []
        vote_distributions = []

        for samples in all_samples:
            answer, conf, votes = SelfConsistencyCalibration.aggregate_votes(
                samples, parse_fn
            )
            predictions.append(answer)
            confidences.append(conf)
            entropies.append(SelfConsistencyCalibration.compute_vote_entropy(votes))
            vote_distributions.append(votes)

        # Compute metrics
        accuracy = sum(p == l for p, l in zip(predictions, labels)) / len(labels)

        ece, cal_data = CalibrationMetrics.compute_ece(
            confidences, predictions, labels
        )

        brier = CalibrationMetrics.compute_brier_score(
            confidences, predictions, labels
        )

        overconf = CalibrationMetrics.compute_overconfidence(
            confidences, predictions, labels
        )

        return {
            'accuracy': accuracy,
            'ece': ece,
            'brier_score': brier,
            'mean_confidence': np.mean(confidences),
            'mean_entropy': np.mean(entropies),
            'calibration_data': cal_data,
            'overconfidence_metrics': overconf,
            'predictions': predictions,
            'confidences': confidences
        }


def reliability_diagram_data(
    calibration_data: Dict[str, Any]
) -> Dict[str, List]:
    """Prepare data for reliability diagram plotting.

    Args:
        calibration_data: Output from compute_ece

    Returns:
        Dict with plot data
    """
    bins = calibration_data['bins']

    return {
        'bin_centers': [(b['bin_start'] + b['bin_end']) / 2 for b in bins],
        'accuracies': [b['accuracy'] for b in bins],
        'confidences': [b['confidence'] for b in bins],
        'sizes': [b['size'] for b in bins],
        'gaps': [b['gap'] for b in bins]
    }
