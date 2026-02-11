"""Statistical significance tests for comparing model predictions."""

from typing import List, Tuple, Dict, Callable

import numpy as np


def mcnemar_test(
    preds1: List[str],
    preds2: List[str],
    labels: List[str],
    alpha: float = 0.05
) -> Dict[str, float]:
    """McNemar's test for comparing two classifiers on the same data.

    Constructs a 2x2 contingency table based on whether each classifier
    got each item correct, then applies McNemar's chi-squared test on the
    off-diagonal entries.

    Args:
        preds1: Predictions from model 1
        preds2: Predictions from model 2
        labels: Correct answers
        alpha: Significance level (default 0.05)

    Returns:
        Dict with:
        - chi2: Chi-squared statistic
        - p_value: Two-sided p-value
        - significant: Whether the test is significant at the given alpha
        - b: Count where model 1 correct and model 2 wrong
        - c: Count where model 1 wrong and model 2 correct
    """
    b = 0  # model 1 correct, model 2 wrong
    c = 0  # model 1 wrong, model 2 correct

    for p1, p2, label in zip(preds1, preds2, labels):
        c1 = p1 == label
        c2 = p2 == label
        if c1 and not c2:
            b += 1
        elif not c1 and c2:
            c += 1

    # McNemar's test with continuity correction
    if b + c == 0:
        return {
            'chi2': 0.0,
            'p_value': 1.0,
            'significant': False,
            'b': b,
            'c': c
        }

    chi2 = float((abs(b - c) - 1) ** 2 / (b + c))

    # Compute p-value from chi-squared distribution with 1 df
    # Using survival function approximation via scipy-free method
    p_value = _chi2_sf(chi2, df=1)

    return {
        'chi2': chi2,
        'p_value': p_value,
        'significant': p_value < alpha,
        'b': b,
        'c': c
    }


def bootstrap_ci(
    predictions: List[str],
    labels: List[str],
    metric_fn: Callable[[List[str], List[str]], float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42
) -> Dict[str, float]:
    """Generic bootstrap confidence interval computation for any metric.

    Args:
        predictions: List of predicted answers
        labels: List of correct answers
        metric_fn: Function(predictions, labels) -> float that computes the metric
        n_bootstrap: Number of bootstrap resamples (default 10000)
        confidence: Confidence level (default 0.95)
        seed: Random seed for reproducibility

    Returns:
        Dict with:
        - estimate: Point estimate of the metric
        - lower: Lower bound of confidence interval
        - upper: Upper bound of confidence interval
        - std_error: Bootstrap standard error
    """
    n = len(predictions)
    if n == 0:
        return {'estimate': 0.0, 'lower': 0.0, 'upper': 0.0, 'std_error': 0.0}

    preds_arr = np.array(predictions)
    labels_arr = np.array(labels)

    estimate = metric_fn(predictions, labels)

    rng = np.random.default_rng(seed)
    boot_values = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        boot_preds = preds_arr[indices].tolist()
        boot_labels = labels_arr[indices].tolist()
        boot_values[i] = metric_fn(boot_preds, boot_labels)

    alpha = 1 - confidence
    lower = float(np.percentile(boot_values, 100 * alpha / 2))
    upper = float(np.percentile(boot_values, 100 * (1 - alpha / 2)))
    std_error = float(np.std(boot_values))

    return {
        'estimate': float(estimate),
        'lower': lower,
        'upper': upper,
        'std_error': std_error
    }


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> Dict[str, object]:
    """Bonferroni correction for multiple comparisons.

    Adjusts p-values by multiplying each by the number of tests and caps
    at 1.0. A comparison is significant if its adjusted p-value < alpha.

    Args:
        p_values: List of raw p-values from individual tests
        alpha: Family-wise significance level (default 0.05)

    Returns:
        Dict with:
        - adjusted_p_values: List of Bonferroni-adjusted p-values
        - significant: List of booleans indicating significance after correction
        - n_significant: Count of significant tests
        - corrected_alpha: The per-test alpha threshold (alpha / n_tests)
    """
    m = len(p_values)
    if m == 0:
        return {
            'adjusted_p_values': [],
            'significant': [],
            'n_significant': 0,
            'corrected_alpha': alpha
        }

    adjusted = [min(p * m, 1.0) for p in p_values]
    significant = [p < alpha for p in adjusted]

    return {
        'adjusted_p_values': adjusted,
        'significant': significant,
        'n_significant': sum(significant),
        'corrected_alpha': alpha / m
    }


def paired_permutation_test(
    preds1: List[str],
    preds2: List[str],
    labels: List[str],
    n_permutations: int = 10000,
    seed: int = 42
) -> Dict[str, float]:
    """Permutation test for paired comparison of two classifiers.

    Tests whether the difference in accuracy between two classifiers is
    statistically significant by randomly swapping predictions between them.

    Args:
        preds1: Predictions from model 1
        preds2: Predictions from model 2
        labels: Correct answers
        n_permutations: Number of random permutations (default 10000)
        seed: Random seed for reproducibility

    Returns:
        Dict with:
        - observed_diff: Observed accuracy difference (acc1 - acc2)
        - p_value: Two-sided p-value
        - significant: Whether the test is significant at alpha=0.05
    """
    n = len(labels)
    if n == 0:
        return {'observed_diff': 0.0, 'p_value': 1.0, 'significant': False}

    correct1 = np.array([p == l for p, l in zip(preds1, labels)], dtype=np.float64)
    correct2 = np.array([p == l for p, l in zip(preds2, labels)], dtype=np.float64)

    observed_diff = float(correct1.mean() - correct2.mean())

    rng = np.random.default_rng(seed)
    count_extreme = 0

    for _ in range(n_permutations):
        # For each sample, randomly decide whether to swap
        swap = rng.random(n) < 0.5
        perm1 = np.where(swap, correct2, correct1)
        perm2 = np.where(swap, correct1, correct2)
        perm_diff = perm1.mean() - perm2.mean()

        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1

    p_value = float(count_extreme / n_permutations)

    return {
        'observed_diff': observed_diff,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def _chi2_sf(x: float, df: int = 1) -> float:
    """Survival function (1 - CDF) for chi-squared distribution.

    Uses the regularized incomplete gamma function approximation for df=1.
    For df=1, chi2 SF = 2 * (1 - Phi(sqrt(x))) where Phi is the standard
    normal CDF.

    Args:
        x: Chi-squared value
        df: Degrees of freedom (only df=1 supported)

    Returns:
        p-value (survival function value)
    """
    if x <= 0:
        return 1.0
    if df != 1:
        raise ValueError("Only df=1 is supported by this approximation")

    # For df=1: P(X > x) = 2 * (1 - Phi(sqrt(x)))
    # Using the complementary error function: erfc(z/sqrt(2)) = 2*(1-Phi(z))
    z = np.sqrt(x)
    from math import erfc, sqrt
    p = erfc(z / sqrt(2))
    return float(p)
