"""Visualization utilities for experiment results."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any, Optional
import os


def set_style():
    """Set consistent plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16


def plot_reliability_diagram(
    calibration_data: Dict[str, Any],
    title: str = "Reliability Diagram",
    output_path: Optional[str] = None
) -> plt.Figure:
    """Generate reliability diagram.

    Args:
        calibration_data: Output from CalibrationMetrics.compute_ece
        title: Plot title
        output_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    set_style()

    bins = calibration_data['bins']
    n_samples = calibration_data['n_samples']

    fig, ax = plt.subplots(figsize=(8, 8))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')

    # Bar chart of accuracy per bin
    bin_centers = [(b['bin_start'] + b['bin_end']) / 2 for b in bins]
    accuracies = [b['accuracy'] for b in bins]
    sizes = [b['size'] for b in bins]

    # Width based on bin boundaries
    width = 1.0 / len(bins) * 0.8

    # Color by gap size
    gaps = [b['gap'] for b in bins]
    colors = plt.cm.RdYlGn_r(np.array(gaps) / max(max(gaps), 0.01))

    bars = ax.bar(bin_centers, accuracies, width=width, alpha=0.7,
                  color=colors, edgecolor='black', linewidth=1)

    # Add sample counts on top of bars
    for bar, size in zip(bars, sizes):
        if size > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'n={size}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{title}\n(n={n_samples})')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_accuracy_comparison(
    results: Dict[str, float],
    title: str = "Accuracy Comparison",
    ylabel: str = "Accuracy",
    output_path: Optional[str] = None
) -> plt.Figure:
    """Bar chart comparing accuracy across conditions.

    Args:
        results: Dict mapping condition names to accuracy values
        title: Plot title
        ylabel: Y-axis label
        output_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    set_style()

    fig, ax = plt.subplots(figsize=(12, 6))

    names = list(results.keys())
    values = list(results.values())

    # Sort by accuracy
    sorted_pairs = sorted(zip(values, names), reverse=True)
    values, names = zip(*sorted_pairs)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))

    bars = ax.bar(names, values, color=colors, edgecolor='black', linewidth=1)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.1%}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.15)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_subject_heatmap(
    subject_results: Dict[str, Dict[str, Any]],
    models: List[str] = None,
    title: str = "Accuracy by Subject",
    output_path: Optional[str] = None
) -> plt.Figure:
    """Heatmap of accuracy by subject and model/condition.

    Args:
        subject_results: Nested dict of model -> subject -> {accuracy, n}
        models: List of model names (keys in subject_results)
        title: Plot title
        output_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    set_style()

    if models is None:
        models = list(subject_results.keys())

    # Get all subjects
    all_subjects = set()
    for model_data in subject_results.values():
        all_subjects.update(model_data.keys())
    subjects = sorted(all_subjects)

    # Build matrix
    matrix = np.zeros((len(subjects), len(models)))
    for j, model in enumerate(models):
        for i, subject in enumerate(subjects):
            if subject in subject_results.get(model, {}):
                matrix[i, j] = subject_results[model][subject].get('accuracy', 0)

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 2), max(8, len(subjects) * 0.4)))

    sns.heatmap(matrix, annot=True, fmt='.1%', cmap='RdYlGn',
                xticklabels=models, yticklabels=subjects,
                ax=ax, vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'})

    ax.set_title(title)
    ax.set_xlabel('Model / Condition')
    ax.set_ylabel('Subject')

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_robustness_comparison(
    robustness_metrics: Dict[str, Dict[str, float]],
    title: str = "Robustness Analysis",
    output_path: Optional[str] = None
) -> plt.Figure:
    """Grouped bar chart for robustness metrics.

    Args:
        robustness_metrics: Dict of perturbation -> metrics
        title: Plot title
        output_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    set_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    perturbations = list(robustness_metrics.keys())

    # Accuracy comparison
    ax1 = axes[0]
    orig_acc = [robustness_metrics[p].get('original_accuracy', 0) for p in perturbations]
    pert_acc = [robustness_metrics[p].get('perturbed_accuracy', 0) for p in perturbations]

    x = np.arange(len(perturbations))
    width = 0.35

    ax1.bar(x - width / 2, orig_acc, width, label='Original', color='steelblue')
    ax1.bar(x + width / 2, pert_acc, width, label='Perturbed', color='coral')

    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy: Original vs Perturbed')
    ax1.set_xticks(x)
    ax1.set_xticklabels(perturbations, rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 1)

    # Flip rate and stability
    ax2 = axes[1]
    flip_rates = [robustness_metrics[p].get('flip_rate', 0) for p in perturbations]

    colors = plt.cm.Reds(np.array(flip_rates) / max(max(flip_rates), 0.01))
    ax2.bar(perturbations, flip_rates, color=colors, edgecolor='black')

    ax2.set_ylabel('Flip Rate')
    ax2.set_title('Answer Flip Rate by Perturbation')
    ax2.set_xticklabels(perturbations, rotation=45, ha='right')
    ax2.set_ylim(0, max(flip_rates) * 1.2 if flip_rates else 1)

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_self_consistency_curve(
    results_by_n: Dict[int, Dict[str, float]],
    title: str = "Self-Consistency: Accuracy vs Sample Count",
    output_path: Optional[str] = None
) -> plt.Figure:
    """Line plot of accuracy vs number of samples.

    Args:
        results_by_n: Dict of N -> {accuracy, ece, ...}
        title: Plot title
        output_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    set_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    n_values = sorted(results_by_n.keys())
    accuracies = [results_by_n[n].get('accuracy', 0) for n in n_values]
    eces = [results_by_n[n].get('ece', 0) for n in n_values]

    # Accuracy curve
    ax1 = axes[0]
    ax1.plot(n_values, accuracies, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax1.set_xlabel('Number of Samples (N)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs Sample Count')
    ax1.set_xticks(n_values)
    ax1.grid(True, alpha=0.3)

    # ECE curve
    ax2 = axes[1]
    ax2.plot(n_values, eces, 's-', linewidth=2, markersize=8, color='coral')
    ax2.set_xlabel('Number of Samples (N)')
    ax2.set_ylabel('ECE (lower is better)')
    ax2.set_title('Calibration Error vs Sample Count')
    ax2.set_xticks(n_values)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig


def plot_context_comparison(
    results: Dict[str, float],
    title: str = "Context Conditioning Effect",
    output_path: Optional[str] = None
) -> plt.Figure:
    """Bar chart comparing different context conditions.

    Args:
        results: Dict mapping context condition to accuracy
        title: Plot title
        output_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    set_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    conditions = list(results.keys())
    values = list(results.values())

    # Color coding: no context = gray, partial = yellow, full = green
    colors = []
    for cond in conditions:
        if 'none' in cond.lower() or 'question' in cond.lower():
            colors.append('gray')
        elif 'full' in cond.lower():
            colors.append('forestgreen')
        else:
            colors.append('goldenrod')

    bars = ax.bar(conditions, values, color=colors, edgecolor='black', linewidth=1)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.1%}', ha='center', va='bottom', fontsize=11)

    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.15)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')

    return fig
