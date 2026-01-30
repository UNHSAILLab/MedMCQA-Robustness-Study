#!/usr/bin/env python3
"""Generate figures for the paper from experimental results."""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

RESULTS_DIR = "../outputs/results"
FIGURES_DIR = "figures"

os.makedirs(FIGURES_DIR, exist_ok=True)


def load_latest_results(experiment_prefix, model_suffix):
    """Load the most recent results file for an experiment."""
    files = [f for f in os.listdir(RESULTS_DIR)
             if f.startswith(experiment_prefix) and model_suffix in f]
    if not files:
        return None
    latest = sorted(files)[-1]
    with open(os.path.join(RESULTS_DIR, latest)) as f:
        return json.load(f)


def figure1_prompt_ablation():
    """Bar chart comparing accuracy across prompt conditions."""
    data = load_latest_results("exp1_prompt_ablation", "4b")
    if not data:
        print("No prompt ablation data found")
        return

    metrics = data.get('metrics', {})

    conditions = ['zero_shot_direct', 'zero_shot_cot', 'few_shot_3_direct',
                  'few_shot_3_cot', 'answer_only']
    labels = ['Zero-shot\nDirect', 'Zero-shot\nCoT', 'Few-shot\nDirect',
              'Few-shot\nCoT', 'Answer\nOnly']

    accuracies = []
    errors = []

    for cond in conditions:
        if cond in metrics:
            acc = metrics[cond].get('accuracy', 0)
            ci = metrics[cond].get('accuracy_ci', [acc, acc])
            accuracies.append(acc * 100)
            errors.append([(acc - ci[0]) * 100, (ci[1] - acc) * 100])
        else:
            accuracies.append(0)
            errors.append([0, 0])

    errors = np.array(errors).T

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, accuracies, yerr=errors, capsize=5,
                  color=['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12'],
                  edgecolor='black', linewidth=1.2)

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Prompt Condition', fontsize=12)
    ax.set_title('MedGemma-4B Performance Across Prompt Strategies\n(MedMCQA)', fontsize=14)
    ax.set_ylim(0, 70)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.annotate(f'{acc:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add horizontal line for random baseline
    ax.axhline(y=25, color='gray', linestyle='--', linewidth=1.5, label='Random (25%)')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_prompt_ablation.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_prompt_ablation.png'), dpi=300, bbox_inches='tight')
    print("Saved figure 1: prompt ablation")


def figure2_position_bias():
    """Heatmap showing predicted vs actual answer distributions."""
    data = load_latest_results("exp1_prompt_ablation", "4b")
    if not data:
        print("No prompt ablation data found")
        return

    metrics = data.get('metrics', {})

    conditions = ['zero_shot_direct', 'few_shot_3_direct']
    condition_labels = ['Zero-shot Direct', 'Few-shot Direct']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, cond, label in zip(axes, conditions, condition_labels):
        if cond not in metrics:
            continue

        pb = metrics[cond].get('position_bias', {})
        pred_dist = pb.get('predicted_distribution', {})
        actual_dist = pb.get('actual_distribution', {})

        options = ['A', 'B', 'C', 'D']
        pred_vals = [pred_dist.get(o, 0) * 100 for o in options]
        actual_vals = [actual_dist.get(o, 0) * 100 for o in options]

        x = np.arange(len(options))
        width = 0.35

        bars1 = ax.bar(x - width/2, pred_vals, width, label='Predicted', color='#3498db')
        bars2 = ax.bar(x + width/2, actual_vals, width, label='Actual', color='#2ecc71')

        ax.set_ylabel('Frequency (%)', fontsize=11)
        ax.set_xlabel('Answer Option', fontsize=11)
        ax.set_title(f'{label}', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(options)
        ax.legend()
        ax.set_ylim(0, 80)

        bias = pb.get('bias_score', 0)
        ax.annotate(f'Bias Score: {bias:.2f}', xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Position Bias in MedGemma-4B Predictions', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_position_bias.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_position_bias.png'), dpi=300, bbox_inches='tight')
    print("Saved figure 2: position bias")


def figure3_evidence_conditioning():
    """Bar chart showing accuracy across evidence conditions."""
    data = load_latest_results("exp3_evidence_conditioning", "4b")
    if not data:
        print("No evidence conditioning data found")
        return

    metrics = data.get('metrics', {})

    conditions = ['question_only', 'full_context', 'truncated_50',
                  'truncated_25', 'background_only', 'results_only']
    labels = ['Question\nOnly', 'Full\nContext', 'Truncated\n50%',
              'Truncated\n25%', 'Background\nOnly', 'Results\nOnly']

    accuracies = []
    for cond in conditions:
        if cond in metrics:
            acc = metrics[cond].get('accuracy', 0)
            accuracies.append(acc * 100)
        else:
            accuracies.append(0)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#95a5a6', '#2ecc71', '#f39c12', '#e74c3c', '#3498db', '#9b59b6']
    bars = ax.bar(labels, accuracies, color=colors, edgecolor='black', linewidth=1.2)

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Context Condition', fontsize=12)
    ax.set_title('MedGemma-4B Performance with Varying Context\n(PubMedQA)', fontsize=14)
    ax.set_ylim(0, 50)

    for bar, acc in zip(bars, accuracies):
        ax.annotate(f'{acc:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.axhline(y=33.3, color='gray', linestyle='--', linewidth=1.5, label='Random (33%)')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_evidence_conditioning.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(FIGURES_DIR, 'fig3_evidence_conditioning.png'), dpi=300, bbox_inches='tight')
    print("Saved figure 3: evidence conditioning")


def figure4_model_comparison():
    """Compare 4B and 27B model performance (if 27B data is valid)."""
    data_4b = load_latest_results("exp1_prompt_ablation", "4b")
    data_27b = load_latest_results("exp1_prompt_ablation", "27b")

    if not data_4b:
        print("No 4B data found")
        return

    # Check if 27B has valid results
    if data_27b:
        metrics_27b = data_27b.get('metrics', {})
        acc_27b = metrics_27b.get('zero_shot_direct', {}).get('accuracy', 0)
        if acc_27b == 0:
            print("27B results appear invalid (0% accuracy), skipping comparison")
            return

    print("Figure 4: Model comparison requires valid 27B results")


if __name__ == "__main__":
    print("Generating paper figures...")
    print(f"Reading results from: {RESULTS_DIR}")
    print(f"Saving figures to: {FIGURES_DIR}")
    print()

    figure1_prompt_ablation()
    figure2_position_bias()
    figure3_evidence_conditioning()
    figure4_model_comparison()

    print("\nDone!")
