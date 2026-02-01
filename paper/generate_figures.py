#!/usr/bin/env python3
"""Generate figures for the MedMCQA Robustness Study paper."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.dpi'] = 150

OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

def save_fig(fig, name):
    """Save figure in both PDF and PNG formats."""
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches='tight', dpi=300)
    fig.savefig(OUTPUT_DIR / f"{name}.png", bbox_inches='tight', dpi=150)
    print(f"Saved {name}")


def fig1_prompt_ablation():
    """Bar chart comparing prompt strategies."""
    conditions = ['Zero-shot\nDirect', 'Answer\nOnly', 'Zero-shot\nCoT', 'Few-shot\nCoT', 'Few-shot\nDirect']
    accuracies = [47.6, 43.0, 41.9, 40.8, 35.7]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#e74c3c', '#e74c3c']

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(conditions, accuracies, color=colors, edgecolor='black', linewidth=1.2)

    # Add random baseline
    ax.axhline(y=25.0, color='gray', linestyle='--', linewidth=2, label='Random baseline (25%)')

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Add delta annotations
    ax.annotate('', xy=(2, 41.9), xytext=(0, 47.6),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax.text(1, 45, '-5.7%\n(CoT hurts)', ha='center', color='red', fontsize=10, fontweight='bold')

    ax.annotate('', xy=(4, 35.7), xytext=(0, 47.6),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=2))
    ax.text(2.5, 38, '-11.9%\n(Few-shot hurts)', ha='center', color='darkred', fontsize=10, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('MedGemma-4B: Prompt Strategy Comparison on MedMCQA (n=4,183)', fontweight='bold')
    ax.set_ylim(0, 55)
    ax.legend(loc='upper right')

    plt.tight_layout()
    save_fig(fig, 'fig1_prompt_ablation')
    plt.close()


def fig2_position_bias():
    """Side-by-side bar charts showing position bias."""
    positions = ['A', 'B', 'C', 'D']
    ground_truth = [32.2, 25.1, 21.4, 21.3]
    zero_shot = [45.9, 22.1, 17.8, 14.2]
    few_shot = [76.0, 12.0, 7.0, 5.0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(positions))
    width = 0.35

    # Zero-shot direct
    ax = axes[0]
    ax.bar(x - width/2, ground_truth, width, label='Ground Truth', color='#3498db', edgecolor='black')
    ax.bar(x + width/2, zero_shot, width, label='Predicted', color='#e74c3c', edgecolor='black')
    ax.set_ylabel('Percentage (%)', fontweight='bold')
    ax.set_title('Zero-shot Direct\n(Position Bias = 0.14)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(positions, fontweight='bold', fontsize=14)
    ax.legend()
    ax.set_ylim(0, 85)

    ax.annotate('Overweights A\nby 13.7%', xy=(0, 45.9), xytext=(0.8, 55),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')

    # Few-shot direct
    ax = axes[1]
    ax.bar(x - width/2, ground_truth, width, label='Ground Truth', color='#3498db', edgecolor='black')
    ax.bar(x + width/2, few_shot, width, label='Predicted', color='#e74c3c', edgecolor='black')
    ax.set_ylabel('Percentage (%)', fontweight='bold')
    ax.set_title('Few-shot Direct\n(Position Bias = 0.47)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(positions, fontweight='bold', fontsize=14)
    ax.legend()
    ax.set_ylim(0, 85)

    ax.annotate('SEVERE: Overweights A\nby 43.8%!', xy=(0, 76), xytext=(1, 70),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                fontsize=11, color='darkred', fontweight='bold')

    plt.suptitle('Position Bias: Few-shot Examples Dramatically Increase Bias Toward Option A',
                 fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    save_fig(fig, 'fig2_position_bias')
    plt.close()


def fig3_evidence_conditioning():
    """Grouped bar chart comparing 4B and 27B on evidence conditions."""
    conditions = ['Question\nOnly', 'Full\nContext', 'Results\nOnly', 'Background\nOnly', 'Truncated\n50%', 'Truncated\n25%']
    acc_4b = [36.7, 45.0, 41.7, 26.5, 14.1, 13.1]
    acc_27b = [31.0, 38.2, 40.0, 19.8, 23.4, 18.6]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(conditions))
    width = 0.35

    bars1 = ax.bar(x - width/2, acc_4b, width, label='MedGemma-4B', color='#3498db', edgecolor='black')
    bars2 = ax.bar(x + width/2, acc_27b, width, label='MedGemma-27B', color='#9b59b6', edgecolor='black')

    ax.axhline(y=33.3, color='gray', linestyle='--', linewidth=2, label='Random baseline (33.3%)')

    for bar, acc in zip(bars1, acc_4b):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, acc in zip(bars2, acc_27b):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.annotate('27B best here!', xy=(2 + width/2, 40.0), xytext=(3.5, 48),
                arrowprops=dict(arrowstyle='->', color='purple', lw=2),
                fontsize=10, color='purple', fontweight='bold')

    ax.add_patch(plt.Rectangle((3.5, 0), 2.5, 25, fill=True, alpha=0.15, color='red'))
    ax.text(4.75, 2, 'DANGER ZONE:\nPartial context worse\nthan no context!',
            ha='center', fontsize=10, color='darkred', fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Evidence Conditioning: How Context Affects PubMedQA Performance (n=1,000)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 55)

    plt.tight_layout()
    save_fig(fig, 'fig3_evidence_conditioning')
    plt.close()


def fig4_option_order():
    """Visualize option order sensitivity with flip rates."""
    perturbations = ['Original', 'Distractor\nSwap', 'Random\nShuffle', 'Rotate-2', 'Rotate-1']
    accuracies = [47.6, 38.7, 29.2, 24.3, 20.2]
    drops = [0, -8.9, -18.4, -23.3, -27.4]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Accuracy by perturbation
    ax = axes[0]
    colors = ['#2ecc71'] + ['#e74c3c']*4
    bars = ax.bar(perturbations, accuracies, color=colors, edgecolor='black', linewidth=1.2)

    ax.axhline(y=25.0, color='gray', linestyle='--', linewidth=2, label='Random baseline (25%)')

    for bar, acc, drop in zip(bars, accuracies, drops):
        label = f'{acc:.1f}%'
        if drop != 0:
            label += f'\n({drop:+.1f}%)'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                label, ha='center', va='bottom', fontsize=10, fontweight='bold',
                color='darkred' if drop != 0 else 'darkgreen')

    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Accuracy Drops When Options Are Reordered', fontweight='bold')
    ax.set_ylim(0, 60)
    ax.legend()

    # Right: Flip rate pie chart
    ax = axes[1]
    sizes = [59.1, 40.9]
    colors_pie = ['#e74c3c', '#2ecc71']
    explode = (0.05, 0)

    wedges, texts, autotexts = ax.pie(sizes, explode=explode, colors=colors_pie,
                                       autopct='%1.1f%%', startangle=90,
                                       textprops={'fontsize': 14, 'fontweight': 'bold'})

    ax.set_title('When Options Are Shuffled,\nThe Model Changes Its Answer...', fontweight='bold', fontsize=13)

    legend_elements = [
        mpatches.Patch(facecolor='#e74c3c', label='Changes answer (59.1%)'),
        mpatches.Patch(facecolor='#2ecc71', label='Stays consistent (40.9%)')
    ]
    ax.legend(handles=legend_elements, loc='lower center', fontsize=11)

    plt.suptitle('Option Order Sensitivity: MedGemma Predictions Depend on Position, Not Content',
                 fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    save_fig(fig, 'fig4_option_order')
    plt.close()


def fig5_cot_helps_hurts():
    """Diagram showing when CoT helps vs hurts."""
    fig, ax = plt.subplots(figsize=(10, 6))

    both_correct = 1511
    both_wrong = 1510
    cot_hurts = 750
    cot_helps = 512

    # Direct correct breakdown
    ax.barh(1, both_correct, color='#2ecc71', edgecolor='black', label='Both correct')
    ax.barh(1, cot_hurts, left=both_correct, color='#e74c3c', edgecolor='black', label='CoT hurts (was correct, now wrong)')

    # Direct wrong breakdown
    ax.barh(0, cot_helps, color='#3498db', edgecolor='black', label='CoT helps (was wrong, now correct)')
    ax.barh(0, both_wrong, left=cot_helps, color='#95a5a6', edgecolor='black', label='Both wrong')

    ax.text(both_correct/2, 1, f'{both_correct}\nstay correct', ha='center', va='center',
            fontweight='bold', fontsize=11, color='white')
    ax.text(both_correct + cot_hurts/2, 1, f'{cot_hurts}\nCoT HURTS', ha='center', va='center',
            fontweight='bold', fontsize=11, color='white')

    ax.text(cot_helps/2, 0, f'{cot_helps}\nCoT helps', ha='center', va='center',
            fontweight='bold', fontsize=11, color='white')
    ax.text(cot_helps + both_wrong/2, 0, f'{both_wrong}\nstay wrong', ha='center', va='center',
            fontweight='bold', fontsize=11, color='white')

    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Direct\nWrong', 'Direct\nCorrect'], fontweight='bold', fontsize=12)
    ax.set_xlabel('Number of Questions', fontweight='bold')
    ax.set_title('Chain-of-Thought: Net Effect = -238 Questions (CoT Hurts More Than It Helps)',
                 fontweight='bold', fontsize=13)

    ax.annotate(f'Net: {cot_helps - cot_hurts} questions\n(CoT hurts overall!)',
                xy=(2000, 0.5), fontsize=14, fontweight='bold', color='darkred',
                bbox=dict(boxstyle='round', facecolor='#ffcccc', edgecolor='red'))

    ax.set_xlim(0, 2500)
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    save_fig(fig, 'fig5_cot_analysis')
    plt.close()


def fig6_key_findings_summary():
    """Visual summary of the three main findings."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Finding 1: CoT hurts
    ax = axes[0]
    ax.bar(['Zero-shot\nDirect', 'Zero-shot\nCoT'], [47.6, 41.9],
           color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=2)
    ax.axhline(y=25, color='gray', linestyle='--', alpha=0.7)
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Finding 1:\nChain-of-Thought HURTS', fontweight='bold', fontsize=14, color='#c0392b')
    ax.set_ylim(0, 55)
    ax.annotate('-5.7%', xy=(1, 41.9), xytext=(1, 50),
                arrowprops=dict(arrowstyle='->', color='red', lw=3),
                fontsize=16, fontweight='bold', color='red', ha='center')

    # Finding 2: 59% flip rate
    ax = axes[1]
    sizes = [59.1, 40.9]
    colors = ['#e74c3c', '#2ecc71']
    wedges, texts, autotexts = ax.pie(sizes, colors=colors, autopct='%1.1f%%',
                                       startangle=90, explode=(0.1, 0),
                                       textprops={'fontsize': 14, 'fontweight': 'bold'})
    ax.set_title('Finding 2:\n59% Flip Rate on Shuffle', fontweight='bold', fontsize=14, color='#c0392b')
    ax.text(0, -1.4, 'Model changes answer\nwhen options reordered', ha='center', fontsize=11)

    # Finding 3: Partial context misleads
    ax = axes[2]
    conditions = ['No\nContext', 'Full\nContext', '50%\nTruncated']
    values = [36.7, 45.0, 14.1]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    ax.bar(conditions, values, color=colors, edgecolor='black', linewidth=2)
    ax.axhline(y=33.3, color='gray', linestyle='--', alpha=0.7)
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Finding 3:\nPartial Context MISLEADS', fontweight='bold', fontsize=14, color='#c0392b')
    ax.set_ylim(0, 55)

    ax.annotate('Worse than\nno context!', xy=(2, 14.1), xytext=(2, 30),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                fontsize=12, fontweight='bold', color='darkred', ha='center')

    plt.suptitle('Three Key Findings: Standard Prompt Engineering Fails for Medical LLMs',
                 fontweight='bold', fontsize=15, y=1.05)
    plt.tight_layout()
    save_fig(fig, 'fig6_key_findings')
    plt.close()


if __name__ == '__main__':
    print("Generating figures...")
    fig1_prompt_ablation()
    fig2_position_bias()
    fig3_evidence_conditioning()
    fig4_option_order()
    fig5_cot_helps_hurts()
    fig6_key_findings_summary()
    print("\nAll figures generated successfully!")
