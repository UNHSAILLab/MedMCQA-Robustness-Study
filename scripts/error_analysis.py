#!/usr/bin/env python3
"""
Error Analysis: Why does Chain-of-Thought hurt performance?

This script analyzes specific examples where CoT prompting leads to incorrect
answers while direct prompting succeeds, and vice versa.
"""

import json
import os
import random
from collections import defaultdict
from typing import List, Dict, Tuple

RESULTS_DIR = "outputs/results"


def load_prompt_ablation_results():
    """Load the most recent prompt ablation results."""
    files = [f for f in os.listdir(RESULTS_DIR)
             if f.startswith('exp1_prompt_ablation') and '4b' in f and f.endswith('.json')]
    if not files:
        raise FileNotFoundError("No prompt ablation results found")

    latest = sorted(files)[-1]
    with open(os.path.join(RESULTS_DIR, latest)) as f:
        return json.load(f)


def find_cot_failures(data: Dict) -> Tuple[List, List]:
    """
    Find cases where:
    1. Direct correct, CoT wrong (CoT hurt)
    2. Direct wrong, CoT correct (CoT helped)
    """
    results = data.get('results', {})

    # Get responses and items from nested structure
    direct_data = results.get('zero_shot_direct', {})
    cot_data = results.get('zero_shot_cot', {})

    direct_responses = {r['item_id']: r for r in direct_data.get('responses', [])}
    cot_responses = {r['item_id']: r for r in cot_data.get('responses', [])}

    # Get correct answers from items
    direct_items = {item['id']: item for item in direct_data.get('items', [])}

    # Merge responses with correct answers
    direct = {}
    for item_id, resp in direct_responses.items():
        item = direct_items.get(item_id, {})
        direct[item_id] = {
            **resp,
            'correct_answer': item.get('correct_answer'),
            'subject': item.get('subject'),
            'question': resp.get('prompt', '').split('\n')[0].replace('Question: ', '')
        }

    cot = {}
    for item_id, resp in cot_responses.items():
        item = direct_items.get(item_id, {})
        cot[item_id] = {
            **resp,
            'correct_answer': item.get('correct_answer'),
            'subject': item.get('subject'),
            'question': resp.get('prompt', '').split('\n')[0].replace('Question: ', '')
        }

    cot_hurt = []  # Direct correct, CoT wrong
    cot_helped = []  # Direct wrong, CoT correct

    for item_id in direct:
        if item_id not in cot:
            continue

        d = direct[item_id]
        c = cot[item_id]

        d_correct = d.get('parsed_answer') == d.get('correct_answer')
        c_correct = c.get('parsed_answer') == c.get('correct_answer')

        if d_correct and not c_correct:
            cot_hurt.append({
                'item_id': item_id,
                'question': d.get('question', ''),
                'correct_answer': d.get('correct_answer'),
                'direct_answer': d.get('parsed_answer'),
                'direct_output': d.get('raw_output', ''),
                'cot_answer': c.get('parsed_answer'),
                'cot_output': c.get('raw_output', ''),
            })
        elif not d_correct and c_correct:
            cot_helped.append({
                'item_id': item_id,
                'question': d.get('question', ''),
                'correct_answer': d.get('correct_answer'),
                'direct_answer': d.get('parsed_answer'),
                'direct_output': d.get('raw_output', ''),
                'cot_answer': c.get('parsed_answer'),
                'cot_output': c.get('raw_output', ''),
            })

    return cot_hurt, cot_helped


def find_few_shot_failures(data: Dict) -> Tuple[List, List]:
    """
    Find cases where:
    1. Zero-shot correct, few-shot wrong
    2. Zero-shot wrong, few-shot correct
    """
    results = data.get('results', {})

    # Get responses and items from nested structure
    zero_shot_data = results.get('zero_shot_direct', {})
    few_shot_data = results.get('few_shot_3_direct', {})

    zero_shot_responses = {r['item_id']: r for r in zero_shot_data.get('responses', [])}
    few_shot_responses = {r['item_id']: r for r in few_shot_data.get('responses', [])}

    # Get correct answers from items
    zero_shot_items = {item['id']: item for item in zero_shot_data.get('items', [])}

    # Merge responses with correct answers
    zero_shot = {}
    for item_id, resp in zero_shot_responses.items():
        item = zero_shot_items.get(item_id, {})
        zero_shot[item_id] = {
            **resp,
            'correct_answer': item.get('correct_answer'),
            'subject': item.get('subject'),
            'question': resp.get('prompt', '').split('\n')[0].replace('Question: ', '')
        }

    few_shot = {}
    for item_id, resp in few_shot_responses.items():
        item = zero_shot_items.get(item_id, {})
        few_shot[item_id] = {
            **resp,
            'correct_answer': item.get('correct_answer'),
            'subject': item.get('subject'),
            'question': resp.get('prompt', '').split('\n')[0].replace('Question: ', '')
        }

    fs_hurt = []
    fs_helped = []

    for item_id in zero_shot:
        if item_id not in few_shot:
            continue

        z = zero_shot[item_id]
        f = few_shot[item_id]

        z_correct = z.get('parsed_answer') == z.get('correct_answer')
        f_correct = f.get('parsed_answer') == f.get('correct_answer')

        if z_correct and not f_correct:
            fs_hurt.append({
                'item_id': item_id,
                'question': z.get('question', ''),
                'correct_answer': z.get('correct_answer'),
                'zero_shot_answer': z.get('parsed_answer'),
                'few_shot_answer': f.get('parsed_answer'),
            })
        elif not z_correct and f_correct:
            fs_helped.append({
                'item_id': item_id,
                'question': z.get('question', ''),
                'correct_answer': z.get('correct_answer'),
                'zero_shot_answer': z.get('parsed_answer'),
                'few_shot_answer': f.get('parsed_answer'),
            })

    return fs_hurt, fs_helped


def analyze_cot_reasoning(cases: List[Dict]) -> Dict:
    """Analyze patterns in CoT failures."""
    patterns = defaultdict(int)

    for case in cases:
        cot_output = case.get('cot_output', '').lower()

        # Check for common failure patterns
        if 'however' in cot_output or 'but' in cot_output:
            patterns['self_contradiction'] += 1
        if 'therefore' in cot_output and case['cot_answer'] != case['correct_answer']:
            patterns['wrong_conclusion'] += 1
        if len(cot_output) > 500:
            patterns['verbose_reasoning'] += 1
        if 'not sure' in cot_output or 'unclear' in cot_output:
            patterns['expressed_uncertainty'] += 1

    return dict(patterns)


def print_examples(cases: List[Dict], title: str, n: int = 5):
    """Print example cases."""
    print(f"\n{'='*70}")
    print(f"{title} (showing {min(n, len(cases))} of {len(cases)} cases)")
    print('='*70)

    for i, case in enumerate(random.sample(cases, min(n, len(cases)))):
        print(f"\n--- Example {i+1} ---")
        print(f"Question: {case.get('question', 'N/A')[:200]}...")
        print(f"Correct Answer: {case.get('correct_answer')}")

        if 'direct_answer' in case:
            print(f"Direct Answer: {case.get('direct_answer')}")
            print(f"CoT Answer: {case.get('cot_answer')}")
            if case.get('cot_output'):
                print(f"CoT Reasoning (first 300 chars): {case['cot_output'][:300]}...")
        elif 'zero_shot_answer' in case:
            print(f"Zero-shot Answer: {case.get('zero_shot_answer')}")
            print(f"Few-shot Answer: {case.get('few_shot_answer')}")


def main():
    print("Loading results...")
    data = load_prompt_ablation_results()

    print("\n" + "="*70)
    print("CHAIN-OF-THOUGHT ERROR ANALYSIS")
    print("="*70)

    # CoT analysis
    cot_hurt, cot_helped = find_cot_failures(data)
    print(f"\nCoT hurt (direct correct, CoT wrong): {len(cot_hurt)} cases")
    print(f"CoT helped (direct wrong, CoT correct): {len(cot_helped)} cases")
    print(f"Net effect: {len(cot_helped) - len(cot_hurt)} (negative = CoT hurts)")

    if cot_hurt:
        patterns = analyze_cot_reasoning(cot_hurt)
        print(f"\nFailure patterns in CoT reasoning:")
        for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
            print(f"  {pattern}: {count} ({100*count/len(cot_hurt):.1f}%)")

    print_examples(cot_hurt, "Cases where CoT HURT performance")

    print("\n" + "="*70)
    print("FEW-SHOT ERROR ANALYSIS")
    print("="*70)

    # Few-shot analysis
    fs_hurt, fs_helped = find_few_shot_failures(data)
    print(f"\nFew-shot hurt (zero-shot correct, few-shot wrong): {len(fs_hurt)} cases")
    print(f"Few-shot helped (zero-shot wrong, few-shot correct): {len(fs_helped)} cases")
    print(f"Net effect: {len(fs_helped) - len(fs_hurt)} (negative = few-shot hurts)")

    print_examples(fs_hurt, "Cases where Few-shot HURT performance")

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    metrics = data.get('metrics', {})
    direct_acc = metrics.get('zero_shot_direct', {}).get('accuracy', 0)
    cot_acc = metrics.get('zero_shot_cot', {}).get('accuracy', 0)
    fs_acc = metrics.get('few_shot_3_direct', {}).get('accuracy', 0)

    print(f"\nAccuracy comparison:")
    print(f"  Zero-shot direct: {direct_acc:.1%}")
    print(f"  Zero-shot CoT: {cot_acc:.1%} (delta: {cot_acc - direct_acc:+.1%})")
    print(f"  Few-shot direct: {fs_acc:.1%} (delta: {fs_acc - direct_acc:+.1%})")

    print(f"\nCase-level analysis:")
    print(f"  CoT hurt {len(cot_hurt)} questions that direct got right")
    print(f"  CoT helped {len(cot_helped)} questions that direct got wrong")
    print(f"  Few-shot hurt {len(fs_hurt)} questions that zero-shot got right")
    print(f"  Few-shot helped {len(fs_helped)} questions that zero-shot got wrong")

    # Save detailed analysis
    analysis = {
        'cot_hurt_count': len(cot_hurt),
        'cot_helped_count': len(cot_helped),
        'cot_net_effect': len(cot_helped) - len(cot_hurt),
        'few_shot_hurt_count': len(fs_hurt),
        'few_shot_helped_count': len(fs_helped),
        'few_shot_net_effect': len(fs_helped) - len(fs_hurt),
        'cot_failure_patterns': analyze_cot_reasoning(cot_hurt) if cot_hurt else {},
        'example_cot_failures': cot_hurt[:10],
        'example_fs_failures': fs_hurt[:10],
    }

    output_path = os.path.join(RESULTS_DIR, 'error_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nDetailed analysis saved to: {output_path}")


if __name__ == "__main__":
    main()
