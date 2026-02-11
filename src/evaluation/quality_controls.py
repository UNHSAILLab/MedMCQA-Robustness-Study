"""Quality controls for experiment validation: label remapping, parser failures, audit support."""

import csv
import re
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple


class LabelRemappingChecker:
    """Verify that option shuffling/rotation correctly remaps answer labels."""

    OPTION_KEYS = ['A', 'B', 'C', 'D']

    def verify_shuffle_mapping(
        self,
        original_item: Dict[str, Any],
        perturbed_item: Dict[str, Any],
        mapping: Dict[str, str]
    ) -> Tuple[bool, str]:
        """Verify that a shuffle mapping correctly preserves answer-text correspondence.

        Checks that:
        1. The text of the correct option is the same before and after shuffling.
        2. The correct_answer label was properly remapped via the mapping.
        3. All option texts are preserved (just reordered).

        Args:
            original_item: Original item dict with 'options' and 'correct_answer'.
            perturbed_item: Perturbed item dict with 'options' and 'correct_answer'.
            mapping: Dict mapping old_key -> new_key.

        Returns:
            Tuple of (is_valid, error_message). error_message is empty if valid.
        """
        orig_options = original_item['options']
        pert_options = perturbed_item['options']
        orig_correct = original_item['correct_answer']
        pert_correct = perturbed_item['correct_answer']

        # Check that the mapping remaps the correct answer label properly
        expected_new_label = mapping.get(orig_correct)
        if expected_new_label is None:
            return False, f"Original correct answer '{orig_correct}' not in mapping keys"
        if expected_new_label != pert_correct:
            return False, (
                f"Mapping says {orig_correct}->{expected_new_label} "
                f"but perturbed correct_answer is '{pert_correct}'"
            )

        # Check that the correct option text is preserved at the new position
        orig_correct_text = orig_options.get(orig_correct, '')
        pert_correct_text = pert_options.get(pert_correct, '')
        if orig_correct_text != pert_correct_text:
            return False, (
                f"Correct option text mismatch: "
                f"original[{orig_correct}]='{orig_correct_text[:60]}' vs "
                f"perturbed[{pert_correct}]='{pert_correct_text[:60]}'"
            )

        # Check that all option texts are preserved (just reordered)
        orig_texts = sorted(orig_options.get(k, '') for k in self.OPTION_KEYS)
        pert_texts = sorted(pert_options.get(k, '') for k in self.OPTION_KEYS)
        if orig_texts != pert_texts:
            return False, "Option texts are not a permutation of the originals"

        # Check each mapping entry
        for old_key, new_key in mapping.items():
            if old_key not in self.OPTION_KEYS or new_key not in self.OPTION_KEYS:
                return False, f"Invalid mapping entry: {old_key}->{new_key}"
            orig_text = orig_options.get(old_key, '')
            pert_text = pert_options.get(new_key, '')
            if orig_text != pert_text:
                return False, (
                    f"Mapping {old_key}->{new_key} text mismatch: "
                    f"'{orig_text[:60]}' vs '{pert_text[:60]}'"
                )

        return True, ""

    def verify_rotation_mapping(
        self,
        original_item: Dict[str, Any],
        perturbed_item: Dict[str, Any],
        rotation: int
    ) -> Tuple[bool, str]:
        """Verify that a rotation correctly preserves answer-text correspondence.

        Args:
            original_item: Original item dict with 'options' and 'correct_answer'.
            perturbed_item: Perturbed item dict with 'options' and 'correct_answer'.
            rotation: Number of positions rotated.

        Returns:
            Tuple of (is_valid, error_message).
        """
        keys = self.OPTION_KEYS
        rotation = rotation % 4

        # Reconstruct expected mapping from rotation
        mapping = {}
        for i, key in enumerate(keys):
            new_idx = (i - rotation) % 4
            mapping[key] = keys[new_idx]

        return self.verify_shuffle_mapping(original_item, perturbed_item, mapping)

    def batch_verify(
        self,
        original_items: List[Dict[str, Any]],
        perturbed_items: List[Dict[str, Any]],
        mappings: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Run verification on an entire dataset.

        Args:
            original_items: List of original item dicts.
            perturbed_items: List of perturbed item dicts.
            mappings: List of mapping dicts (old_key -> new_key).

        Returns:
            Dict with 'error_count', 'total', 'error_rate', and 'problematic_items'.
        """
        errors = []
        for i, (orig, pert, mapping) in enumerate(
            zip(original_items, perturbed_items, mappings)
        ):
            is_valid, error_msg = self.verify_shuffle_mapping(orig, pert, mapping)
            if not is_valid:
                errors.append({
                    'index': i,
                    'item_id': orig.get('id', f'item_{i}'),
                    'error': error_msg,
                    'original_correct': orig.get('correct_answer'),
                    'perturbed_correct': pert.get('correct_answer'),
                    'mapping': mapping,
                })

        total = len(original_items)
        return {
            'error_count': len(errors),
            'total': total,
            'error_rate': len(errors) / total if total > 0 else 0.0,
            'problematic_items': errors,
        }


class ParserFailureTracker:
    """Track and analyze parser failures across experiment conditions."""

    UNKNOWN_VALUES = {'UNKNOWN', 'unknown', None}

    def track_failures(
        self,
        responses: List[Dict[str, Any]],
        condition_name: str
    ) -> Dict[str, Any]:
        """Count how many responses parsed as UNKNOWN/unknown per condition.

        Args:
            responses: List of response dicts with 'parsed_answer' and 'raw_output'.
            condition_name: Name of the experiment condition.

        Returns:
            Dict with total, failures, failure_rate, and sample_failures.
        """
        failures = []
        for r in responses:
            parsed = r.get('parsed_answer')
            if parsed in self.UNKNOWN_VALUES:
                failures.append(r.get('raw_output', ''))

        total = len(responses)
        return {
            'condition': condition_name,
            'total': total,
            'failures': len(failures),
            'failure_rate': len(failures) / total if total > 0 else 0.0,
            'sample_failures': failures[:10],
        }

    def failure_rate_report(
        self,
        all_conditions_results: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Dict[str, Any]]:
        """Generate a report mapping condition_name to failure stats.

        Args:
            all_conditions_results: Dict mapping condition_name to list of
                response dicts.

        Returns:
            Dict mapping condition_name -> {total, failures, failure_rate, sample_failures}.
        """
        report = {}
        for condition_name, responses in all_conditions_results.items():
            report[condition_name] = self.track_failures(responses, condition_name)
        return report

    def suggest_parser_improvements(
        self,
        failures: List[str]
    ) -> List[Dict[str, str]]:
        """Analyze common failure patterns and suggest regex improvements.

        Args:
            failures: List of raw output strings that failed to parse.

        Returns:
            List of dicts with 'pattern', 'count', 'example', and 'suggestion'.
        """
        if not failures:
            return []

        patterns = {
            'option_text_answer': {
                'regex': re.compile(
                    r'(?:the\s+)?(?:correct\s+)?answer\s+is\s+(?:option\s+)?([A-Da-d])\b',
                    re.IGNORECASE
                ),
                'suggestion': (
                    r"Add pattern: re.search(r'answer\\s+is\\s+(?:option\\s+)?([A-Da-d])', output, re.I)"
                ),
            },
            'parenthesized_letter': {
                'regex': re.compile(r'\(([A-Da-d])\)'),
                'suggestion': (
                    r"Add pattern: re.search(r'\\(([A-Da-d])\\)', output)"
                ),
            },
            'option_colon': {
                'regex': re.compile(
                    r'(?:option|choice)\s*[:\-]\s*([A-Da-d])\b', re.IGNORECASE
                ),
                'suggestion': (
                    r"Add pattern: re.search(r'(?:option|choice)\\s*[:/-]\\s*([A-Da-d])', output, re.I)"
                ),
            },
            'yes_no_maybe_variant': {
                'regex': re.compile(
                    r'\b(yes|no|maybe|possibly|uncertain|inconclusive)\b',
                    re.IGNORECASE
                ),
                'suggestion': (
                    "Map 'possibly' -> 'maybe', 'uncertain' -> 'maybe', "
                    "'inconclusive' -> 'maybe' in parse_pubmedqa_answer"
                ),
            },
            'full_option_text': {
                'regex': re.compile(r'^[^A-Da-d]{20,}$'),
                'suggestion': (
                    "Model returned full option text instead of letter. "
                    "Consider fuzzy-matching output against option texts."
                ),
            },
            'empty_or_whitespace': {
                'regex': re.compile(r'^\s*$'),
                'suggestion': (
                    "Empty/whitespace output. Check max_new_tokens or "
                    "generation config for premature stopping."
                ),
            },
        }

        results = []
        pattern_counts = Counter()
        pattern_examples = {}

        for raw_output in failures:
            matched_any = False
            for name, spec in patterns.items():
                if spec['regex'].search(raw_output):
                    pattern_counts[name] += 1
                    if name not in pattern_examples:
                        pattern_examples[name] = raw_output[:200]
                    matched_any = True
            if not matched_any:
                pattern_counts['unclassified'] += 1
                if 'unclassified' not in pattern_examples:
                    pattern_examples['unclassified'] = raw_output[:200]

        for name, count in pattern_counts.most_common():
            suggestion = (
                patterns[name]['suggestion'] if name in patterns
                else "Review these outputs manually for new parse patterns."
            )
            results.append({
                'pattern': name,
                'count': count,
                'example': pattern_examples.get(name, ''),
                'suggestion': suggestion,
            })

        return results


class TruncationAuditExporter:
    """Export audit samples where truncation changed the model's answer."""

    def export_audit_samples(
        self,
        full_context_results: List[Dict[str, Any]],
        truncated_results: List[Dict[str, Any]],
        labels: List[str],
        n_samples: int = 50
    ) -> List[Dict[str, Any]]:
        """Export samples where truncation changed the model's answer.

        Args:
            full_context_results: List of response dicts from full-context condition.
            truncated_results: List of response dicts from truncated condition.
            labels: List of correct answer strings.
            n_samples: Maximum number of flip samples to export.

        Returns:
            List of audit sample dicts with question, contexts, answers.
        """
        audit_samples = []

        for i, (full_r, trunc_r) in enumerate(
            zip(full_context_results, truncated_results)
        ):
            full_answer = full_r.get('parsed_answer', '')
            trunc_answer = trunc_r.get('parsed_answer', '')
            correct = labels[i] if i < len(labels) else ''

            if full_answer != trunc_answer:
                sample = {
                    'index': i,
                    'item_id': full_r.get('item_id', ''),
                    'question': self._extract_question(full_r.get('prompt', '')),
                    'full_context_prompt': full_r.get('prompt', ''),
                    'truncated_prompt': trunc_r.get('prompt', ''),
                    'full_context_answer': full_answer,
                    'truncated_answer': trunc_answer,
                    'correct_answer': correct,
                    'full_correct': full_answer == correct,
                    'truncated_correct': trunc_answer == correct,
                    'full_raw_output': full_r.get('raw_output', '')[:500],
                    'truncated_raw_output': trunc_r.get('raw_output', '')[:500],
                }
                audit_samples.append(sample)

                if len(audit_samples) >= n_samples:
                    break

        return audit_samples

    def export_to_csv(
        self,
        audit_samples: List[Dict[str, Any]],
        output_path: str
    ) -> str:
        """Save audit samples as CSV for manual review.

        Args:
            audit_samples: List of audit sample dicts.
            output_path: Path to write the CSV file.

        Returns:
            The output path written to.
        """
        if not audit_samples:
            return output_path

        fieldnames = [
            'index', 'item_id', 'question',
            'full_context_prompt', 'truncated_prompt',
            'full_context_answer', 'truncated_answer', 'correct_answer',
            'full_correct', 'truncated_correct',
            'full_raw_output', 'truncated_raw_output',
        ]

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for sample in audit_samples:
                writer.writerow(sample)

        return output_path

    def compute_truncation_failure_stats(
        self,
        full_results: List[Dict[str, Any]],
        truncated_results: Dict[str, List[Dict[str, Any]]],
        labels: List[str]
    ) -> Dict[str, Any]:
        """Compute stats on what types of truncation cause the most failures.

        Args:
            full_results: Response dicts from full-context condition.
            truncated_results: Dict mapping truncation_name -> list of response dicts.
                e.g. {'truncated_50': [...], 'truncated_25': [...]}
            labels: List of correct answer strings.

        Returns:
            Dict with per-truncation-type stats and overall summary.
        """
        stats = {}

        for trunc_name, trunc_responses in truncated_results.items():
            flips = 0
            correct_to_wrong = 0
            wrong_to_correct = 0
            total = min(len(full_results), len(trunc_responses))

            for i in range(total):
                full_ans = full_results[i].get('parsed_answer', '')
                trunc_ans = trunc_responses[i].get('parsed_answer', '')
                correct = labels[i] if i < len(labels) else ''

                if full_ans != trunc_ans:
                    flips += 1
                    if full_ans == correct and trunc_ans != correct:
                        correct_to_wrong += 1
                    elif full_ans != correct and trunc_ans == correct:
                        wrong_to_correct += 1

            stats[trunc_name] = {
                'total': total,
                'flips': flips,
                'flip_rate': flips / total if total > 0 else 0.0,
                'correct_to_wrong': correct_to_wrong,
                'wrong_to_correct': wrong_to_correct,
                'net_loss': correct_to_wrong - wrong_to_correct,
                'net_loss_rate': (correct_to_wrong - wrong_to_correct) / total if total > 0 else 0.0,
            }

        # Summary: rank truncation types by damage
        if stats:
            ranked = sorted(stats.items(), key=lambda x: x[1]['net_loss'], reverse=True)
            stats['_summary'] = {
                'most_damaging': ranked[0][0] if ranked else None,
                'least_damaging': ranked[-1][0] if ranked else None,
                'ranking': [
                    {'truncation': name, 'net_loss': s['net_loss'], 'flip_rate': s['flip_rate']}
                    for name, s in ranked
                ],
            }

        return stats

    @staticmethod
    def _extract_question(prompt: str) -> str:
        """Extract the question text from a prompt string."""
        # Try to extract question between "Question:" and "Options:" or "Context:"
        match = re.search(
            r'(?:Question:\s*)(.*?)(?:\n\s*(?:Options|Context|A\)|A\.))',
            prompt, re.DOTALL
        )
        if match:
            return match.group(1).strip()
        # Fallback: first line
        lines = prompt.strip().split('\n')
        return lines[0] if lines else prompt[:200]
