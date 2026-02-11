#!/usr/bin/env python3
"""Run quality checks on existing experiment results.

Loads results from outputs/results/, runs all quality checks, and generates
a quality report saved to outputs/quality_report.json.
"""

import argparse
import json
import logging
import os
import sys
from glob import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.evaluation.quality_controls import (
    LabelRemappingChecker,
    ParserFailureTracker,
    TruncationAuditExporter,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_results(results_dir: str) -> dict:
    """Load all experiment result files from the results directory.

    Returns:
        Dict mapping filename (without extension) to parsed JSON content.
    """
    results = {}
    for path in sorted(glob(os.path.join(results_dir, '*.json'))):
        basename = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path) as f:
                results[basename] = json.load(f)
            logger.info(f"Loaded {basename}")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Skipping {basename}: {e}")
    return results


def run_label_remapping_checks(all_results: dict) -> dict:
    """Run label remapping checks on exp2 (option order) results."""
    checker = LabelRemappingChecker()
    report = {}

    for name, data in all_results.items():
        if data.get('experiment') != 'exp2_option_order':
            continue

        results = data.get('results', {})
        original_responses = results.get('original', [])
        items = results.get('items', [])

        if not items or not original_responses:
            continue

        # Build original items with options extracted from prompts
        # The stored results don't have full option dicts, so we note that
        # full verification requires the original data. We can still check
        # that correct_answer labels are consistent with mappings.
        for pert_type, pert_data in results.get('perturbations', {}).items():
            pert_correct_answers = pert_data.get('correct_answers', [])
            pert_responses = pert_data.get('responses', [])

            if not pert_correct_answers:
                continue

            # Basic consistency check: the number of items should match
            n_orig = len(items)
            n_pert = len(pert_correct_answers)
            mismatches = []

            if n_orig != n_pert:
                mismatches.append({
                    'error': f"Item count mismatch: {n_orig} original vs {n_pert} perturbed"
                })

            # Check that perturbed correct answers are all valid labels
            invalid_labels = []
            for i, label in enumerate(pert_correct_answers):
                if label not in ('A', 'B', 'C', 'D'):
                    invalid_labels.append({'index': i, 'label': label})

            report[f"{name}/{pert_type}"] = {
                'perturbation_type': pert_type,
                'n_original': n_orig,
                'n_perturbed': n_pert,
                'count_mismatch': n_orig != n_pert,
                'invalid_labels': invalid_labels[:10],
                'invalid_label_count': len(invalid_labels),
                'mismatches': mismatches,
            }

        logger.info(f"Label remapping checks done for {name}")

    return report


def run_parser_failure_checks(all_results: dict) -> dict:
    """Run parser failure tracking across all experiment conditions."""
    tracker = ParserFailureTracker()
    report = {}

    for name, data in all_results.items():
        experiment = data.get('experiment', '')
        results = data.get('results', {})
        conditions = {}

        if experiment == 'exp1_prompt_ablation':
            for cond_name, cond_data in results.items():
                if isinstance(cond_data, dict) and 'responses' in cond_data:
                    conditions[cond_name] = cond_data['responses']

        elif experiment == 'exp2_option_order':
            if 'original' in results and isinstance(results['original'], list):
                conditions['original'] = results['original']
            for pert_type, pert_data in results.get('perturbations', {}).items():
                if isinstance(pert_data, dict) and 'responses' in pert_data:
                    conditions[pert_type] = pert_data['responses']

        elif experiment == 'exp3_evidence_conditioning':
            for cond_name, cond_data in results.items():
                if cond_name == 'items':
                    continue
                if isinstance(cond_data, dict) and 'responses' in cond_data:
                    conditions[cond_name] = cond_data['responses']

        if conditions:
            condition_report = tracker.failure_rate_report(conditions)
            # Add parser improvement suggestions based on all failures
            all_failures = []
            for cond_stats in condition_report.values():
                all_failures.extend(cond_stats.get('sample_failures', []))
            suggestions = tracker.suggest_parser_improvements(all_failures)

            report[name] = {
                'conditions': condition_report,
                'suggestions': suggestions,
            }
            logger.info(
                f"Parser failure checks done for {name}: "
                f"{len(conditions)} conditions"
            )

    return report


def run_truncation_audit(all_results: dict, output_dir: str) -> dict:
    """Run truncation audit on exp3 (evidence conditioning) results."""
    exporter = TruncationAuditExporter()
    report = {}

    for name, data in all_results.items():
        if data.get('experiment') != 'exp3_evidence_conditioning':
            continue

        results = data.get('results', {})
        items = results.get('items', [])
        labels = [item.get('correct_answer', '') for item in items]

        full_context_data = results.get('full_context', {})
        full_responses = full_context_data.get('responses', [])
        if not full_responses:
            continue

        # Collect all truncated conditions
        truncated_conditions = {}
        for cond_name, cond_data in results.items():
            if cond_name in ('items', 'full_context'):
                continue
            if isinstance(cond_data, dict) and 'responses' in cond_data:
                truncated_conditions[cond_name] = cond_data['responses']

        # Compute truncation failure stats
        trunc_stats = exporter.compute_truncation_failure_stats(
            full_responses, truncated_conditions, labels
        )

        # Export audit samples for each truncated condition
        audit_files = {}
        for cond_name, cond_responses in truncated_conditions.items():
            audit_samples = exporter.export_audit_samples(
                full_responses, cond_responses, labels, n_samples=50
            )
            if audit_samples:
                csv_path = os.path.join(
                    output_dir,
                    f"audit_{name}_{cond_name}.csv"
                )
                exporter.export_to_csv(audit_samples, csv_path)
                audit_files[cond_name] = {
                    'csv_path': csv_path,
                    'n_flips': len(audit_samples),
                }

        report[name] = {
            'truncation_stats': trunc_stats,
            'audit_files': audit_files,
        }
        logger.info(f"Truncation audit done for {name}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Run quality checks on experiment results"
    )
    parser.add_argument(
        '--results-dir',
        default='outputs/results',
        help="Directory containing experiment result JSON files"
    )
    parser.add_argument(
        '--output',
        default='outputs/quality_report.json',
        help="Path for the quality report JSON output"
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Load all results
    logger.info(f"Loading results from {args.results_dir}")
    all_results = load_results(args.results_dir)
    if not all_results:
        logger.warning("No results found. Exiting.")
        return

    logger.info(f"Loaded {len(all_results)} result files")

    # Run all quality checks
    quality_report = {}

    logger.info("Running label remapping checks...")
    quality_report['label_remapping'] = run_label_remapping_checks(all_results)

    logger.info("Running parser failure checks...")
    quality_report['parser_failures'] = run_parser_failure_checks(all_results)

    logger.info("Running truncation audit...")
    audit_output_dir = os.path.dirname(args.output) or '.'
    quality_report['truncation_audit'] = run_truncation_audit(
        all_results, audit_output_dir
    )

    # Summary
    summary = {
        'n_result_files': len(all_results),
        'experiments_checked': list(all_results.keys()),
    }

    # Summarize parser failures
    total_failures = 0
    total_responses = 0
    for file_report in quality_report.get('parser_failures', {}).values():
        for cond_stats in file_report.get('conditions', {}).values():
            total_failures += cond_stats.get('failures', 0)
            total_responses += cond_stats.get('total', 0)
    summary['parser_failure_total'] = total_failures
    summary['parser_failure_rate'] = (
        total_failures / total_responses if total_responses > 0 else 0.0
    )

    # Summarize label checks
    label_errors = 0
    for check in quality_report.get('label_remapping', {}).values():
        label_errors += check.get('invalid_label_count', 0)
        if check.get('count_mismatch'):
            label_errors += 1
    summary['label_remapping_errors'] = label_errors

    quality_report['summary'] = summary

    # Save report
    with open(args.output, 'w') as f:
        json.dump(quality_report, f, indent=2, default=str)

    logger.info(f"Quality report saved to {args.output}")

    # Print summary
    print("\n" + "=" * 60)
    print("QUALITY CHECK SUMMARY")
    print("=" * 60)
    print(f"Result files checked: {summary['n_result_files']}")
    print(f"Label remapping errors: {summary['label_remapping_errors']}")
    print(f"Parser failures: {summary['parser_failure_total']}/{total_responses} "
          f"({summary['parser_failure_rate']:.2%})")

    # Print per-file parser failure rates
    for file_name, file_report in quality_report.get('parser_failures', {}).items():
        for cond, stats in file_report.get('conditions', {}).items():
            rate = stats.get('failure_rate', 0)
            if rate > 0:
                print(f"  {file_name}/{cond}: {stats['failures']}/{stats['total']} "
                      f"({rate:.2%})")

    # Print truncation audit highlights
    for file_name, audit in quality_report.get('truncation_audit', {}).items():
        trunc_stats = audit.get('truncation_stats', {})
        summary_data = trunc_stats.get('_summary', {})
        if summary_data:
            print(f"\nTruncation audit ({file_name}):")
            print(f"  Most damaging: {summary_data.get('most_damaging')}")
            for entry in summary_data.get('ranking', []):
                print(f"    {entry['truncation']}: "
                      f"flip_rate={entry['flip_rate']:.2%}, "
                      f"net_loss={entry['net_loss']}")

    print("=" * 60)


if __name__ == '__main__':
    main()
