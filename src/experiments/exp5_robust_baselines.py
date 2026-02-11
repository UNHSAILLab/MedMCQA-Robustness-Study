"""Experiment 5: Robust baselines -- CoT self-consistency, permutation voting, cloze scoring."""

from typing import Dict, Any, List
from collections import Counter
import logging
import numpy as np

from .base import BaseExperiment
from ..data.loaders import load_medmcqa
from ..data.schemas import MCQItem
from ..prompts.templates import MedMCQAPromptTemplate, PromptConfig, PromptStyle
from ..evaluation.metrics import parse_mcq_answer, MCQMetrics
from ..perturbations.option_shuffle import OptionShuffler

logger = logging.getLogger(__name__)


class CoTSelfConsistencyExperiment(BaseExperiment):
    """CoT self-consistency baseline.

    Explicitly uses chain-of-thought prompting with temperature sampling
    and majority vote aggregation. Serves as a distinct, documented baseline
    condition separate from the general SelfConsistencyExperiment in exp4.

    Strategy:
    - Use MedMCQA ZERO_SHOT_COT template
    - Sample N answers with temperature=0.7
    - Majority vote for final answer
    - Report accuracy at each N
    """

    SAMPLE_COUNTS = [3, 5, 10]

    @property
    def name(self) -> str:
        return "exp5_cot_self_consistency"

    def run(self) -> Dict[str, Any]:
        """Run CoT self-consistency on MedMCQA."""
        temperature = self.config.get('cot_sc', {}).get('temperature', 0.7)
        sample_counts = self.config.get('cot_sc', {}).get(
            'sample_counts', self.SAMPLE_COUNTS
        )
        limit = self.config.get('dataset', {}).get('limit')

        logger.info("Running CoT self-consistency baseline on MedMCQA...")
        data = load_medmcqa(split="validation", limit=limit)
        template = MedMCQAPromptTemplate()
        config = PromptConfig(style=PromptStyle.ZERO_SHOT_COT)

        results = {
            'items': [
                {'id': item.id, 'correct_answer': item.correct_answer}
                for item in data
            ]
        }

        generation_config = {
            'temperature': temperature,
            'do_sample': True,
            'max_new_tokens': 512,
        }

        from tqdm import tqdm
        for n_samples in sample_counts:
            logger.info(f"CoT SC with N={n_samples} samples, temp={temperature}")
            sc_results = []

            for item in tqdm(data, desc=f"CoT-SC (N={n_samples})"):
                prompt = template.format(item, config)

                samples = []
                for _ in range(n_samples):
                    outputs = self.model.generate([prompt], **generation_config)
                    samples.append(outputs[0])

                parsed = [parse_mcq_answer(s) for s in samples]
                vote_counts = Counter(parsed)
                majority_answer = vote_counts.most_common(1)[0][0]
                confidence = vote_counts[majority_answer] / n_samples

                sc_results.append({
                    'item_id': item.id,
                    'samples': samples,
                    'parsed_samples': parsed,
                    'vote_counts': dict(vote_counts),
                    'majority_answer': majority_answer,
                    'confidence': confidence,
                })

            results[f'n_{n_samples}'] = sc_results

        return results

    def analyze(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute accuracy at each sample count."""
        items = results.get('items', [])
        labels = [item['correct_answer'] for item in items]
        metrics = {}

        for key in results:
            if not key.startswith('n_'):
                continue

            n_value = int(key.split('_')[1])
            sc_results = results[key]

            predictions = [r['majority_answer'] for r in sc_results]
            confidences = [r['confidence'] for r in sc_results]

            accuracy = MCQMetrics.accuracy(predictions, labels)
            acc, ci_lo, ci_hi = MCQMetrics.accuracy_with_ci(predictions, labels)

            metrics[key] = {
                'n_samples': n_value,
                'accuracy': accuracy,
                'accuracy_ci_lower': ci_lo,
                'accuracy_ci_upper': ci_hi,
                'mean_confidence': float(np.mean(confidences)),
                'std_confidence': float(np.std(confidences)),
            }

        # Accuracy gain from increasing N
        if 'n_3' in metrics and 'n_10' in metrics:
            metrics['accuracy_gain_3_to_10'] = (
                metrics['n_10']['accuracy'] - metrics['n_3']['accuracy']
            )

        return metrics


class PermutationVoteExperiment(BaseExperiment):
    """Permutation-vote aggregation baseline.

    For each question, generate K permutations of option order, run greedy
    inference on each, map predicted answers back to the original label space,
    and aggregate by majority vote.

    Reports: aggregated accuracy, per-permutation accuracy variance, and
    agreement rate across permutations.
    """

    @property
    def name(self) -> str:
        return "exp5_permutation_vote"

    def run(self) -> Dict[str, Any]:
        """Run permutation-vote on MedMCQA."""
        limit = self.config.get('dataset', {}).get('limit')
        k_permutations = self.config.get('permutation_vote', {}).get('k', 6)
        use_all = self.config.get('permutation_vote', {}).get('use_all', False)

        logger.info("Running permutation-vote baseline on MedMCQA...")
        data = load_medmcqa(split="validation", limit=limit)
        template = MedMCQAPromptTemplate()
        config = PromptConfig(style=PromptStyle.ZERO_SHOT_DIRECT)
        shuffler = OptionShuffler(seed=42)

        generation_config = {
            'max_new_tokens': 256,
            'temperature': 0.0,
            'do_sample': False,
        }

        results = {
            'items': [
                {'id': item.id, 'correct_answer': item.correct_answer}
                for item in data
            ],
            'k_permutations': k_permutations,
            'use_all': use_all,
            'per_item': [],
        }

        from tqdm import tqdm
        for item in tqdm(data, desc="Permutation-vote"):
            if use_all:
                perms = shuffler.get_all_permutations(item)
            else:
                perms = self._sample_permutations(shuffler, item, k_permutations)

            perm_preds_original = []  # predictions mapped back to original space
            perm_preds_raw = []       # raw predictions in permuted space

            for perm_item, mapping in perms:
                prompt = template.format(perm_item, config)
                outputs = self.model.generate([prompt], **generation_config)
                raw_pred = parse_mcq_answer(outputs[0])
                perm_preds_raw.append(raw_pred)

                # Map back to original label space
                # mapping is old_key -> new_key, we need inverse: new_key -> old_key
                inverse_mapping = {v: k for k, v in mapping.items()}
                original_pred = inverse_mapping.get(raw_pred, raw_pred)
                perm_preds_original.append(original_pred)

            # Majority vote in original space
            vote_counts = Counter(perm_preds_original)
            majority_answer = vote_counts.most_common(1)[0][0]

            # Agreement rate: fraction of permutations that agree with majority
            agreement_rate = vote_counts[majority_answer] / len(perm_preds_original)

            results['per_item'].append({
                'item_id': item.id,
                'perm_predictions_original': perm_preds_original,
                'perm_predictions_raw': perm_preds_raw,
                'vote_counts': dict(vote_counts),
                'majority_answer': majority_answer,
                'agreement_rate': agreement_rate,
            })

        return results

    def _sample_permutations(
        self, shuffler: OptionShuffler, item: MCQItem, k: int
    ) -> list:
        """Sample K permutations using different seeds."""
        all_perms = shuffler.get_all_permutations(item)
        # Deterministically select K from the 24
        rng = np.random.default_rng(42)
        indices = rng.choice(len(all_perms), size=min(k, len(all_perms)), replace=False)
        return [all_perms[i] for i in sorted(indices)]

    def analyze(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute aggregated accuracy, per-permutation variance, agreement."""
        items = results.get('items', [])
        labels = [item['correct_answer'] for item in items]
        per_item = results.get('per_item', [])

        # Aggregated (majority-vote) accuracy
        agg_predictions = [r['majority_answer'] for r in per_item]
        agg_accuracy = MCQMetrics.accuracy(agg_predictions, labels)
        agg_acc, agg_ci_lo, agg_ci_hi = MCQMetrics.accuracy_with_ci(agg_predictions, labels)

        # Per-permutation accuracies (for variance calculation)
        n_perms = len(per_item[0]['perm_predictions_original']) if per_item else 0
        perm_accuracies = []
        for perm_idx in range(n_perms):
            perm_preds = [r['perm_predictions_original'][perm_idx] for r in per_item]
            perm_accuracies.append(MCQMetrics.accuracy(perm_preds, labels))

        agreement_rates = [r['agreement_rate'] for r in per_item]

        metrics = {
            'aggregated_accuracy': agg_accuracy,
            'aggregated_accuracy_ci_lower': agg_ci_lo,
            'aggregated_accuracy_ci_upper': agg_ci_hi,
            'per_permutation_accuracies': perm_accuracies,
            'per_permutation_accuracy_mean': float(np.mean(perm_accuracies)) if perm_accuracies else 0.0,
            'per_permutation_accuracy_std': float(np.std(perm_accuracies)) if perm_accuracies else 0.0,
            'mean_agreement_rate': float(np.mean(agreement_rates)) if agreement_rates else 0.0,
            'std_agreement_rate': float(np.std(agreement_rates)) if agreement_rates else 0.0,
            'n_permutations': n_perms,
        }

        return metrics


class ClozeScoreExperiment(BaseExperiment):
    """Cloze (token-level logprob) scoring baseline.

    For each question, present the question and use generate_with_logprobs()
    to score each option token. The option with the highest logprob is
    selected as the answer. Tests whether token-level scoring outperforms
    generative answering.
    """

    CLOZE_PROMPT = """Question: {question}

Options:
A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}

The correct answer is:"""

    @property
    def name(self) -> str:
        return "exp5_cloze_score"

    def run(self) -> Dict[str, Any]:
        """Run cloze scoring on MedMCQA."""
        limit = self.config.get('dataset', {}).get('limit')

        logger.info("Running cloze-score baseline on MedMCQA...")
        data = load_medmcqa(split="validation", limit=limit)

        results = {
            'items': [
                {'id': item.id, 'correct_answer': item.correct_answer}
                for item in data
            ],
            'per_item': [],
        }

        from tqdm import tqdm
        for item in tqdm(data, desc="Cloze scoring"):
            prompt = self.CLOZE_PROMPT.format(
                question=item.question,
                option_a=item.options.get('A', ''),
                option_b=item.options.get('B', ''),
                option_c=item.options.get('C', ''),
                option_d=item.options.get('D', ''),
            )

            logprobs = self.model.generate_with_logprobs(
                prompt=prompt,
                choices=['A', 'B', 'C', 'D'],
            )

            # Select highest logprob option
            predicted = max(logprobs, key=logprobs.get)

            results['per_item'].append({
                'item_id': item.id,
                'logprobs': logprobs,
                'predicted_answer': predicted,
            })

        return results

    def analyze(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute cloze scoring accuracy."""
        items = results.get('items', [])
        labels = [item['correct_answer'] for item in items]
        per_item = results.get('per_item', [])

        predictions = [r['predicted_answer'] for r in per_item]
        accuracy = MCQMetrics.accuracy(predictions, labels)
        acc, ci_lo, ci_hi = MCQMetrics.accuracy_with_ci(predictions, labels)

        # Analyze logprob distribution
        all_logprobs = [r['logprobs'] for r in per_item]
        mean_margins = []
        for lp in all_logprobs:
            sorted_vals = sorted(lp.values(), reverse=True)
            if len(sorted_vals) >= 2:
                mean_margins.append(sorted_vals[0] - sorted_vals[1])

        metrics = {
            'accuracy': accuracy,
            'accuracy_ci_lower': ci_lo,
            'accuracy_ci_upper': ci_hi,
            'mean_logprob_margin': float(np.mean(mean_margins)) if mean_margins else 0.0,
            'std_logprob_margin': float(np.std(mean_margins)) if mean_margins else 0.0,
            'position_bias': MCQMetrics.position_bias(predictions, labels),
        }

        return metrics
