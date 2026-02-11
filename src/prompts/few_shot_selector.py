"""Few-shot example selection methods for medical QA experiments.

Provides different strategies for selecting few-shot examples from a
training pool, including random, label-balanced, and subject-matched
selection, as well as order permutation generation.
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional

from ..data.schemas import MCQItem


class FewShotSelector:
    """Selects few-shot examples from a training pool using various strategies."""

    @staticmethod
    def _mcq_item_to_example(item: MCQItem) -> Dict:
        """Convert an MCQItem to the dict format expected by _format_examples."""
        example = {
            "question": item.question,
            "options": item.options,
            "correct_answer": item.correct_answer,
            "subject": item.subject or "",
        }
        if item.explanation:
            example["reasoning"] = item.explanation
        return example

    @staticmethod
    def random_select(
        pool: List[MCQItem], n: int, seed: int = 42
    ) -> List[Dict]:
        """Randomly select n examples from a pool of training items.

        Args:
            pool: List of MCQItem objects (training split).
            n: Number of examples to select.
            seed: Random seed for reproducibility.

        Returns:
            List of example dicts in the format expected by the template.
        """
        rng = random.Random(seed)
        selected = rng.sample(pool, min(n, len(pool)))
        return [FewShotSelector._mcq_item_to_example(item) for item in selected]

    @staticmethod
    def label_balanced_select(
        pool: List[MCQItem], n: int, seed: int = 42
    ) -> List[Dict]:
        """Select examples with balanced answer labels (equal A/B/C/D).

        If n=4, pick 1 from each label. If n=3, pick from 3 different labels.
        Falls back to filling remaining slots randomly if a label bucket is empty.

        Args:
            pool: List of MCQItem objects (training split).
            n: Number of examples to select.
            seed: Random seed for reproducibility.

        Returns:
            List of example dicts with balanced label representation.
        """
        rng = random.Random(seed)
        labels = ["A", "B", "C", "D"]

        # Group pool items by correct answer label
        by_label = defaultdict(list)
        for item in pool:
            by_label[item.correct_answer].append(item)

        # Shuffle each bucket
        for label in labels:
            rng.shuffle(by_label[label])

        selected = []
        # Pick labels to use (if n < 4, randomly choose which labels to include)
        use_labels = rng.sample(labels, min(n, len(labels)))

        # Pick one from each selected label
        for label in use_labels:
            if by_label[label]:
                selected.append(by_label[label].pop())

        # If we still need more (n > 4), fill from remaining pool
        if len(selected) < n:
            used_ids = {item.id for item in selected}
            remaining = [item for item in pool if item.id not in used_ids]
            rng.shuffle(remaining)
            selected.extend(remaining[: n - len(selected)])

        # Shuffle final selection order
        rng.shuffle(selected)
        return [FewShotSelector._mcq_item_to_example(item) for item in selected[:n]]

    @staticmethod
    def subject_matched_select(
        pool: List[MCQItem],
        n: int,
        target_subject: str,
        seed: int = 42,
    ) -> List[Dict]:
        """Select examples from the same medical subject as the target question.

        Falls back to random selection if not enough subject matches exist.

        Args:
            pool: List of MCQItem objects (training split).
            n: Number of examples to select.
            target_subject: Subject of the target question to match.
            seed: Random seed for reproducibility.

        Returns:
            List of example dicts, preferring same-subject items.
        """
        rng = random.Random(seed)

        same_subject = [
            item for item in pool
            if item.subject and item.subject == target_subject
        ]
        rng.shuffle(same_subject)

        if len(same_subject) >= n:
            selected = same_subject[:n]
        else:
            # Use all same-subject items, fill rest randomly from other subjects
            selected = list(same_subject)
            used_ids = {item.id for item in selected}
            others = [item for item in pool if item.id not in used_ids]
            rng.shuffle(others)
            selected.extend(others[: n - len(selected)])

        return [FewShotSelector._mcq_item_to_example(item) for item in selected[:n]]

    @staticmethod
    def get_multiple_orderings(
        examples: List[Dict], n_orderings: int, seed: int = 42
    ) -> List[List[Dict]]:
        """Generate n_orderings different random orderings of the same examples.

        The first ordering is always the original order. Subsequent orderings
        are random permutations (guaranteed distinct from prior ones when possible).

        Args:
            examples: List of example dicts.
            n_orderings: Number of orderings to generate.
            seed: Random seed for reproducibility.

        Returns:
            List of lists, each being a different ordering of the input examples.
        """
        rng = random.Random(seed)
        orderings = [list(examples)]  # ordering 1 = original order

        seen = {tuple(range(len(examples)))}
        indices = list(range(len(examples)))

        attempts = 0
        max_attempts = n_orderings * 100
        while len(orderings) < n_orderings and attempts < max_attempts:
            perm = list(indices)
            rng.shuffle(perm)
            perm_tuple = tuple(perm)
            if perm_tuple not in seen:
                seen.add(perm_tuple)
                orderings.append([examples[i] for i in perm])
            attempts += 1

        return orderings
