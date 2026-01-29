"""Option shuffling perturbations for MCQ robustness testing."""

import random
import copy
from typing import Dict, List, Tuple
from itertools import permutations

from ..data.schemas import MCQItem


class OptionShuffler:
    """Generate option order perturbations for MCQ robustness testing."""

    OPTION_KEYS = ['A', 'B', 'C', 'D']

    def __init__(self, seed: int = 42):
        """Initialize shuffler with random seed.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)

    def shuffle_random(self, item: MCQItem) -> Tuple[MCQItem, Dict[str, str]]:
        """Random shuffle of all options.

        Args:
            item: Original MCQ item

        Returns:
            Tuple of (perturbed_item, mapping) where mapping shows old->new positions
        """
        options = item.options
        keys = self.OPTION_KEYS
        values = [options[k] for k in keys]

        # Create shuffled order
        indices = list(range(4))
        self.rng.shuffle(indices)

        # Build new options and mapping
        new_options = {}
        mapping = {}  # old_key -> new_key

        for new_idx, orig_idx in enumerate(indices):
            new_key = keys[new_idx]
            orig_key = keys[orig_idx]
            new_options[new_key] = values[orig_idx]
            mapping[orig_key] = new_key

        # Create perturbed item
        perturbed = item.model_copy(deep=True)
        perturbed.options = new_options
        perturbed.correct_answer = mapping[item.correct_answer]
        perturbed.original_correct = item.correct_answer
        perturbed.perturbation = {
            'type': 'shuffle_random',
            'mapping': mapping,
            'original_correct': item.correct_answer
        }

        return perturbed, mapping

    def rotate_options(
        self,
        item: MCQItem,
        positions: int = 1
    ) -> Tuple[MCQItem, Dict[str, str]]:
        """Rotate options by N positions (cyclic shift).

        Args:
            item: Original MCQ item
            positions: Number of positions to rotate

        Returns:
            Tuple of (perturbed_item, mapping)
        """
        options = item.options
        keys = self.OPTION_KEYS
        values = [options[k] for k in keys]

        # Rotate values
        positions = positions % 4
        rotated = values[-positions:] + values[:-positions]

        # Build new options and mapping
        new_options = {}
        mapping = {}

        for new_idx, key in enumerate(keys):
            new_options[key] = rotated[new_idx]
            # Find which original key this value came from
            orig_idx = (new_idx + positions) % 4
            mapping[keys[orig_idx]] = key

        perturbed = item.model_copy(deep=True)
        perturbed.options = new_options
        perturbed.correct_answer = mapping[item.correct_answer]
        perturbed.original_correct = item.correct_answer
        perturbed.perturbation = {
            'type': f'rotate_{positions}',
            'positions': positions,
            'mapping': mapping,
            'original_correct': item.correct_answer
        }

        return perturbed, mapping

    def get_all_permutations(self, item: MCQItem) -> List[Tuple[MCQItem, Dict[str, str]]]:
        """Generate all 24 permutations of options.

        Useful for complete position sensitivity analysis.

        Args:
            item: Original MCQ item

        Returns:
            List of (perturbed_item, mapping) tuples for all 24 permutations
        """
        options = item.options
        keys = self.OPTION_KEYS
        values = [options[k] for k in keys]

        results = []

        for perm in permutations(range(4)):
            new_options = {}
            mapping = {}

            for new_idx, orig_idx in enumerate(perm):
                new_key = keys[new_idx]
                orig_key = keys[orig_idx]
                new_options[new_key] = values[orig_idx]
                mapping[orig_key] = new_key

            perturbed = item.model_copy(deep=True)
            perturbed.options = new_options
            perturbed.correct_answer = mapping[item.correct_answer]
            perturbed.original_correct = item.correct_answer
            perturbed.perturbation = {
                'type': 'all_permutations',
                'permutation': perm,
                'mapping': mapping,
                'original_correct': item.correct_answer
            }

            results.append((perturbed, mapping))

        return results

    def move_correct_to_position(
        self,
        item: MCQItem,
        target_position: str
    ) -> Tuple[MCQItem, Dict[str, str]]:
        """Move the correct answer to a specific position.

        Useful for testing position bias (e.g., always put correct at A).

        Args:
            item: Original MCQ item
            target_position: Target position for correct answer ('A', 'B', 'C', 'D')

        Returns:
            Tuple of (perturbed_item, mapping)
        """
        if target_position not in self.OPTION_KEYS:
            raise ValueError(f"target_position must be in {self.OPTION_KEYS}")

        current_pos = item.correct_answer

        if current_pos == target_position:
            # Already in target position, return copy
            perturbed = item.model_copy(deep=True)
            mapping = {k: k for k in self.OPTION_KEYS}
            perturbed.perturbation = {
                'type': 'move_correct',
                'target': target_position,
                'mapping': mapping
            }
            return perturbed, mapping

        # Swap current position with target position
        options = dict(item.options)
        options[current_pos], options[target_position] = (
            options[target_position], options[current_pos]
        )

        mapping = {k: k for k in self.OPTION_KEYS}
        mapping[current_pos] = target_position
        mapping[target_position] = current_pos

        perturbed = item.model_copy(deep=True)
        perturbed.options = options
        perturbed.correct_answer = target_position
        perturbed.original_correct = item.correct_answer
        perturbed.perturbation = {
            'type': 'move_correct',
            'target': target_position,
            'mapping': mapping,
            'original_correct': item.correct_answer
        }

        return perturbed, mapping


class DistractorSwapper:
    """Swap positions of distractor options (incorrect answers)."""

    OPTION_KEYS = ['A', 'B', 'C', 'D']

    def swap_distractors(
        self,
        item: MCQItem,
        swap_pair: Tuple[str, str] = None
    ) -> Tuple[MCQItem, Dict[str, str]]:
        """Swap two distractor positions.

        Args:
            item: Original MCQ item
            swap_pair: Tuple of positions to swap (e.g., ('A', 'B')).
                      If None, swaps first two distractors.

        Returns:
            Tuple of (perturbed_item, mapping)
        """
        correct = item.correct_answer
        distractors = [k for k in self.OPTION_KEYS if k != correct]

        if swap_pair is None:
            # Default: swap first two distractors
            d1, d2 = distractors[0], distractors[1]
        else:
            d1, d2 = swap_pair
            if d1 == correct or d2 == correct:
                raise ValueError("Cannot swap with correct answer position")

        # Swap the distractor positions
        options = dict(item.options)
        options[d1], options[d2] = options[d2], options[d1]

        mapping = {k: k for k in self.OPTION_KEYS}
        mapping[d1] = d2
        mapping[d2] = d1

        perturbed = item.model_copy(deep=True)
        perturbed.options = options
        # Correct answer position unchanged
        perturbed.perturbation = {
            'type': 'distractor_swap',
            'swapped': (d1, d2),
            'mapping': mapping
        }

        return perturbed, mapping

    def get_all_distractor_swaps(
        self,
        item: MCQItem
    ) -> List[Tuple[MCQItem, str]]:
        """Generate all pairwise swaps of distractors.

        For 3 distractors, generates 3 variants.

        Args:
            item: Original MCQ item

        Returns:
            List of (perturbed_item, swap_name) tuples
        """
        correct = item.correct_answer
        distractors = [k for k in self.OPTION_KEYS if k != correct]

        results = []
        for i in range(len(distractors)):
            for j in range(i + 1, len(distractors)):
                d1, d2 = distractors[i], distractors[j]
                perturbed, _ = self.swap_distractors(item, (d1, d2))
                results.append((perturbed, f"swap_{d1}_{d2}"))

        return results


def generate_shuffled_dataset(
    items: List[MCQItem],
    seed: int = 42
) -> List[MCQItem]:
    """Generate one shuffled version per item.

    Args:
        items: List of original items
        seed: Random seed

    Returns:
        List of shuffled items
    """
    shuffler = OptionShuffler(seed=seed)
    return [shuffler.shuffle_random(item)[0] for item in items]


def generate_rotated_dataset(
    items: List[MCQItem],
    positions: int = 1
) -> List[MCQItem]:
    """Generate rotated version of all items.

    Args:
        items: List of original items
        positions: Rotation positions

    Returns:
        List of rotated items
    """
    shuffler = OptionShuffler()
    return [shuffler.rotate_options(item, positions)[0] for item in items]
