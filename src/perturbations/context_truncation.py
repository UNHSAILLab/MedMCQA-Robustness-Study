"""Context manipulation for PubMedQA evidence conditioning experiments."""

from typing import List, Optional
from ..data.schemas import PubMedQAItem


class ContextTruncator:
    """Context manipulation strategies for PubMedQA."""

    def remove_context(self, item: PubMedQAItem) -> PubMedQAItem:
        """Remove all context, question-only mode.

        Args:
            item: Original PubMedQA item

        Returns:
            Item with empty context
        """
        perturbed = item.model_copy(deep=True)
        perturbed.context_sections = []
        perturbed.section_labels = []
        perturbed.perturbation = {'type': 'no_context'}
        return perturbed

    def truncate_by_ratio(
        self,
        item: PubMedQAItem,
        ratio: float = 0.5
    ) -> PubMedQAItem:
        """Keep first N% of context by word count.

        Args:
            item: Original item
            ratio: Proportion of context to keep (0.0 to 1.0)

        Returns:
            Item with truncated context
        """
        if ratio <= 0:
            return self.remove_context(item)
        if ratio >= 1:
            perturbed = item.model_copy(deep=True)
            perturbed.perturbation = {'type': 'full_context', 'ratio': 1.0}
            return perturbed

        # Concatenate all context
        full_text = " ".join(item.context_sections)
        words = full_text.split()
        original_length = len(words)

        # Truncate
        truncated_length = int(len(words) * ratio)
        truncated_words = words[:truncated_length]
        truncated_text = " ".join(truncated_words)

        perturbed = item.model_copy(deep=True)
        perturbed.context_sections = [truncated_text]
        perturbed.section_labels = ['TRUNCATED']
        perturbed.perturbation = {
            'type': 'truncated_context',
            'ratio': ratio,
            'original_word_count': original_length,
            'truncated_word_count': truncated_length
        }

        return perturbed

    def keep_sections(
        self,
        item: PubMedQAItem,
        sections: List[str]
    ) -> PubMedQAItem:
        """Keep only specified sections.

        Args:
            item: Original item
            sections: List of section labels to keep (e.g., ['BACKGROUND', 'METHODS'])

        Returns:
            Item with filtered sections
        """
        sections_upper = [s.upper() for s in sections]

        kept_sections = []
        kept_labels = []

        for section, label in zip(item.context_sections, item.section_labels):
            if label.upper() in sections_upper:
                kept_sections.append(section)
                kept_labels.append(label)

        perturbed = item.model_copy(deep=True)
        perturbed.context_sections = kept_sections
        perturbed.section_labels = kept_labels
        perturbed.perturbation = {
            'type': 'section_filter',
            'kept_sections': sections,
            'original_sections': item.section_labels,
            'retained_count': len(kept_sections)
        }

        return perturbed

    def keep_background_only(self, item: PubMedQAItem) -> PubMedQAItem:
        """Keep only BACKGROUND and OBJECTIVE sections."""
        return self.keep_sections(item, ['BACKGROUND', 'OBJECTIVE', 'OBJECTIVES'])

    def keep_results_only(self, item: PubMedQAItem) -> PubMedQAItem:
        """Keep only RESULTS and CONCLUSIONS sections."""
        return self.keep_sections(item, ['RESULTS', 'CONCLUSIONS', 'FINDINGS'])

    def keep_methods_only(self, item: PubMedQAItem) -> PubMedQAItem:
        """Keep only METHODS section."""
        return self.keep_sections(item, ['METHODS', 'MATERIALS AND METHODS'])

    def remove_results(self, item: PubMedQAItem) -> PubMedQAItem:
        """Remove RESULTS and CONCLUSIONS, keeping only setup info."""
        exclude = ['RESULTS', 'CONCLUSIONS', 'FINDINGS']
        exclude_upper = [s.upper() for s in exclude]

        kept_sections = []
        kept_labels = []

        for section, label in zip(item.context_sections, item.section_labels):
            if label.upper() not in exclude_upper:
                kept_sections.append(section)
                kept_labels.append(label)

        perturbed = item.model_copy(deep=True)
        perturbed.context_sections = kept_sections
        perturbed.section_labels = kept_labels
        perturbed.perturbation = {
            'type': 'remove_results',
            'excluded_sections': exclude,
            'original_sections': item.section_labels
        }

        return perturbed

    def truncate_each_section(
        self,
        item: PubMedQAItem,
        max_words_per_section: int = 50
    ) -> PubMedQAItem:
        """Truncate each section to a maximum word count.

        Args:
            item: Original item
            max_words_per_section: Maximum words to keep per section

        Returns:
            Item with truncated sections
        """
        truncated_sections = []
        truncated_labels = []

        for section, label in zip(item.context_sections, item.section_labels):
            words = section.split()
            if len(words) > max_words_per_section:
                truncated = " ".join(words[:max_words_per_section]) + "..."
            else:
                truncated = section
            truncated_sections.append(truncated)
            truncated_labels.append(label)

        perturbed = item.model_copy(deep=True)
        perturbed.context_sections = truncated_sections
        perturbed.section_labels = truncated_labels
        perturbed.perturbation = {
            'type': 'truncate_each_section',
            'max_words': max_words_per_section
        }

        return perturbed


def generate_context_variants(
    item: PubMedQAItem,
    truncation_ratios: List[float] = None
) -> List[PubMedQAItem]:
    """Generate multiple context variants for an item.

    Args:
        item: Original item
        truncation_ratios: List of truncation ratios to generate

    Returns:
        List of context variants
    """
    if truncation_ratios is None:
        truncation_ratios = [0.25, 0.5, 0.75]

    truncator = ContextTruncator()

    variants = [
        truncator.remove_context(item),  # No context
    ]

    for ratio in truncation_ratios:
        variants.append(truncator.truncate_by_ratio(item, ratio))

    # Full context (original)
    full = item.model_copy(deep=True)
    full.perturbation = {'type': 'full_context', 'ratio': 1.0}
    variants.append(full)

    return variants
