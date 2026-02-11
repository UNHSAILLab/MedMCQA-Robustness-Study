"""Context manipulation for PubMedQA evidence conditioning experiments."""

import math
from typing import List, Optional
from ..data.schemas import PubMedQAItem

# Common English stop words for salience scoring
_STOP_WORDS = frozenset({
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'shall', 'can', 'this', 'that',
    'these', 'those', 'it', 'its', 'not', 'no', 'nor', 'so', 'if', 'as',
    'than', 'too', 'very', 'just', 'about', 'also', 'more', 'other', 'some',
    'such', 'only', 'then', 'into', 'over', 'after', 'before', 'between',
    'under', 'above', 'up', 'out', 'all', 'each', 'both', 'which', 'who',
    'whom', 'what', 'when', 'where', 'how', 'there', 'here', 'we', 'they',
    'he', 'she', 'i', 'you', 'me', 'him', 'her', 'us', 'them', 'my', 'our',
    'your', 'his', 'their',
})


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

    def truncate_front(
        self,
        item: PubMedQAItem,
        ratio: float = 0.5
    ) -> PubMedQAItem:
        """Keep the first N% of context words. Alias for truncate_by_ratio.

        Args:
            item: Original item
            ratio: Proportion of context to keep (0.0 to 1.0)

        Returns:
            Item with truncated context (front kept)
        """
        return self.truncate_by_ratio(item, ratio)

    def truncate_back(
        self,
        item: PubMedQAItem,
        ratio: float = 0.5
    ) -> PubMedQAItem:
        """Keep the last N% of context words.

        Args:
            item: Original item
            ratio: Proportion of context to keep (0.0 to 1.0)

        Returns:
            Item with truncated context (back kept)
        """
        if ratio <= 0:
            return self.remove_context(item)
        if ratio >= 1:
            perturbed = item.model_copy(deep=True)
            perturbed.perturbation = {'type': 'full_context', 'ratio': 1.0}
            return perturbed

        full_text = " ".join(item.context_sections)
        words = full_text.split()
        original_length = len(words)

        keep_count = int(len(words) * ratio)
        kept_words = words[-keep_count:] if keep_count > 0 else []
        truncated_text = " ".join(kept_words)

        perturbed = item.model_copy(deep=True)
        perturbed.context_sections = [truncated_text]
        perturbed.section_labels = ['TRUNCATED_BACK']
        perturbed.perturbation = {
            'type': 'truncated_back',
            'ratio': ratio,
            'position': 'back',
            'original_word_count': original_length,
            'truncated_word_count': len(kept_words)
        }

        return perturbed

    def truncate_middle(
        self,
        item: PubMedQAItem,
        ratio: float = 0.5
    ) -> PubMedQAItem:
        """Keep the middle N% of context words, removing from both ends equally.

        Args:
            item: Original item
            ratio: Proportion of context to keep (0.0 to 1.0)

        Returns:
            Item with truncated context (middle kept)
        """
        if ratio <= 0:
            return self.remove_context(item)
        if ratio >= 1:
            perturbed = item.model_copy(deep=True)
            perturbed.perturbation = {'type': 'full_context', 'ratio': 1.0}
            return perturbed

        full_text = " ".join(item.context_sections)
        words = full_text.split()
        original_length = len(words)

        keep_count = int(len(words) * ratio)
        remove_total = original_length - keep_count
        remove_front = remove_total // 2
        start_idx = remove_front
        end_idx = start_idx + keep_count

        kept_words = words[start_idx:end_idx]
        truncated_text = " ".join(kept_words)

        perturbed = item.model_copy(deep=True)
        perturbed.context_sections = [truncated_text]
        perturbed.section_labels = ['TRUNCATED_MIDDLE']
        perturbed.perturbation = {
            'type': 'truncated_middle',
            'ratio': ratio,
            'position': 'middle',
            'original_word_count': original_length,
            'truncated_word_count': len(kept_words)
        }

        return perturbed

    def truncate_by_tokens(
        self,
        item: PubMedQAItem,
        max_tokens: int,
        position: str = 'front'
    ) -> PubMedQAItem:
        """Truncate at token level using approximate 4 chars per token estimate.

        Uses a simple whitespace tokenizer where each token is approximately
        4 characters.

        Args:
            item: Original item
            max_tokens: Maximum number of tokens to keep
            position: Where to keep tokens from ('front', 'back', 'middle')

        Returns:
            Item with token-level truncated context
        """
        full_text = " ".join(item.context_sections)
        # Approximate tokens: split on whitespace, estimate ~4 chars per token
        words = full_text.split()
        # Estimate token count per word: ceil(len(word) / 4)
        token_counts = [max(1, math.ceil(len(w) / 4)) for w in words]
        cumulative = 0
        max_word_idx = len(words)

        if position == 'front':
            for i, tc in enumerate(token_counts):
                cumulative += tc
                if cumulative > max_tokens:
                    max_word_idx = i
                    break
            kept_words = words[:max_word_idx]
        elif position == 'back':
            # Walk backwards
            for i in range(len(token_counts) - 1, -1, -1):
                cumulative += token_counts[i]
                if cumulative > max_tokens:
                    max_word_idx = i + 1
                    break
            else:
                max_word_idx = 0
            kept_words = words[max_word_idx:]
        elif position == 'middle':
            total_tokens = sum(token_counts)
            if total_tokens <= max_tokens:
                kept_words = words
            else:
                excess = total_tokens - max_tokens
                front_remove_tokens = excess // 2
                cumulative_front = 0
                start_idx = 0
                for i, tc in enumerate(token_counts):
                    cumulative_front += tc
                    if cumulative_front >= front_remove_tokens:
                        start_idx = i + 1
                        break
                # Find end from back
                back_remove_tokens = excess - front_remove_tokens
                cumulative_back = 0
                end_idx = len(words)
                for i in range(len(token_counts) - 1, -1, -1):
                    cumulative_back += token_counts[i]
                    if cumulative_back >= back_remove_tokens:
                        end_idx = i
                        break
                kept_words = words[start_idx:end_idx]
        else:
            kept_words = words[:max_word_idx]

        truncated_text = " ".join(kept_words)

        perturbed = item.model_copy(deep=True)
        perturbed.context_sections = [truncated_text]
        perturbed.section_labels = ['TOKEN_TRUNCATED']
        perturbed.perturbation = {
            'type': 'token_truncated',
            'max_tokens': max_tokens,
            'position': position,
            'original_word_count': len(words),
            'truncated_word_count': len(kept_words)
        }

        return perturbed

    def truncate_by_sentences(
        self,
        item: PubMedQAItem,
        ratio: float = 0.5,
        position: str = 'front'
    ) -> PubMedQAItem:
        """Truncate at sentence boundaries, keeping N% of sentences.

        Splits on '. ' to identify sentence boundaries.

        Args:
            item: Original item
            ratio: Proportion of sentences to keep (0.0 to 1.0)
            position: Which sentences to keep ('front', 'back', 'middle')

        Returns:
            Item with sentence-level truncated context
        """
        if ratio <= 0:
            return self.remove_context(item)

        full_text = " ".join(item.context_sections)
        # Split on sentence boundary '. '
        sentences = full_text.split('. ')
        original_count = len(sentences)

        if ratio >= 1:
            perturbed = item.model_copy(deep=True)
            perturbed.perturbation = {'type': 'full_context', 'ratio': 1.0}
            return perturbed

        keep_count = max(1, int(len(sentences) * ratio))

        if position == 'front':
            kept = sentences[:keep_count]
        elif position == 'back':
            kept = sentences[-keep_count:]
        elif position == 'middle':
            remove_total = original_count - keep_count
            start_idx = remove_total // 2
            kept = sentences[start_idx:start_idx + keep_count]
        else:
            kept = sentences[:keep_count]

        truncated_text = '. '.join(kept)
        # Restore trailing period if the original text ends with one
        if not truncated_text.endswith('.'):
            truncated_text += '.'

        perturbed = item.model_copy(deep=True)
        perturbed.context_sections = [truncated_text]
        perturbed.section_labels = ['SENTENCE_TRUNCATED']
        perturbed.perturbation = {
            'type': 'sentence_truncated',
            'ratio': ratio,
            'position': position,
            'original_sentence_count': original_count,
            'kept_sentence_count': len(kept)
        }

        return perturbed

    def extract_salient_sentences(
        self,
        item: PubMedQAItem,
        top_k: int = 5
    ) -> PubMedQAItem:
        """Extract top-K sentences with highest keyword overlap with the question.

        Scores each context sentence by the count of overlapping non-stopword
        tokens with the question.

        Args:
            item: Original item
            top_k: Number of top sentences to keep

        Returns:
            Item with only the most salient sentences
        """
        full_text = " ".join(item.context_sections)
        sentences = full_text.split('. ')

        # Tokenize question into non-stopword tokens
        question_tokens = {
            w.lower().strip('.,;:?!()[]')
            for w in item.question.split()
            if w.lower().strip('.,;:?!()[]') not in _STOP_WORDS
            and len(w.strip('.,;:?!()[]')) > 1
        }

        # Score each sentence
        scored = []
        for i, sent in enumerate(sentences):
            sent_tokens = {
                w.lower().strip('.,;:?!()')
                for w in sent.split()
                if w.lower().strip('.,;:?!()') not in _STOP_WORDS
                and len(w.strip('.,;:?!()')) > 1
            }
            overlap = len(question_tokens & sent_tokens)
            scored.append((overlap, i, sent))

        # Sort by score descending, then by original order for ties
        scored.sort(key=lambda x: (-x[0], x[1]))

        # Take top_k, then re-sort by original order to preserve flow
        top = scored[:top_k]
        top.sort(key=lambda x: x[1])

        kept_sentences = [s[2] for s in top]
        truncated_text = '. '.join(kept_sentences)
        if not truncated_text.endswith('.'):
            truncated_text += '.'

        perturbed = item.model_copy(deep=True)
        perturbed.context_sections = [truncated_text]
        perturbed.section_labels = ['SALIENT']
        perturbed.perturbation = {
            'type': 'salient_sentences',
            'top_k': top_k,
            'original_sentence_count': len(sentences),
            'kept_sentence_count': len(kept_sentences),
            'scores': [s[0] for s in top]
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
