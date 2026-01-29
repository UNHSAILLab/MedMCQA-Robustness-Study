"""Dataset loaders for MedMCQA and PubMedQA."""

from datasets import load_dataset
from typing import List, Optional
from tqdm import tqdm

from .schemas import MCQItem, PubMedQAItem


class MedMCQALoader:
    """Load and preprocess MedMCQA dataset from HuggingFace.

    Note: The 'test' split does NOT have answers (cop=-1).
    Use 'validation' for evaluation with answers (4,183 items).
    Use 'train' for training/few-shot examples (182,822 items).
    """

    DATASET_ID = "openlifescienceai/medmcqa"
    OPTION_KEYS = ['opa', 'opb', 'opc', 'opd']
    COP_MAPPING = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    def __init__(
        self,
        split: str = "validation",  # Use validation by default (has answers)
        cache_dir: Optional[str] = None,
        limit: Optional[int] = None
    ):
        self.split = split
        self.cache_dir = cache_dir
        self.limit = limit
        self._dataset = None
        self._subjects = None

    def load(self, show_progress: bool = True) -> List[MCQItem]:
        """Load dataset with standardized format."""
        self._dataset = load_dataset(
            self.DATASET_ID,
            split=self.split,
            cache_dir=self.cache_dir
        )

        items = []
        skipped = 0
        iterator = tqdm(self._dataset, desc="Loading MedMCQA") if show_progress else self._dataset

        for idx, item in enumerate(iterator):
            if self.limit and len(items) >= self.limit:
                break
            # Skip items with invalid correct answer (cop = -1 means unknown)
            if item['cop'] not in self.COP_MAPPING:
                skipped += 1
                continue
            items.append(self._standardize(item))

        if skipped > 0 and show_progress:
            print(f"Skipped {skipped} items with invalid correct answer")

        # Cache subjects
        self._subjects = list(set(item.subject for item in items if item.subject))

        return items

    def _standardize(self, item: dict) -> MCQItem:
        """Convert HuggingFace item to standard MCQ format."""
        return MCQItem(
            id=item['id'],
            question=item['question'],
            options={
                'A': item['opa'],
                'B': item['opb'],
                'C': item['opc'],
                'D': item['opd']
            },
            correct_answer=self.COP_MAPPING[item['cop']],
            subject=item.get('subject_name'),
            topic=item.get('topic_name'),
            explanation=item.get('exp'),
            metadata={
                'choice_type': item.get('choice_type', 'single')
            }
        )

    @property
    def subjects(self) -> List[str]:
        """Return list of unique medical subjects."""
        if self._subjects is None:
            raise ValueError("Must call load() before accessing subjects")
        return sorted(self._subjects)

    def get_by_subject(self, items: List[MCQItem], subject: str) -> List[MCQItem]:
        """Filter items by subject."""
        return [item for item in items if item.subject == subject]


class PubMedQALoader:
    """Load and preprocess PubMedQA labeled dataset."""

    DATASET_ID = "qiaojin/PubMedQA"
    SUBSET = "pqa_labeled"

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        limit: Optional[int] = None
    ):
        self.cache_dir = cache_dir
        self.limit = limit
        self._dataset = None

    def load(self, show_progress: bool = True) -> List[PubMedQAItem]:
        """Load pqa_labeled subset."""
        self._dataset = load_dataset(
            self.DATASET_ID,
            self.SUBSET,
            split="train",  # PubMedQA labeled only has train split
            cache_dir=self.cache_dir
        )

        items = []
        iterator = tqdm(self._dataset, desc="Loading PubMedQA") if show_progress else self._dataset

        for idx, item in enumerate(iterator):
            if self.limit and idx >= self.limit:
                break
            items.append(self._standardize(item))

        return items

    def _standardize(self, item: dict) -> PubMedQAItem:
        """Convert HuggingFace item to standard format."""
        context = item['context']

        return PubMedQAItem(
            id=str(item['pubid']),
            question=item['question'],
            context_sections=context['contexts'],
            section_labels=context['labels'],
            mesh_terms=context.get('meshes', []),
            long_answer=item['long_answer'],
            correct_answer=item['final_decision'],
            metadata={
                'reasoning_required_pred': context.get('reasoning_required_pred'),
                'reasoning_free_pred': context.get('reasoning_free_pred')
            }
        )

    def get_by_answer_type(
        self,
        items: List[PubMedQAItem],
        answer_type: str
    ) -> List[PubMedQAItem]:
        """Filter items by answer type (yes/no/maybe)."""
        return [item for item in items if item.correct_answer == answer_type]


def load_medmcqa(
    split: str = "validation",  # Use validation by default (has answers)
    limit: Optional[int] = None,
    cache_dir: Optional[str] = None
) -> List[MCQItem]:
    """Convenience function to load MedMCQA.

    Note: Use 'validation' split for evaluation (4,183 items with answers).
    The 'test' split does NOT have answers.
    """
    loader = MedMCQALoader(split=split, cache_dir=cache_dir, limit=limit)
    return loader.load()


def load_pubmedqa(
    limit: Optional[int] = None,
    cache_dir: Optional[str] = None
) -> List[PubMedQAItem]:
    """Convenience function to load PubMedQA."""
    loader = PubMedQALoader(cache_dir=cache_dir, limit=limit)
    return loader.load()
