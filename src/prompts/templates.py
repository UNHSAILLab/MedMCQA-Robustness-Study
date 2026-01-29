"""Prompt templates for medical MCQ tasks."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from enum import Enum

from ..data.schemas import MCQItem, PubMedQAItem


class PromptStyle(Enum):
    """Prompt style variants."""
    ZERO_SHOT_DIRECT = "zero_shot_direct"
    ZERO_SHOT_COT = "zero_shot_cot"
    FEW_SHOT_DIRECT = "few_shot_direct"
    FEW_SHOT_COT = "few_shot_cot"
    ANSWER_ONLY = "answer_only"


@dataclass
class PromptConfig:
    """Configuration for prompt generation."""
    style: PromptStyle = PromptStyle.ZERO_SHOT_DIRECT
    num_examples: int = 3
    include_explanation: bool = False


class MedMCQAPromptTemplate:
    """Prompt template for MedMCQA 4-choice questions."""

    SYSTEM_PROMPT = (
        "You are a medical expert answering multiple choice questions. "
        "Analyze each question carefully and select the most accurate answer."
    )

    # Template variants
    ZERO_SHOT_DIRECT = """Question: {question}

Options:
A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}

Answer with a single letter (A, B, C, or D)."""

    ZERO_SHOT_COT = """Question: {question}

Options:
A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}

Think through this step-by-step, analyzing each option. Then provide your final answer as a single letter (A, B, C, or D)."""

    ANSWER_ONLY = """Q: {question}
A) {option_a} B) {option_b} C) {option_c} D) {option_d}
Answer:"""

    FEW_SHOT_HEADER_DIRECT = "Answer the following medical multiple choice questions.\n\n"
    FEW_SHOT_HEADER_COT = "Answer the following medical multiple choice questions. For each, think step-by-step before giving the answer.\n\n"

    def format(
        self,
        item: MCQItem,
        config: PromptConfig,
        few_shot_examples: Optional[List[Dict]] = None
    ) -> str:
        """Format a prompt for the given item and configuration."""
        options = item.options

        if config.style == PromptStyle.ANSWER_ONLY:
            return self.ANSWER_ONLY.format(
                question=item.question,
                option_a=options.get('A', ''),
                option_b=options.get('B', ''),
                option_c=options.get('C', ''),
                option_d=options.get('D', '')
            )

        # Select template
        if config.style in [PromptStyle.ZERO_SHOT_COT, PromptStyle.FEW_SHOT_COT]:
            template = self.ZERO_SHOT_COT
        else:
            template = self.ZERO_SHOT_DIRECT

        # Format main question
        main_prompt = template.format(
            question=item.question,
            option_a=options.get('A', ''),
            option_b=options.get('B', ''),
            option_c=options.get('C', ''),
            option_d=options.get('D', '')
        )

        # Add few-shot examples if needed
        if config.style in [PromptStyle.FEW_SHOT_DIRECT, PromptStyle.FEW_SHOT_COT]:
            if few_shot_examples is None:
                from .few_shot_examples import MEDMCQA_FEW_SHOT_EXAMPLES
                few_shot_examples = MEDMCQA_FEW_SHOT_EXAMPLES

            examples = few_shot_examples[:config.num_examples]
            include_cot = config.style == PromptStyle.FEW_SHOT_COT

            header = self.FEW_SHOT_HEADER_COT if include_cot else self.FEW_SHOT_HEADER_DIRECT
            examples_text = self._format_examples(examples, include_cot)

            return header + examples_text + "\n---\n\n" + main_prompt

        return main_prompt

    def _format_examples(self, examples: List[Dict], include_cot: bool) -> str:
        """Format few-shot examples."""
        formatted = []

        for i, ex in enumerate(examples, 1):
            ex_text = f"Example {i}:\n"
            ex_text += f"Question: {ex['question']}\n"
            ex_text += f"A) {ex['options']['A']}\n"
            ex_text += f"B) {ex['options']['B']}\n"
            ex_text += f"C) {ex['options']['C']}\n"
            ex_text += f"D) {ex['options']['D']}\n"

            if include_cot and 'reasoning' in ex:
                ex_text += f"\nReasoning: {ex['reasoning']}\n"

            ex_text += f"Answer: {ex['correct_answer']}\n"
            formatted.append(ex_text)

        return "\n".join(formatted)


class PubMedQAPromptTemplate:
    """Prompt template for PubMedQA yes/no/maybe questions."""

    QUESTION_ONLY = """Based on your medical knowledge, answer the following research question.

Question: {question}

Respond with one word: yes, no, or maybe"""

    WITH_CONTEXT = """Based on the following research context, answer the question.

Context:
{context}

Question: {question}

Respond with one word: yes, no, or maybe"""

    TRUNCATED_CONTEXT = """Based on the partial research context below, answer the question. Note that this context may be incomplete.

Context (truncated):
{context}

Question: {question}

Respond with one word: yes, no, or maybe"""

    COT_WITH_CONTEXT = """Based on the following research context, answer the question.

Context:
{context}

Question: {question}

Think through this step-by-step based on the evidence provided. Then give your final answer as one word: yes, no, or maybe"""

    def format(
        self,
        item: PubMedQAItem,
        context_mode: str = "full",  # 'none', 'full', 'truncated'
        include_cot: bool = False,
        truncation_ratio: float = 0.5
    ) -> str:
        """Format a prompt for the given item and context mode."""
        if context_mode == "none":
            return self.QUESTION_ONLY.format(question=item.question)

        # Build context string
        context = self._format_context(
            item,
            truncate=(context_mode == "truncated"),
            ratio=truncation_ratio
        )

        if include_cot:
            return self.COT_WITH_CONTEXT.format(
                question=item.question,
                context=context
            )

        template = self.TRUNCATED_CONTEXT if context_mode == "truncated" else self.WITH_CONTEXT

        return template.format(
            question=item.question,
            context=context
        )

    def _format_context(
        self,
        item: PubMedQAItem,
        truncate: bool = False,
        ratio: float = 0.5
    ) -> str:
        """Format context sections with optional truncation."""
        formatted = []
        for label, text in zip(item.section_labels, item.context_sections):
            formatted.append(f"[{label}] {text}")

        full_context = "\n\n".join(formatted)

        if truncate:
            words = full_context.split()
            truncated_words = words[:int(len(words) * ratio)]
            return " ".join(truncated_words) + "..."

        return full_context

    def format_with_sections(
        self,
        item: PubMedQAItem,
        sections: List[str],
        include_cot: bool = False
    ) -> str:
        """Format prompt keeping only specified sections."""
        # Filter to keep only specified sections
        kept_parts = []
        for label, text in zip(item.section_labels, item.context_sections):
            if label.upper() in [s.upper() for s in sections]:
                kept_parts.append(f"[{label}] {text}")

        if not kept_parts:
            return self.format(item, context_mode="none")

        context = "\n\n".join(kept_parts)

        if include_cot:
            return self.COT_WITH_CONTEXT.format(
                question=item.question,
                context=context
            )

        return self.WITH_CONTEXT.format(
            question=item.question,
            context=context
        )
