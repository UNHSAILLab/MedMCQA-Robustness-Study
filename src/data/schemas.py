"""Data schemas for medical MCQ datasets."""

from pydantic import BaseModel
from typing import Dict, List, Optional, Any


class MCQItem(BaseModel):
    """Standardized format for multiple choice questions."""
    id: str
    question: str
    options: Dict[str, str]  # {'A': 'option text', 'B': '...', ...}
    correct_answer: str  # 'A', 'B', 'C', or 'D'
    subject: Optional[str] = None
    topic: Optional[str] = None
    explanation: Optional[str] = None
    metadata: Dict[str, Any] = {}

    # Perturbation tracking
    perturbation: Optional[Dict[str, Any]] = None
    original_correct: Optional[str] = None  # Original answer before perturbation


class PubMedQAItem(BaseModel):
    """Standardized format for PubMedQA questions."""
    id: str
    question: str
    context_sections: List[str]
    section_labels: List[str]
    mesh_terms: List[str] = []
    long_answer: str
    correct_answer: str  # 'yes', 'no', or 'maybe'
    metadata: Dict[str, Any] = {}

    # Perturbation tracking
    perturbation: Optional[Dict[str, Any]] = None

    @property
    def full_context(self) -> str:
        """Concatenate all context sections."""
        parts = []
        for label, text in zip(self.section_labels, self.context_sections):
            parts.append(f"[{label}] {text}")
        return "\n\n".join(parts)


class ModelResponse(BaseModel):
    """Response from model inference."""
    item_id: str
    prompt: str
    raw_output: str
    parsed_answer: Optional[str] = None
    confidence: Optional[float] = None
    generation_time_ms: float = 0.0
    metadata: Dict[str, Any] = {}


class ExperimentResult(BaseModel):
    """Result container for experiment runs."""
    experiment_name: str
    model_name: str
    condition: str
    responses: List[ModelResponse]
    metrics: Dict[str, Any] = {}
    config: Dict[str, Any] = {}
