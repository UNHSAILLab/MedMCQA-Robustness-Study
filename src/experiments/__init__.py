"""Experiment modules."""

from .exp1_prompt_ablation import PromptAblationExperiment
from .exp2_option_order import OptionOrderExperiment
from .exp3_evidence_conditioning import EvidenceConditioningExperiment
from .exp4_self_consistency import SelfConsistencyExperiment
from .exp5_robust_baselines import (
    CoTSelfConsistencyExperiment,
    PermutationVoteExperiment,
    ClozeScoreExperiment,
)

__all__ = [
    'PromptAblationExperiment',
    'OptionOrderExperiment',
    'EvidenceConditioningExperiment',
    'SelfConsistencyExperiment',
    'CoTSelfConsistencyExperiment',
    'PermutationVoteExperiment',
    'ClozeScoreExperiment',
]
