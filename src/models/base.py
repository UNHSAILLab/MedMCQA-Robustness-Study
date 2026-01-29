"""Abstract base class for model implementations."""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any


class BaseModel(ABC):
    """Abstract base class for all model implementations."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier for logging and caching."""
        pass

    @abstractmethod
    def load(self, device: str = "cuda") -> None:
        """Load model and tokenizer to device."""
        pass

    @abstractmethod
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
        num_return_sequences: int = 1,
        **kwargs
    ) -> List[str]:
        """Generate responses for a batch of prompts.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 for greedy)
            do_sample: Whether to sample (True) or use greedy decoding (False)
            num_return_sequences: Number of sequences per prompt

        Returns:
            List of generated text strings
        """
        pass

    @abstractmethod
    def generate_with_logprobs(
        self,
        prompt: str,
        choices: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """Get log probabilities for specific choice tokens.

        Args:
            prompt: Input prompt
            choices: List of choice strings to get probabilities for

        Returns:
            Dict mapping choice strings to log probabilities
        """
        pass

    @property
    @abstractmethod
    def max_context_length(self) -> int:
        """Maximum context length supported by the model."""
        pass

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return hasattr(self, 'model') and self.model is not None

    def unload(self) -> None:
        """Unload model from memory."""
        if hasattr(self, 'model'):
            del self.model
            self.model = None
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
            self.tokenizer = None

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
