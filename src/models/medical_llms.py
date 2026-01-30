"""Medical LLM wrappers for BioMistral, Meditron, and other models."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Optional
import logging

from .base import BaseModel

logger = logging.getLogger(__name__)


class MedicalLLM(BaseModel):
    """Generic wrapper for medical LLMs (BioMistral, Meditron, etc.)"""

    MODEL_CONFIGS = {
        'biomistral-7b': {
            'model_id': 'BioMistral/BioMistral-7B',
            'context_length': 4096,
        },
        'meditron-7b': {
            'model_id': 'epfl-llm/meditron-7b',
            'context_length': 4096,
        },
        'meditron-70b': {
            'model_id': 'epfl-llm/meditron-70b',
            'context_length': 4096,
        },
        'medalpaca-7b': {
            'model_id': 'medalpaca/medalpaca-7b',
            'context_length': 2048,
        },
        'pmc-llama-7b': {
            'model_id': 'axiong/PMC_LLaMA_7B',
            'context_length': 2048,
        },
    }

    def __init__(
        self,
        model_name: str,
        quantization: Optional[str] = None,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        use_flash_attention: bool = True
    ):
        """Initialize medical LLM.

        Args:
            model_name: One of 'biomistral-7b', 'meditron-7b', 'meditron-70b', etc.
            quantization: Quantization mode (None, '4bit', '8bit')
            device_map: Device mapping strategy
            torch_dtype: Torch data type
            use_flash_attention: Whether to use flash attention 2
        """
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODEL_CONFIGS.keys())}")

        self.model_name = model_name
        self.config = self.MODEL_CONFIGS[model_name]
        self.model_id = self.config['model_id']
        self.quantization = quantization
        self.device_map = device_map
        self.torch_dtype = getattr(torch, torch_dtype)
        self.use_flash_attention = use_flash_attention

        self.model = None
        self.tokenizer = None

    @property
    def name(self) -> str:
        quant_suffix = f"_{self.quantization}" if self.quantization else ""
        return f"{self.model_name.replace('-', '_')}{quant_suffix}"

    def load(self, device: str = "cuda") -> None:
        """Load model with appropriate quantization."""
        logger.info(f"Loading {self.model_id} with quantization={self.quantization}")

        # Configure quantization
        quant_config = None
        if self.quantization == '4bit':
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.quantization == '8bit':
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prepare model kwargs
        model_kwargs = {
            "device_map": self.device_map,
            "trust_remote_code": True,
        }

        if quant_config:
            model_kwargs["quantization_config"] = quant_config
        else:
            model_kwargs["torch_dtype"] = self.torch_dtype

        # Try flash attention if requested and available
        if self.use_flash_attention:
            try:
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using Flash Attention 2")
            except ImportError:
                logger.info("Flash attention not available, using default attention")

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs
        )
        self.model.eval()

        logger.info(f"Model loaded successfully on {self.model.device}")

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
        num_return_sequences: int = 1,
        **kwargs
    ) -> List[str]:
        """Generate responses for batch of prompts."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        # Tokenize with padding
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_context_length - max_new_tokens
        ).to(self.model.device)

        # Generation config
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if do_sample and temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = kwargs.get("top_p", 0.95)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode only generated portion
        input_length = inputs['input_ids'].shape[1]
        generated = outputs[:, input_length:]

        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        # Handle num_return_sequences > 1
        if num_return_sequences > 1:
            results = []
            for i in range(0, len(decoded), num_return_sequences):
                results.append(decoded[i:i + num_return_sequences])
            return results

        return decoded

    def generate_with_logprobs(
        self,
        prompt: str,
        choices: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """Get log probabilities for specific choice tokens."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_context_length - 10
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]
            log_probs = torch.log_softmax(logits, dim=-1)

        choice_logprobs = {}
        for choice in choices:
            choice_tokens = self.tokenizer.encode(choice, add_special_tokens=False)
            if choice_tokens:
                token_id = choice_tokens[0]
                choice_logprobs[choice] = log_probs[0, token_id].item()
            else:
                choice_logprobs[choice] = float('-inf')

        return choice_logprobs

    @property
    def max_context_length(self) -> int:
        return self.config.get('context_length', 4096)


def load_medical_llm(
    model_name: str,
    quantization: Optional[str] = None,
    device: str = "cuda"
) -> MedicalLLM:
    """Convenience function to load a medical LLM.

    Args:
        model_name: One of 'biomistral-7b', 'meditron-7b', etc.
        quantization: None, '4bit', or '8bit'
        device: Device to load on

    Returns:
        Loaded MedicalLLM instance
    """
    model = MedicalLLM(model_name=model_name, quantization=quantization)
    model.load(device=device)
    return model
