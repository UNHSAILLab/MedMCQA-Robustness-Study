#!/usr/bin/env python3
"""Debug script v2 for MedGemma 27B empty response issue."""

import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model():
    model_id = "google/medgemma-27b-text-it"
    
    logger.info(f"Loading tokenizer for {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Check special tokens
    logger.info(f"pad_token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    logger.info(f"eos_token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
    logger.info(f"bos_token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})")
    logger.info(f"unk_token: {tokenizer.unk_token} (id: {tokenizer.unk_token_id})")
    
    # Check if there's a chat template
    logger.info(f"Has chat_template: {hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None}")
    
    logger.info("Loading model with 4-bit quantization")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto"
    )
    model.eval()
    
    # Test with chat template format (MedGemma uses Gemma format)
    prompt = "Based on your medical knowledge, does aspirin help prevent heart attacks? Answer yes, no, or maybe."
    
    # Try different approaches
    logger.info("\n=== Approach 1: Using chat template with proper format ===")
    messages = [{"role": "user", "content": prompt}]
    
    # Apply chat template
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    logger.info(f"Chat template text:\n{chat_text}")
    
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    logger.info(f"Full output (with special tokens): {full_text}")
    
    new_tokens = outputs[0, inputs['input_ids'].shape[1]:]
    new_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    logger.info(f"New tokens (with special): '{new_text}'")
    new_text_clean = tokenizer.decode(new_tokens, skip_special_tokens=True)
    logger.info(f"New tokens (clean): '{new_text_clean}'")
    
    # Approach 2: Try without setting pad_token_id to eos
    logger.info("\n=== Approach 2: Not setting pad_token at all ===")
    tokenizer2 = AutoTokenizer.from_pretrained(model_id)
    # Don't modify pad_token
    
    chat_text2 = tokenizer2.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs2 = tokenizer2(chat_text2, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs2 = model.generate(
            **inputs2,
            max_new_tokens=100,
            do_sample=False
        )
    
    new_tokens2 = outputs2[0, inputs2['input_ids'].shape[1]:]
    new_text2 = tokenizer2.decode(new_tokens2, skip_special_tokens=True)
    logger.info(f"Output: '{new_text2}'")
    
    # Approach 3: Try with sampling
    logger.info("\n=== Approach 3: With sampling ===")
    with torch.no_grad():
        outputs3 = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    new_tokens3 = outputs3[0, inputs['input_ids'].shape[1]:]
    new_text3 = tokenizer.decode(new_tokens3, skip_special_tokens=True)
    logger.info(f"Output: '{new_text3}'")

if __name__ == "__main__":
    test_model()
