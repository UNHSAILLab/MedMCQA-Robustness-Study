#!/usr/bin/env python3
"""Debug script for MedGemma 27B empty response issue."""

import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # Use GPU 5 for testing

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model():
    model_id = "google/medgemma-27b-text-it"
    
    logger.info(f"Loading tokenizer for {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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
    logger.info(f"Model loaded on {model.device}")
    
    # Test prompts
    test_prompts = [
        "Based on your medical knowledge, answer the following research question.\n\nQuestion: Does aspirin help prevent heart attacks?\n\nRespond with one word: yes, no, or maybe",
        "Question: What is the capital of France?\nAnswer:",
        "Hello, how are you?"
    ]
    
    for i, prompt in enumerate(test_prompts):
        logger.info(f"\n=== Test {i+1} ===")
        logger.info(f"Prompt: {prompt[:100]}...")
        
        # Method 1: Direct tokenization
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        logger.info(f"Input shape: {inputs['input_ids'].shape}")
        logger.info(f"Input tokens: {inputs['input_ids'][0][:20].tolist()}...")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        logger.info(f"Output shape: {outputs.shape}")
        logger.info(f"Output tokens: {outputs[0].tolist()[-20:]}")
        
        # Decode full output
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Full output: {full_output}")
        
        # Decode only new tokens
        input_length = inputs['input_ids'].shape[1]
        new_tokens = outputs[0, input_length:]
        logger.info(f"New tokens count: {len(new_tokens)}")
        new_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
        logger.info(f"New output: '{new_output}'")
        
        # Method 2: Try with chat template if available
        if hasattr(tokenizer, 'apply_chat_template'):
            logger.info("\n--- Testing with chat template ---")
            messages = [{"role": "user", "content": prompt}]
            chat_input = tokenizer.apply_chat_template(
                messages, 
                return_tensors="pt",
                add_generation_prompt=True
            ).to(model.device)
            
            logger.info(f"Chat input shape: {chat_input.shape}")
            
            with torch.no_grad():
                chat_outputs = model.generate(
                    chat_input,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            chat_input_length = chat_input.shape[1]
            chat_new_tokens = chat_outputs[0, chat_input_length:]
            chat_new_output = tokenizer.decode(chat_new_tokens, skip_special_tokens=True)
            logger.info(f"Chat template output: '{chat_new_output}'")

if __name__ == "__main__":
    test_model()
