#!/usr/bin/env python3
"""Debug script v4 - try different quantization settings."""

import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_quantization(compute_dtype, quant_type="nf4"):
    model_id = "google/medgemma-27b-text-it"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    logger.info(f"\n=== Testing with compute_dtype={compute_dtype}, quant_type={quant_type} ===")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type=quant_type
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto"
    )
    model.eval()
    
    messages = [{"role": "user", "content": "Does aspirin help prevent heart attacks? Answer yes, no, or maybe."}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        has_nan = torch.isnan(logits).any().item()
        logger.info(f"Has NaN: {has_nan}")
        
        if not has_nan:
            top_k = torch.topk(logits, k=5)
            for token_id, logit in zip(top_k.indices.tolist(), top_k.values.tolist()):
                token_text = tokenizer.decode([token_id])
                logger.info(f"  Token '{token_text}' (id={token_id}): {logit:.4f}")
            
            # Try generation
            gen_outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            new_tokens = gen_outputs[0, inputs['input_ids'].shape[1]:]
            output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            logger.info(f"Generated: '{output_text}'")
            return True
    
    del model
    torch.cuda.empty_cache()
    return False

def test_8bit():
    model_id = "google/medgemma-27b-text-it"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    logger.info(f"\n=== Testing with 8-bit quantization ===")
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto"
    )
    model.eval()
    
    messages = [{"role": "user", "content": "Does aspirin help prevent heart attacks? Answer yes, no, or maybe."}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        has_nan = torch.isnan(logits).any().item()
        logger.info(f"Has NaN: {has_nan}")
        
        if not has_nan:
            top_k = torch.topk(logits, k=5)
            for token_id, logit in zip(top_k.indices.tolist(), top_k.values.tolist()):
                token_text = tokenizer.decode([token_id])
                logger.info(f"  Token '{token_text}' (id={token_id}): {logit:.4f}")
            
            gen_outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            new_tokens = gen_outputs[0, inputs['input_ids'].shape[1]:]
            output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            logger.info(f"Generated: '{output_text}'")
            return True
    
    del model
    torch.cuda.empty_cache()
    return False

if __name__ == "__main__":
    # Try different 4-bit configs
    configs = [
        (torch.float32, "nf4"),
        (torch.float16, "nf4"),
        (torch.float16, "fp4"),
    ]
    
    for compute_dtype, quant_type in configs:
        try:
            if test_quantization(compute_dtype, quant_type):
                logger.info(f"SUCCESS with {compute_dtype}, {quant_type}")
                break
        except Exception as e:
            logger.error(f"Failed with {compute_dtype}, {quant_type}: {e}")
        torch.cuda.empty_cache()
    
    # Try 8-bit
    try:
        if test_8bit():
            logger.info("SUCCESS with 8-bit")
    except Exception as e:
        logger.error(f"Failed with 8-bit: {e}")
