#!/usr/bin/env python3
"""Test 4B model to compare with 27B."""

import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_4b():
    model_id = "google/medgemma-4b-it"
    
    logger.info(f"Loading {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    # Check chat template
    logger.info(f"Chat template exists: {tokenizer.chat_template is not None}")
    
    # Test prompt  
    messages = [{"role": "user", "content": "Does aspirin help prevent heart attacks? Answer yes, no, or maybe."}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    logger.info(f"Chat template:\n{chat_text}")
    
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        logger.info(f"Has NaN: {torch.isnan(logits).any().item()}")
        
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
    
    # Also test PubMedQA-style prompt
    logger.info("\n=== Testing PubMedQA prompt ===")
    pubmed_prompt = """Based on your medical knowledge, answer the following research question.

Question: Do mitochondria play a role in cancer?

Respond with one word: yes, no, or maybe"""
    
    messages2 = [{"role": "user", "content": pubmed_prompt}]
    chat_text2 = tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
    inputs2 = tokenizer(chat_text2, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        gen_outputs2 = model.generate(
            **inputs2,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
        new_tokens2 = gen_outputs2[0, inputs2['input_ids'].shape[1]:]
        output_text2 = tokenizer.decode(new_tokens2, skip_special_tokens=True)
        logger.info(f"Generated: '{output_text2}'")

if __name__ == "__main__":
    test_4b()
