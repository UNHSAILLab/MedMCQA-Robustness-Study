#!/usr/bin/env python3
"""Test 27B without quantization on multi-GPU."""

import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"  # Use 3 GPUs

from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_27b_full():
    model_id = "google/medgemma-27b-text-it"
    
    logger.info(f"Loading {model_id} at full precision on multi-GPU")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    
    logger.info(f"Model device map: {model.hf_device_map}")
    
    messages = [{"role": "user", "content": "Does aspirin help prevent heart attacks? Answer yes, no, or maybe."}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    logger.info(f"Chat template:\n{chat_text}")
    
    inputs = tokenizer(chat_text, return_tensors="pt").to("cuda:0")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        logger.info(f"Logits dtype: {logits.dtype}")
        logger.info(f"Has NaN: {torch.isnan(logits).any().item()}")
        
        if not torch.isnan(logits).any():
            top_k = torch.topk(logits.float(), k=5)
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
    return False

if __name__ == "__main__":
    test_27b_full()
