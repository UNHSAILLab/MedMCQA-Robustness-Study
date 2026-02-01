#!/usr/bin/env python3
"""Test 27B with single free GPU and simple config."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_27b():
    model_id = "google/medgemma-27b-text-it"
    
    logger.info(f"Loading {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Try loading on GPU 3 and let it overflow to CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={3: "75GB", "cpu": "100GB"}
    )
    model.eval()
    
    logger.info(f"Model loaded, checking device map...")
    unique_devices = set(model.hf_device_map.values())
    logger.info(f"Unique devices: {unique_devices}")
    
    messages = [{"role": "user", "content": "Does aspirin help prevent heart attacks? Answer yes, no, or maybe."}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(chat_text, return_tensors="pt").to("cuda:3")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :].float()
        logger.info(f"Has NaN: {torch.isnan(logits).any().item()}")
        
        if not torch.isnan(logits).any():
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
        else:
            logger.error("NaN in logits!")

if __name__ == "__main__":
    test_27b()
