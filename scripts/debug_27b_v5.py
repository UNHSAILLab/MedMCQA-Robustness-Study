#!/usr/bin/env python3
"""Test 27B with different loading strategies."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_27b():
    model_id = "google/medgemma-27b-text-it"
    
    logger.info(f"Loading {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Try loading with bfloat16 but using specific memory allocation
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={3: "70GB", 5: "70GB", 6: "70GB", 7: "70GB"}  # Use free GPUs
    )
    model.eval()
    
    logger.info(f"Model device map keys: {list(model.hf_device_map.keys())[:10]}...")
    
    messages = [{"role": "user", "content": "Does aspirin help prevent heart attacks? Answer yes, no, or maybe."}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(chat_text, return_tensors="pt")
    # Move to first device in the model
    first_device = next(iter(model.hf_device_map.values()))
    if isinstance(first_device, int):
        inputs = inputs.to(f"cuda:{first_device}")
    else:
        inputs = inputs.to(first_device)
    
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
        else:
            logger.error("NaN in logits - model output corrupted")

if __name__ == "__main__":
    test_27b()
