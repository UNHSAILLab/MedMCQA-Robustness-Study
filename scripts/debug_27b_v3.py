#!/usr/bin/env python3
"""Debug script v3 - check logits and attention mask."""

import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model():
    model_id = "google/medgemma-27b-text-it"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
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
    
    # Test prompt
    messages = [{"role": "user", "content": "Does aspirin help prevent heart attacks? Answer yes, no, or maybe."}]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
    
    logger.info(f"Input IDs: {inputs['input_ids'].shape}")
    logger.info(f"Attention mask: {inputs.get('attention_mask', 'None')}")
    
    # Check logits directly
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        logger.info(f"Logits shape: {logits.shape}")
        logger.info(f"Logits dtype: {logits.dtype}")
        
        # Check last token logits
        last_logits = logits[0, -1, :]
        logger.info(f"Last logits - min: {last_logits.min()}, max: {last_logits.max()}, mean: {last_logits.mean()}")
        logger.info(f"Has NaN: {torch.isnan(last_logits).any()}, Has Inf: {torch.isinf(last_logits).any()}")
        
        # Get top predictions
        top_k = torch.topk(last_logits, k=10)
        logger.info(f"Top 10 tokens: {top_k.indices.tolist()}")
        logger.info(f"Top 10 logits: {top_k.values.tolist()}")
        
        # Decode top tokens
        for idx, (token_id, logit) in enumerate(zip(top_k.indices.tolist(), top_k.values.tolist())):
            token_text = tokenizer.decode([token_id])
            logger.info(f"  {idx+1}. Token {token_id} ('{token_text}'): {logit:.4f}")
        
        # Try manual greedy decoding
        logger.info("\n=== Manual greedy decoding ===")
        current_ids = inputs['input_ids'].clone()
        generated_tokens = []
        
        for step in range(20):
            with torch.no_grad():
                out = model(current_ids)
                next_logits = out.logits[0, -1, :]
                
                if torch.isnan(next_logits).any() or torch.isinf(next_logits).any():
                    logger.info(f"Step {step}: NaN/Inf detected!")
                    break
                
                next_token = torch.argmax(next_logits).unsqueeze(0).unsqueeze(0)
                generated_tokens.append(next_token.item())
                
                if next_token.item() == tokenizer.eos_token_id:
                    logger.info(f"Step {step}: EOS token generated")
                    break
                
                if next_token.item() == tokenizer.pad_token_id:
                    logger.info(f"Step {step}: PAD token generated (id={next_token.item()})")
                
                current_ids = torch.cat([current_ids, next_token], dim=1)
        
        logger.info(f"Generated tokens: {generated_tokens}")
        logger.info(f"Generated text: '{tokenizer.decode(generated_tokens, skip_special_tokens=True)}'")

if __name__ == "__main__":
    test_model()
