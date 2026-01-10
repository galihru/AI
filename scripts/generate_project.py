#!/usr/bin/env python3
"""Generate Arduino project from specification."""
import argparse
import json
import os
import torch
import numpy as np
from tokenizers import Tokenizer
from pathlib import Path


def load_model(model_path, device='cpu'):
    """Load trained GPT model."""
    checkpoint = torch.load(f"{model_path}/model.pt", map_location=device)
    
    # Rebuild model
    from train_lm import GPT, GPTConfig
    config_dict = checkpoint['config']
    config = GPTConfig(**config_dict)
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model = model.to(device)
    
    return model, config


def generate(model, tokenizer, prompt, max_tokens=1024, temperature=0.8, device='cpu'):
    """Generate text from prompt."""
    encoded = tokenizer.encode(prompt)
    idx = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    
    for _ in range(max_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        
        with torch.no_grad():
            logits, _ = model(idx_cond)
        
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        
        # Stop at EOS
        if idx_next.item() == tokenizer.token_to_id("<EOS>"):
            break
    
    return tokenizer.decode(idx[0].tolist())


def retrieve_context(index_file, query, top_k=3):
    """Retrieve relevant code from index."""
    with open(index_file, 'r', encoding='utf-8') as f:
        index_data = json.load(f)
    
    # Simple keyword matching for now
    documents = index_data.get('documents', [])
    if not documents:
        return ""
    
    # Return top documents
    return "\n\n".join(documents[:top_k])


def main():
    parser = argparse.ArgumentParser(description='Generate Arduino project')
    parser.add_argument('--model', required=True, help='Model directory')
    parser.add_argument('--tokenizer', required=True, help='Tokenizer directory')
    parser.add_argument('--index', required=True, help='Index JSON file')
    parser.add_argument('--spec', required=True, help='Specification file')
    parser.add_argument('--out', required=True, help='Output directory')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Loading model...")
    model, config = load_model(args.model, device)
    
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(f"{args.tokenizer}/tokenizer.json")
    
    print("Reading specification...")
    with open(args.spec, 'r', encoding='utf-8') as f:
        spec = f.read()
    
    print("Retrieving context...")
    context = retrieve_context(args.index, spec)
    
    print("Generating project...")
    prompt = f"""// Arduino Project Generator
// Specification: {spec}
// Context examples:
{context[:500]}

// Generated code:
// FILE: main.ino
"""
    
    generated = generate(model, tokenizer, prompt, max_tokens=512, device=device)
    
    # Save output
    os.makedirs(args.out, exist_ok=True)
    output_file = f"{args.out}/generated.ino"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(generated)
    
    print(f"✓ Project generated: {output_file}")
    print("\n--- Generated Code ---")
    print(generated[:500])


if __name__ == '__main__':
    main()
