#!/usr/bin/env python3
"""Convert corpus to binary token dataset."""
import argparse
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset')
    parser.add_argument('--corpus', required=True, help='Corpus text file')
    parser.add_argument('--tokenizer', required=True, help='Tokenizer directory')
    parser.add_argument('--out', required=True, help='Output binary file')
    args = parser.parse_args()
    
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(f"{args.tokenizer}/tokenizer.json")
    
    print("Processing corpus in chunks...")
    chunk_size = 10_000_000  # 10MB chunks
    all_tokens = []
    
    with open(args.corpus, 'r', encoding='utf-8', errors='ignore') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            print(f"Tokenizing chunk ({len(chunk):,} chars)...")
            encoded = tokenizer.encode(chunk)
            all_tokens.extend(encoded.ids)
    
    print(f"Converting to array...")
    tokens = np.array(all_tokens, dtype=np.uint16)
    
    print(f"Saving {len(tokens):,} tokens...")
    tokens.tofile(args.out)
    
    print(f"✓ Dataset saved: {args.out}")
    print(f"  Token count: {len(tokens):,}")
    print(f"  File size: {len(tokens) * 2 / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    main()
