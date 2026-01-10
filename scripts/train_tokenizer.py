#!/usr/bin/env python3
"""Train a ByteLevel BPE tokenizer from corpus."""
import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing


def main():
    parser = argparse.ArgumentParser(description='Train tokenizer')
    parser.add_argument('--corpus', required=True, help='Corpus text file')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--vocab', type=int, default=8000, help='Vocab size')
    args = parser.parse_args()
    
    print(f"Training tokenizer (vocab={args.vocab})...")
    
    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    
    # Train
    trainer = BpeTrainer(
        vocab_size=args.vocab,
        special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
        show_progress=True
    )
    
    tokenizer.train([args.corpus], trainer)
    
    # Post-processing
    tokenizer.post_processor = TemplateProcessing(
        single="<BOS> $A <EOS>",
        special_tokens=[
            ("<BOS>", tokenizer.token_to_id("<BOS>")),
            ("<EOS>", tokenizer.token_to_id("<EOS>")),
        ]
    )
    
    # Save
    import os
    os.makedirs(args.out, exist_ok=True)
    tokenizer.save(f"{args.out}/tokenizer.json")
    
    print(f"✓ Tokenizer saved: {args.out}/tokenizer.json")
    print(f"  Vocab size: {tokenizer.get_vocab_size()}")


if __name__ == '__main__':
    main()
