#!/usr/bin/env python3
"""Train GPT-style language model from scratch."""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm


class GPTConfig:
    def __init__(self, vocab_size, n_embd=256, n_head=8, n_layer=6, block_size=512):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss


class TokenDataset(Dataset):
    def __init__(self, data_file, block_size):
        self.tokens = np.fromfile(data_file, dtype=np.uint16)
        self.block_size = block_size
    
    def __len__(self):
        return len(self.tokens) - self.block_size - 1
    
    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.block_size + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


def main():
    parser = argparse.ArgumentParser(description='Train language model')
    parser.add_argument('--tokenizer', required=True, help='Tokenizer directory')
    parser.add_argument('--data', required=True, help='Binary dataset file')
    parser.add_argument('--out', required=True, help='Output model directory')
    parser.add_argument('--steps', type=int, default=2000, help='Training steps')
    parser.add_argument('--resume', help='Resume from checkpoint')
    args = parser.parse_args()
    
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(f"{args.tokenizer}/tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()
    
    print("Preparing dataset...")
    dataset = TokenDataset(args.data, block_size=512)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    print("Initializing model...")
    config = GPTConfig(vocab_size=vocab_size)
    model = GPT(config)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    if args.resume and os.path.exists(f"{args.resume}/model.pt"):
        print(f"Resuming from {args.resume}...")
        checkpoint = torch.load(f"{args.resume}/model.pt", map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    print(f"Training for {args.steps} steps...")
    model.train()
    iter_loader = iter(loader)
    
    for step in tqdm(range(args.steps)):
        try:
            x, y = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            x, y = next(iter_loader)
        
        x, y = x.to(device), y.to(device)
        
        logits, loss = model(x, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            tqdm.write(f"Step {step}: loss = {loss.item():.4f}")
    
    print("Saving model...")
    os.makedirs(args.out, exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'config': config.__dict__
    }, f"{args.out}/model.pt")
    
    print(f"✓ Model saved: {args.out}/model.pt")


if __name__ == '__main__':
    main()
