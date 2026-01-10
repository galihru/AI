#!/usr/bin/env python3
"""
Autonomous AI Training System - ArduScratch
Trains continuously until reaching human-level code generation.

Features:
- Continuous training with auto-checkpoint
- Loss tracking and visualization
- Auto-validation every N steps
- Resume on crash
- Stop when loss target achieved
- Multi-million step capable
"""

import argparse
import os
import json
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm
import subprocess


# Import model architecture
import sys
sys.path.insert(0, os.path.dirname(__file__))
from train_lm import GPT, GPTConfig, TokenDataset


class TrainingLogger:
    """Logs training metrics and progress."""
    
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, 'training_log.json')
        self.history = self.load_history()
    
    def load_history(self):
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return {
            'steps': [],
            'losses': [],
            'timestamps': [],
            'checkpoints': [],
            'best_loss': float('inf'),
            'total_time': 0
        }
    
    def log(self, step, loss):
        self.history['steps'].append(step)
        self.history['losses'].append(float(loss))
        self.history['timestamps'].append(datetime.now().isoformat())
        
        if loss < self.history['best_loss']:
            self.history['best_loss'] = float(loss)
        
        # Save every 10 steps
        if len(self.history['steps']) % 10 == 0:
            self.save()
    
    def save(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def get_stats(self):
        if not self.history['losses']:
            return {}
        
        recent_losses = self.history['losses'][-100:] if len(self.history['losses']) > 100 else self.history['losses']
        
        return {
            'current_step': self.history['steps'][-1] if self.history['steps'] else 0,
            'current_loss': self.history['losses'][-1] if self.history['losses'] else float('inf'),
            'best_loss': self.history['best_loss'],
            'avg_recent_loss': np.mean(recent_losses),
            'total_steps': len(self.history['steps']),
            'improvement': self.history['losses'][0] - self.history['losses'][-1] if len(self.history['losses']) > 1 else 0
        }


class AutoTrainer:
    """Autonomous training system."""
    
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger = TrainingLogger(config['log_dir'])
        
        print(f"🚀 Auto Training System Initialized")
        print(f"   Device: {self.device}")
        print(f"   Target Loss: {config['target_loss']}")
        print(f"   Max Steps: {config['max_steps']}")
        
    def setup(self):
        """Setup model, optimizer, dataset."""
        print("\n📚 Loading tokenizer...")
        self.tokenizer = Tokenizer.from_file(f"{self.config['tokenizer_dir']}/tokenizer.json")
        vocab_size = self.tokenizer.get_vocab_size()
        
        print("📊 Loading dataset...")
        self.dataset = TokenDataset(self.config['data_file'], block_size=512)
        self.loader = DataLoader(
            self.dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True, 
            num_workers=0
        )
        
        print("🧠 Initializing model...")
        config = GPTConfig(vocab_size=vocab_size, n_embd=256, n_head=8, n_layer=6, block_size=512)
        self.model = GPT(config).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config['learning_rate'],
            weight_decay=0.01
        )
        
        # Load checkpoint if exists
        checkpoint_path = os.path.join(self.config['model_dir'], 'model.pt')
        if os.path.exists(checkpoint_path):
            print("🔄 Resuming from checkpoint...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.current_step = checkpoint.get('step', 0)
            print(f"   Resumed at step {self.current_step}")
        else:
            self.current_step = 0
            print("   Starting fresh training")
    
    def save_checkpoint(self, step, loss):
        """Save model checkpoint."""
        os.makedirs(self.config['model_dir'], exist_ok=True)
        
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': step,
            'loss': loss,
            'config': self.model.config.__dict__
        }
        
        # Save current
        torch.save(checkpoint, os.path.join(self.config['model_dir'], 'model.pt'))
        
        # Save best
        if loss < self.logger.history['best_loss']:
            torch.save(checkpoint, os.path.join(self.config['model_dir'], 'best_model.pt'))
            print(f"   💎 New best model! Loss: {loss:.4f}")
        
        # Save periodic backup
        if step % 5000 == 0:
            backup_path = os.path.join(self.config['model_dir'], f'checkpoint_{step}.pt')
            torch.save(checkpoint, backup_path)
            print(f"   💾 Backup saved: checkpoint_{step}.pt")
        
        self.logger.history['checkpoints'].append({
            'step': step,
            'loss': float(loss),
            'timestamp': datetime.now().isoformat()
        })
        self.logger.save()
    
    def validate(self):
        """Validate model by generating sample code."""
        print("\n🔍 Validating model...")
        
        self.model.eval()
        
        test_prompts = [
            "// LED blink on pin 13\nvoid setup() {",
            "// Read temperature sensor\n#include <DHT.h>",
            "void loop() {\n  digitalWrite("
        ]
        
        with torch.no_grad():
            for prompt in test_prompts[:1]:  # Test one prompt
                encoded = self.tokenizer.encode(prompt)
                idx = torch.tensor([encoded.ids[:50]], dtype=torch.long, device=self.device)
                
                for _ in range(50):
                    idx_cond = idx if idx.size(1) <= 512 else idx[:, -512:]
                    logits, _ = self.model(idx_cond)
                    logits = logits[:, -1, :] / 0.8
                    probs = torch.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                    idx = torch.cat((idx, idx_next), dim=1)
                
                generated = self.tokenizer.decode(idx[0].tolist())
                print(f"\n   Prompt: {prompt[:30]}...")
                print(f"   Output: {generated[:100]}...")
        
        self.model.train()
    
    def train_epoch(self, steps_per_checkpoint):
        """Train for one checkpoint period."""
        self.model.train()
        iter_loader = iter(self.loader)
        
        losses = []
        start_time = time.time()
        
        for i in range(steps_per_checkpoint):
            try:
                x, y = next(iter_loader)
            except StopIteration:
                iter_loader = iter(self.loader)
                x, y = next(iter_loader)
            
            x, y = x.to(self.device), y.to(self.device)
            
            logits, loss = self.model(x, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            losses.append(loss.item())
            self.current_step += 1
            
            # Log every 10 steps
            if i % 10 == 0:
                self.logger.log(self.current_step, loss.item())
            
            # Print progress every 100 steps
            if i % 100 == 0:
                avg_loss = np.mean(losses[-100:])
                elapsed = time.time() - start_time
                steps_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
                
                print(f"   Step {self.current_step:,} | Loss: {avg_loss:.4f} | {steps_per_sec:.2f} steps/s")
        
        return np.mean(losses)
    
    def run(self):
        """Main training loop - runs until target achieved."""
        self.setup()
        
        print("\n" + "="*60)
        print("🎯 AUTONOMOUS TRAINING STARTED")
        print("="*60)
        print(f"Target: Loss < {self.config['target_loss']}")
        print(f"Max Steps: {self.config['max_steps']:,}")
        print(f"Checkpoint Every: {self.config['checkpoint_interval']} steps")
        print(f"Validation Every: {self.config['validation_interval']} steps")
        print("="*60 + "\n")
        
        try:
            while self.current_step < self.config['max_steps']:
                epoch_start = time.time()
                
                # Train for checkpoint interval
                avg_loss = self.train_epoch(self.config['checkpoint_interval'])
                
                # Save checkpoint
                self.save_checkpoint(self.current_step, avg_loss)
                
                # Print stats
                stats = self.logger.get_stats()
                epoch_time = time.time() - epoch_start
                
                print("\n" + "─"*60)
                print(f"📊 CHECKPOINT #{len(self.logger.history['checkpoints'])}")
                print(f"   Step: {self.current_step:,} / {self.config['max_steps']:,}")
                print(f"   Current Loss: {avg_loss:.4f}")
                print(f"   Best Loss: {stats['best_loss']:.4f}")
                print(f"   Improvement: {stats['improvement']:.4f}")
                print(f"   Epoch Time: {epoch_time:.1f}s")
                print("─"*60)
                
                # Validate periodically
                if self.current_step % self.config['validation_interval'] == 0:
                    self.validate()
                
                # Check if target achieved
                if avg_loss < self.config['target_loss']:
                    print("\n" + "🎉"*30)
                    print(f"🏆 TARGET ACHIEVED! Loss: {avg_loss:.4f} < {self.config['target_loss']}")
                    print("🎉"*30 + "\n")
                    break
                
                # Adaptive learning rate
                if self.current_step % 10000 == 0 and self.current_step > 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.9
                    print(f"   📉 Learning rate adjusted to {param_group['lr']:.6f}")
        
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
            self.save_checkpoint(self.current_step, avg_loss)
        
        except Exception as e:
            print(f"\n❌ Error occurred: {e}")
            self.save_checkpoint(self.current_step, avg_loss)
            raise
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETED")
        print("="*60)
        stats = self.logger.get_stats()
        print(f"Total Steps: {self.current_step:,}")
        print(f"Final Loss: {stats['current_loss']:.4f}")
        print(f"Best Loss: {stats['best_loss']:.4f}")
        print(f"Total Improvement: {stats['improvement']:.4f}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Autonomous AI Training System')
    parser.add_argument('--tokenizer', default='data/tokenizer', help='Tokenizer directory')
    parser.add_argument('--data', default='data/dataset.bin', help='Dataset file')
    parser.add_argument('--model', default='models/latest', help='Model directory')
    parser.add_argument('--logs', default='logs', help='Log directory')
    parser.add_argument('--target-loss', type=float, default=0.5, help='Target loss to achieve')
    parser.add_argument('--max-steps', type=int, default=100000000, help='Maximum training steps')
    parser.add_argument('--checkpoint-interval', type=int, default=500, help='Steps per checkpoint')
    parser.add_argument('--validation-interval', type=int, default=2000, help='Steps per validation')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    config = {
        'tokenizer_dir': args.tokenizer,
        'data_file': args.data,
        'model_dir': args.model,
        'log_dir': args.logs,
        'target_loss': args.target_loss,
        'max_steps': args.max_steps,
        'checkpoint_interval': args.checkpoint_interval,
        'validation_interval': args.validation_interval,
        'batch_size': args.batch_size,
        'learning_rate': args.lr
    }
    
    trainer = AutoTrainer(config)
    trainer.run()


if __name__ == '__main__':
    main()
