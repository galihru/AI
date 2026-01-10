#!/usr/bin/env python3
"""
Autonomous AI Training System with Cloud Integration
Trains continuously until target loss is achieved
Auto-commits progress to GitHub
"""
import os
import sys
import time
import json
import subprocess
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm

# Configuration
CONFIG = {
    "target_loss": 0.5,              # Target loss (mendekati 0)
    "min_loss": 0.1,                 # Minimum achievable loss
    "checkpoint_every": 500,         # Save every N steps
    "validate_every": 100,           # Validate every N steps
    "auto_commit": True,             # Auto commit to GitHub
    "commit_every": 1000,            # Commit every N steps
    "max_hours_per_session": 5,     # Max hours per session (for cloud limits)
    "learning_rate_schedule": True,  # Adaptive learning rate
    "early_stopping": False,         # Don't stop until target reached
}

class ContinuousTrainer:
    def __init__(self, tokenizer_path, data_path, model_path, config):
        self.config = config
        self.tokenizer_path = tokenizer_path
        self.data_path = data_path
        self.model_path = model_path
        self.start_time = time.time()
        self.total_steps = 0
        self.best_loss = float('inf')
        self.loss_history = []
        
        # Load components
        self.load_components()
    
    def load_components(self):
        """Load tokenizer, model, and dataset."""
        print("Loading components...")
        
        # Tokenizer
        self.tokenizer = Tokenizer.from_file(f"{self.tokenizer_path}/tokenizer.json")
        vocab_size = self.tokenizer.get_vocab_size()
        
        # Dataset
        from train_lm import TokenDataset
        self.dataset = TokenDataset(self.data_path, block_size=512)
        self.loader = DataLoader(self.dataset, batch_size=4, shuffle=True, num_workers=0)
        
        # Model
        from train_lm import GPT, GPTConfig
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {self.device}")
        
        # Load or create model
        if os.path.exists(f"{self.model_path}/model.pt"):
            print("Loading existing model...")
            checkpoint = torch.load(f"{self.model_path}/model.pt", map_location=self.device)
            config_dict = checkpoint['config']
            self.model_config = GPTConfig(**config_dict)
            self.model = GPT(self.model_config)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.total_steps = checkpoint.get('total_steps', 0)
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"Resumed from step {self.total_steps}, best loss: {self.best_loss:.4f}")
        else:
            print("Creating new model...")
            self.model_config = GPTConfig(vocab_size=vocab_size)
            self.model = GPT(self.model_config)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4)
        
        self.model = self.model.to(self.device)
        
    def adjust_learning_rate(self, loss):
        """Adaptive learning rate based on loss."""
        if not self.config['learning_rate_schedule']:
            return
        
        if loss < 2.0:
            lr = 1e-4
        elif loss < 3.0:
            lr = 2e-4
        else:
            lr = 3e-4
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def save_checkpoint(self, loss, force=False):
        """Save model checkpoint."""
        if not force and self.total_steps % self.config['checkpoint_every'] != 0:
            return
        
        os.makedirs(self.model_path, exist_ok=True)
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.model_config.__dict__,
            'total_steps': self.total_steps,
            'best_loss': self.best_loss,
            'loss_history': self.loss_history[-100:],  # Last 100 losses
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, f"{self.model_path}/model.pt")
        
        # Save metadata
        metadata = {
            'total_steps': self.total_steps,
            'current_loss': loss,
            'best_loss': self.best_loss,
            'timestamp': datetime.now().isoformat(),
            'device': self.device,
            'training_hours': (time.time() - self.start_time) / 3600
        }
        with open(f"{self.model_path}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Checkpoint saved (step {self.total_steps}, loss {loss:.4f})")
    
    def commit_to_github(self):
        """Auto commit progress to GitHub."""
        if not self.config['auto_commit']:
            return
        
        if self.total_steps % self.config['commit_every'] != 0:
            return
        
        try:
            commit_msg = f"Auto-training: Step {self.total_steps}, Loss {self.best_loss:.4f}"
            
            # Add model files
            subprocess.run(['git', 'add', f'{self.model_path}/*'], check=False)
            
            # Commit
            result = subprocess.run(
                ['git', 'commit', '-m', commit_msg],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Push
                subprocess.run(['git', 'push'], check=False)
                print(f"✓ Committed to GitHub: {commit_msg}")
        except Exception as e:
            print(f"⚠ Git commit failed: {e}")
    
    def should_continue(self):
        """Check if training should continue."""
        # Check target loss
        if self.best_loss <= self.config['target_loss']:
            print(f"🎉 Target loss {self.config['target_loss']} reached!")
            return False
        
        # Check time limit (for cloud sessions)
        hours_elapsed = (time.time() - self.start_time) / 3600
        if hours_elapsed >= self.config['max_hours_per_session']:
            print(f"⏰ Session time limit reached ({hours_elapsed:.1f}h)")
            return False
        
        return True
    
    def train_forever(self):
        """Train until target loss is achieved."""
        print("\n" + "="*60)
        print("🚀 AUTONOMOUS AI TRAINING STARTED")
        print("="*60)
        print(f"Target Loss: {self.config['target_loss']}")
        print(f"Current Best: {self.best_loss:.4f}")
        print(f"Device: {self.device}")
        print(f"Auto-commit: {self.config['auto_commit']}")
        print("="*60 + "\n")
        
        self.model.train()
        iter_loader = iter(self.loader)
        
        with tqdm(desc="Training", unit="step") as pbar:
            while self.should_continue():
                try:
                    x, y = next(iter_loader)
                except StopIteration:
                    iter_loader = iter(self.loader)
                    x, y = next(iter_loader)
                
                x, y = x.to(self.device), y.to(self.device)
                
                # Forward pass
                logits, loss = self.model(x, y)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                # Track progress
                self.total_steps += 1
                current_loss = loss.item()
                self.loss_history.append(current_loss)
                
                # Update best loss
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.save_checkpoint(current_loss, force=True)
                
                # Adjust learning rate
                self.adjust_learning_rate(current_loss)
                
                # Save checkpoint
                self.save_checkpoint(current_loss)
                
                # Commit to GitHub
                self.commit_to_github()
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'best': f'{self.best_loss:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
                
                # Log every 100 steps
                if self.total_steps % 100 == 0:
                    avg_loss = np.mean(self.loss_history[-100:])
                    print(f"\nStep {self.total_steps}: loss={current_loss:.4f}, avg={avg_loss:.4f}, best={self.best_loss:.4f}")
        
        # Final save
        self.save_checkpoint(self.best_loss, force=True)
        print("\n" + "="*60)
        print(f"🎯 Training session completed!")
        print(f"Total steps: {self.total_steps}")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"Time elapsed: {(time.time() - self.start_time)/3600:.2f} hours")
        print("="*60)


def main():
    # Paths
    tokenizer_path = "data/tokenizer"
    data_path = "data/dataset.bin"
    model_path = "models/latest"
    
    # Create trainer
    trainer = ContinuousTrainer(tokenizer_path, data_path, model_path, CONFIG)
    
    # Train forever
    trainer.train_forever()


if __name__ == '__main__':
    main()
