#!/usr/bin/env python3
"""
Training Monitor Dashboard
Real-time monitoring of autonomous training progress.
"""

import json
import os
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path


class TrainingMonitor:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, 'training_log.json')
    
    def load_history(self):
        if not os.path.exists(self.log_file):
            return None
        
        with open(self.log_file, 'r') as f:
            return json.load(f)
    
    def print_status(self):
        history = self.load_history()
        
        if not history or not history['steps']:
            print("❌ No training data found")
            return
        
        current_step = history['steps'][-1]
        current_loss = history['losses'][-1]
        best_loss = history['best_loss']
        
        # Calculate stats
        recent_losses = history['losses'][-100:]
        avg_recent = sum(recent_losses) / len(recent_losses)
        
        improvement = history['losses'][0] - current_loss if len(history['losses']) > 1 else 0
        improvement_pct = (improvement / history['losses'][0] * 100) if history['losses'][0] > 0 else 0
        
        # Estimate time
        if len(history['timestamps']) > 1:
            start_time = datetime.fromisoformat(history['timestamps'][0])
            current_time = datetime.fromisoformat(history['timestamps'][-1])
            elapsed = current_time - start_time
            steps_per_hour = len(history['steps']) / (elapsed.total_seconds() / 3600) if elapsed.total_seconds() > 0 else 0
        else:
            elapsed = timedelta(0)
            steps_per_hour = 0
        
        # Print dashboard
        print("\n" + "="*70)
        print("🤖 ARDUSCRATCH AI - AUTONOMOUS TRAINING MONITOR")
        print("="*70)
        print(f"📍 Status: TRAINING")
        print(f"⏱️  Elapsed: {str(elapsed).split('.')[0]}")
        print(f"📊 Progress:")
        print(f"   Current Step: {current_step:,}")
        print(f"   Current Loss: {current_loss:.4f}")
        print(f"   Best Loss: {best_loss:.4f} ⭐")
        print(f"   Avg (last 100): {avg_recent:.4f}")
        print(f"\n📈 Improvement:")
        print(f"   Total: {improvement:.4f} ({improvement_pct:.1f}%)")
        print(f"\n⚡ Speed:")
        print(f"   {steps_per_hour:.0f} steps/hour")
        
        # Estimate to target
        if current_loss > 0.5 and steps_per_hour > 0:
            # Rough estimate assuming linear decrease
            steps_to_target = ((current_loss - 0.5) / (improvement / current_step)) * current_step if improvement > 0 else float('inf')
            hours_to_target = steps_to_target / steps_per_hour if steps_per_hour > 0 else float('inf')
            
            if hours_to_target < 1000:
                print(f"\n🎯 Estimated to Loss 0.5:")
                print(f"   ~{hours_to_target:.1f} hours (~{hours_to_target/24:.1f} days)")
        
        # Recent checkpoints
        if history['checkpoints']:
            print(f"\n💾 Recent Checkpoints:")
            for cp in history['checkpoints'][-5:]:
                print(f"   Step {cp['step']:,}: Loss {cp['loss']:.4f}")
        
        print("="*70 + "\n")
    
    def plot_loss(self, save_path='training_plot.png'):
        history = self.load_history()
        
        if not history or not history['steps']:
            print("No data to plot")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['steps'], history['losses'], alpha=0.3, label='Loss')
        
        # Moving average
        window = min(100, len(history['losses']) // 10)
        if window > 1:
            moving_avg = []
            for i in range(len(history['losses'])):
                start = max(0, i - window)
                moving_avg.append(sum(history['losses'][start:i+1]) / (i - start + 1))
            plt.plot(history['steps'], moving_avg, label=f'Moving Avg ({window})', linewidth=2)
        
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot improvement
        plt.subplot(1, 2, 2)
        if len(history['losses']) > 1:
            improvements = [history['losses'][0] - loss for loss in history['losses']]
            plt.plot(history['steps'], improvements)
            plt.xlabel('Steps')
            plt.ylabel('Improvement from Start')
            plt.title('Total Improvement')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"📊 Plot saved: {save_path}")
    
    def watch(self, interval=30):
        """Continuously monitor training."""
        print("👁️  Watching training... (Ctrl+C to stop)")
        
        try:
            while True:
                os.system('cls' if os.name == 'nt' else 'clear')
                self.print_status()
                print(f"⏳ Next update in {interval}s...")
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n✅ Monitoring stopped")


if __name__ == '__main__':
    import sys
    
    monitor = TrainingMonitor()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'watch':
            interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30
            monitor.watch(interval)
        elif sys.argv[1] == 'plot':
            monitor.plot_loss()
        else:
            monitor.print_status()
    else:
        monitor.print_status()
