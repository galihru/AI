#!/usr/bin/env python3
"""
Quick script to fetch and test latest model from GitHub
"""
import subprocess
import json
import sys

print("🔄 Fetching latest AI model from cloud...")

# Pull from GitHub
result = subprocess.run(['git', 'pull'], capture_output=True, text=True)
if result.returncode == 0:
    print("✅ Successfully pulled latest model")
else:
    print(f"❌ Git pull failed: {result.stderr}")
    sys.exit(1)

# Check metadata
try:
    with open('models/latest/metadata.json', 'r') as f:
        meta = json.load(f)
    
    print("\n📊 Model Status:")
    print(f"  Total steps: {meta['total_steps']:,}")
    print(f"  Best loss: {meta['best_loss']:.4f}")
    print(f"  Training hours: {meta['training_hours']:.2f}")
    print(f"  Last update: {meta['timestamp']}")
    print(f"  Device: {meta['device']}")
    
    # Check if ready
    if meta['best_loss'] < 3.0:
        print("\n✅ Model is ready for use!")
    elif meta['best_loss'] < 4.0:
        print("\n⚠️  Model is training... check back later")
    else:
        print("\n⏳ Model is still in early training")
    
except FileNotFoundError:
    print("❌ Model metadata not found")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error reading metadata: {e}")
    sys.exit(1)
