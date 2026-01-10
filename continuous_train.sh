#!/bin/bash
# Continuous Training Loop for Railway/Cloud platforms
# This script runs indefinitely and auto-restarts on timeout

echo "🚀 ArduScratch Continuous Training System"
echo "=========================================="

# Configuration
export TARGET_LOSS=0.5
export SESSION_HOURS=5
export AUTO_RESTART=true

while true; do
    echo ""
    echo "Starting training session at $(date)"
    echo "Target loss: $TARGET_LOSS"
    echo "Session limit: $SESSION_HOURS hours"
    
    # Run training
    python scripts/autonomous_trainer.py
    
    EXIT_CODE=$?
    
    # Check exit reason
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Training completed successfully!"
        
        # Check if target reached
        BEST_LOSS=$(python -c "import json; print(json.load(open('models/latest/metadata.json'))['best_loss'])")
        
        if (( $(echo "$BEST_LOSS <= $TARGET_LOSS" | bc -l) )); then
            echo "🎉 Target loss $TARGET_LOSS reached! (Best: $BEST_LOSS)"
            echo "Training complete. Exiting."
            break
        fi
    else
        echo "⚠️ Training session ended (exit code: $EXIT_CODE)"
    fi
    
    # Commit progress to git
    if [ -d ".git" ]; then
        echo "Committing progress to git..."
        git add models/
        git commit -m "Auto-checkpoint: Loss $BEST_LOSS at $(date)" || true
        git push || true
    fi
    
    # Check if should restart
    if [ "$AUTO_RESTART" = "false" ]; then
        echo "Auto-restart disabled. Exiting."
        break
    fi
    
    # Wait before restart
    echo "Waiting 60 seconds before restarting..."
    sleep 60
done

echo "=========================================="
echo "Training system stopped"
