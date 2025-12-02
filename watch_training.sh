#!/bin/bash
# Real-time training progress monitor

echo "📊 Real-Time Training Monitor"
echo "Press Ctrl+C to exit"
echo ""

while true; do
    clear
    echo "════════════════════════════════════════════════════════════════"
    echo "           REAL-TIME TRAINING PROGRESS"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    
    # Check if training is running
    if pgrep -f "train.py" > /dev/null; then
        echo "✅ Status: TRAINING IN PROGRESS"
        echo ""
        
        # Show last 25 lines of progress
        if [ -f training_progress.log ]; then
            tail -25 training_progress.log | grep -E "(Epoch|loss|accuracy|ETA|saving|val_loss)" | tail -15
        fi
        
        # Show process info
        echo ""
        echo "────────────────────────────────────────────────────────────"
        ps aux | grep "train.py" | grep -v grep | awk '{printf "💻 CPU: %.1f%% | Memory: %.1f%% | Runtime: %s\n", $3, $4, $10}'
    else
        echo "⏸️  Status: TRAINING NOT RUNNING"
        echo ""
        if [ -f training_progress.log ]; then
            echo "Last log entries:"
            tail -10 training_progress.log
        fi
    fi
    
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "Refreshing in 3 seconds... (Ctrl+C to exit)"
    sleep 3
done

