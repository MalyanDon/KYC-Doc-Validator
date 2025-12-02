#!/bin/bash
# Monitor training progress

echo "📊 Training Monitor"
echo "=================="
echo ""

while true; do
    clear
    echo "📊 Training Progress - $(date '+%H:%M:%S')"
    echo "=========================================="
    echo ""
    
    # Check if training is running
    if pgrep -f "train.py" > /dev/null; then
        echo "✅ Training is RUNNING"
        echo ""
        
        # Show last 20 lines of log
        if [ -f training_progress.log ]; then
            tail -20 training_progress.log
        else
            echo "Waiting for log file..."
        fi
    else
        echo "❌ Training is NOT running"
        echo ""
        if [ -f training_progress.log ]; then
            echo "Last log entries:"
            tail -10 training_progress.log
        fi
    fi
    
    echo ""
    echo "Press Ctrl+C to exit"
    echo "Refreshing in 5 seconds..."
    sleep 5
done

