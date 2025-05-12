#!/bin/bash

# Activate the virtual environment
source /home/psxkf4/Genesis/myenv/bin/activate

# Training loop (run exactly 2 times)
for i in {1..2}; do
    echo "Starting training (Iteration $i)..."
    python slosh_free_train.py
    
    echo "Training finished (Iteration $i)."
    
    # Optional: short pause before the next iteration
    if [ "$i" -lt 2 ]; then
        sleep 5
    fi
done

echo "Completed 2 training iterations."
