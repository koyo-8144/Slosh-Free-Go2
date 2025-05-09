#!/bin/bash

# Activate the virtual environment
source /home/psxkf4/Genesis/myenv/bin/activate

# Training loop
while true; do
    echo "Starting training..."
    python slosh_free_train.py
    
    echo "Training finished. Restarting..."
    sleep 5  # Optional: short pause before restarting
done
