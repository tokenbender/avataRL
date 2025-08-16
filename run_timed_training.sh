#!/bin/bash

# run_timed_training.sh - Timed AvataRL training script
# Runs avatarl.py with 8 GPUs and 600 iterations, tracking start/end/total time

set -e  # Exit on error

echo "=== Timed AvataRL Training Script ==="
echo "Configuration: 8 GPUs, 600 iterations"
echo

# Record start time
start_time=$(date +%s)
start_time_readable=$(date '+%Y-%m-%d %H:%M:%S')

echo "Training Start Time: $start_time_readable"
echo "Timestamp: $start_time"
echo "----------------------------------------"

# Run AvataRL training with 8 GPUs and 600 iterations
# Using torchrun for distributed training across 8 GPUs
torchrun --standalone --nproc_per_node=8 avatarl.py

# Record end time
end_time=$(date +%s)
end_time_readable=$(date '+%Y-%m-%d %H:%M:%S')

# Calculate total time
total_time=$((end_time - start_time))
total_hours=$((total_time / 3600))
total_minutes=$(((total_time % 3600) / 60))
total_seconds=$((total_time % 60))

echo "----------------------------------------"
echo "Training End Time: $end_time_readable"
echo "Timestamp: $end_time"
echo
echo "=== TRAINING SUMMARY ==="
echo "Start Time:  $start_time_readable"
echo "End Time:    $end_time_readable"
echo "Total Time:  ${total_hours}h ${total_minutes}m ${total_seconds}s"
echo "Total Seconds: $total_time"
echo "========================="
