#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Logic to handle different execution modes specified in the assignment
if [ "$1" == "test1" ]; then
    # ==========================================
    # Task 1: Author Verification and Ranking
    # Usage: ./run_model.sh test1 <test_file> <output_dir>
    # ==========================================
    echo "Running Task 1 Inference..."
    python3 -m src.inference_task1 "$2" "$3"

elif [ "$1" == "test2" ]; then
    # ==========================================
    # Task 2: Author Clustering
    # Usage: ./run_model.sh test2 <test_file> <output_dir>
    # ==========================================
    echo "Running Task 2 Inference..."
    python3 -m src.inference_task2 "$2" "$3"

else
    # ==========================================
    # Training Mode
    # Usage: ./run_model.sh <train_dir>
    # ==========================================
    echo "Running Training..."
    # $1 is the train_dir passed by the TA (e.g., data/train_data/)
    python3 main.py "$1"
fi