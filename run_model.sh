#!/bin/bash

# This script is the entry point for your submission.

if [ "$1" = "test1" ]; then
    # Task 1 Inference Mode (Author Verification)
    TEST_FILE=$2
    OUTPUT_DIR=$3

    # Ensure output dir exists
    mkdir -p "$OUTPUT_DIR"

    echo "Running Task 1 Inference..."
    python3 src/inference_task1.py "$TEST_FILE" "$OUTPUT_DIR"

elif [ "$1" = "test2" ]; then
    # Task 2 Inference Mode (Author Clustering)
    TEST_FILE=$2
    OUTPUT_DIR=$3

    # Ensure output dir exists
    mkdir -p "$OUTPUT_DIR"

    echo "Running Task 2 Inference..."
    python3 src/inference_task2.py "$TEST_FILE" "$OUTPUT_DIR"

else
    # Training Mode
    TRAIN_DIR=$1

    echo "Training model on data in $TRAIN_DIR..."
    python3 src/train.py "$TRAIN_DIR"
fi