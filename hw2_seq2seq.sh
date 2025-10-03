#!/bin/bash

DATA_DIR=$1
OUTPUT_FILE=$2

# Create SavedModel folder if needed
mkdir -p SavedModel

# Download model if missing
if [ ! -f "SavedModel/model_final.pth" ]; then
    echo "Downloading model..."
    pip install gdown -q
    gdown https://drive.google.com/uc?id=1RevHMfXZ1zYjUm4fPU1CfFKAjyMJjdgJ -O SavedModel/model_final.pth
fi

# Run the test
python3 test_model.py "$DATA_DIR" "$OUTPUT_FILE"
