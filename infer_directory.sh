#!/bin/bash

# Script to run inference on all JPG files and organize results
# Usage: ./infer_directory.sh

# Set paths
IMG_DIR="/localhome/aha220/Hairdar/assets/test_data/img"
OUTPUT_DIR="/localhome/aha220/Hairdar/modules/difflocks/outputs_inference"
RESULTS_BASE_DIR="/localhome/aha220/Hairdar/assets/results/Difflocks_new/hair3D"
INFERENCE_SCRIPT="./inference_difflocks.py"

# Check if the image directory exists
if [ ! -d "$IMG_DIR" ]; then
    echo "Error: Image directory $IMG_DIR does not exist"
    exit 1
fi

# Check if the inference script exists
if [ ! -f "$INFERENCE_SCRIPT" ]; then
    echo "Error: Inference script $INFERENCE_SCRIPT does not exist"
    exit 1
fi

# Create base results directory if it doesn't exist
mkdir -p "$RESULTS_BASE_DIR"

# Process each JPG file in the directory
for img_file in "$IMG_DIR"/*.jpg; do
    # Check if any JPG files exist
    if [ ! -f "$img_file" ]; then
        echo "No JPG files found in $IMG_DIR"
        exit 1
    fi
    
    # Extract filename without path and extension
    filename=$(basename "$img_file" .jpg)
    
    echo "Processing: $filename"
    
    # Create output directory for this specific image
    result_dir="$RESULTS_BASE_DIR/$filename"    
    mkdir -p "$result_dir"
    
    # Run inference
    echo "Running inference on $img_file..."
    python "$INFERENCE_SCRIPT" \
        --img_path="$img_file" \
        --out_path="$OUTPUT_DIR"
    
    # Check if inference was successful
    if [ $? -ne 0 ]; then
        echo "Error: Inference failed for $img_file"
        continue
    fi
    
    # Move results to organized directory
    if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR 2>/dev/null)" ]; then
        echo "Moving results to $result_dir..."
        mv "$OUTPUT_DIR"/* "$result_dir/"
        echo "Results moved successfully for $filename"
    else
        echo "Warning: No output files found for $filename"
    fi
    
    echo "Completed processing: $filename"
    echo "----------------------------------------"
done

echo "All files processed successfully!"