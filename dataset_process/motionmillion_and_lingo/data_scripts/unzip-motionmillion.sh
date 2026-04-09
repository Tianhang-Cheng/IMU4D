#!/bin/bash

# Script to extract all tar.gz files under motion_272rpr directory

echo "Starting extraction of all tar.gz files..."

# Find all tar.gz files under motion_272rpr and extract them
find MotionMillion/motion_272rpr -name "*.tar.gz" -type f | while read -r tarfile; do

    dir=$(dirname "$tarfile")
    filename=$(basename "$tarfile")
    
    echo "Extracting $filename in directory $dir"
    tar -xzvf "$tarfile" -C "$dir"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully extracted $filename"
    else
        echo "✗ Failed to extract $filename"
    fi
    echo "---"
done

echo "All extractions completed!"