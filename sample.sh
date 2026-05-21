#!/bin/bash

CLASSES=("airplane" "automobile" "bird" "cat" "deer" "dog" "frog" "horse" "ship" "truck")

for class in "${CLASSES[@]}"; do
    echo "Generating: $class"
    uv run python sample.py --prompt "a photo of $class" --w 5.0
done
