#!/bin/bash
# Run this script to download the models to the workspace
# Usage: ./setup_weights.sh

echo "Downloading NVIDIA-Reason-CXR-3B..."
hf download nvidia/NV-Reason-CXR-3B --local-dir ../checkpoints/nvidia-reason-3b

echo "Downloading MedGemma-1.5-4B..."
hf download google/medgemma-1.5-4b-it --local-dir ../checkpoints/medgemma-1.5-4b

echo "Done! Models are ready in ../checkpoints/"

