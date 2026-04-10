#!/bin/bash
# Install EfficientViT-SAM for Jetson (aarch64)
set -e

echo "=== Cloning EfficientViT ==="
git clone https://github.com/mit-han-lab/efficientvit /tmp/efficientvit_src
pip install -e /tmp/efficientvit_src

echo "=== Downloading SAM l0 weights ==="
mkdir -p weights
wget -O weights/efficientvit_sam_l0.pt \
  https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l0.pt

echo "=== Done ==="
echo "Update SAM_WEIGHTS in detect.py to: weights/efficientvit_sam_l0.pt"
