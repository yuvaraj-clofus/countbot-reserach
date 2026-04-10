#!/bin/bash
# Install MobileSAM for Jetson (aarch64)
set -e

echo "=== Installing dependencies ==="
pip install timm

echo "=== Installing MobileSAM ==="
pip install git+https://github.com/ChaoningZhang/MobileSAM.git

echo "=== Downloading MobileSAM weights ==="
mkdir -p weights
wget -O weights/mobile_sam.pt \
  "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"

echo ""
echo "=== Done ==="
echo "Run:  python3 detect.py"
