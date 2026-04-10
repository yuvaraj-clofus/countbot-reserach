#!/bin/bash
# Complete GPU fix for Jetson R36.5 (JetPack 6.2)
# Run with: sudo bash fix_gpu.sh
set -e

echo "=== Step 1: Add NVIDIA CUDA apt repository ==="
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update -q

echo "=== Step 2: Install libcusparseLt ==="
apt-get install -y libcusparselt0 libcusparselt-dev

echo "=== Step 3: Verify library is present ==="
find /usr -name "libcusparseLt.so*" 2>/dev/null
ldconfig
ldconfig -p | grep cusparseLt

echo "=== Step 4: Install Jetson-native PyTorch ==="
WHEEL="/home/vikbot/Documents/countbot/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
su -c "pip install --no-deps $WHEEL" vikbot

echo "=== Step 5: Build torchvision from source ==="
apt-get install -y libpython3-dev libopenblas-dev
su -c "cd /tmp/torchvision && FORCE_CUDA=1 python3 setup.py install --user" vikbot

echo "=== Step 6: Verify GPU ==="
su -c 'python3 -c "import torch; print(\"torch:\", torch.__version__); print(\"cuda:\", torch.cuda.is_available()); print(\"gpu:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")"' vikbot

echo "=== Done ==="
