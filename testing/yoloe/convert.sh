#!/bin/bash
set -e

cd "$(dirname "$0")"

ONNX="/home/vikbot/Documents/countbot/testing/yoloe/weights/yolo26s-seg.onnx"
ENGINE="/home/vikbot/Documents/countbot/testing/yoloe/weights/yolo26s-seg.engine"

echo "[1/2] Exporting PT -> ONNX ..."
python3 convert.py

echo ""
echo "[2/2] Converting ONNX -> TensorRT engine ..."
echo "      This may stay quiet for a few minutes while TensorRT profiles tactics."
/usr/src/tensorrt/bin/trtexec \
    --onnx="$ONNX" \
    --saveEngine="$ENGINE" \
    --fp16 \
    --memPoolSize=workspace:4096 \
    --skipInference

echo ""
echo "[done] TensorRT engine saved -> $ENGINE"
