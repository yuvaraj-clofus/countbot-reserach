#!/usr/bin/env python3
"""
Export the local YOLO26 segmentation checkpoint to ONNX.

TensorRT engine building is handled by convert.sh, which calls trtexec directly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "weights" / "yolo26s-seg.pt"
ONNX_PATH = SCRIPT_DIR / "weights" / "yolo26s-seg.onnx"
ENGINE_PATH = SCRIPT_DIR / "weights" / "yolo26s-seg.engine"
IMGSZ = 640


def export_onnx() -> None:
    if not MODEL_PATH.is_file():
        print(f"[error] model not found: {MODEL_PATH}")
        sys.exit(1)

    print(f"[1/2] Exporting ONNX from {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    exported_path = model.export(format="onnx", imgsz=IMGSZ, opset=17, simplify=True)
    exported_path = Path(str(exported_path))

    if exported_path.resolve() != ONNX_PATH.resolve():
        exported_path.replace(ONNX_PATH)

    if not ONNX_PATH.is_file():
        print(f"[error] ONNX export did not create: {ONNX_PATH}")
        sys.exit(1)

    print(f"[1/2] ONNX saved -> {ONNX_PATH}")


def main() -> None:
    print(f"[init] imgsz={IMGSZ}")
    export_onnx()
    print(f"[next] build TensorRT from ONNX with: /usr/src/tensorrt/bin/trtexec --onnx={ONNX_PATH} --saveEngine={ENGINE_PATH} --fp16 --memPoolSize=workspace:4096 --skipInference")


if __name__ == "__main__":
    main()
