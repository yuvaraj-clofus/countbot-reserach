#!/usr/bin/env python3
"""
YOLOE segmentation test runner for object detection on videos or images.

Why this script exists:
  - Uses a single segmentation model pass for boxes + masks
  - Lets you test open-vocabulary classes via `--classes`
  - Works with local `.pt` checkpoints such as `yolo26s-seg.pt` or `yoloe-26s-seg.pt`
  - Saves an annotated output video/image for quick comparison with FastSAM

Examples:
  python3 detect.py \
      --model /path/to/yoloe-11s-seg.pt \
      --source /home/vikbot/Documents/countbot/testing/video/bearing_wheel.mp4 \
      --classes bearing wheel

  python3 detect.py \
      --model /path/to/yoloe-26s-seg.pt \
      --source 0 \
      --classes phone
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import cv2
import torch

torch.backends.cudnn.benchmark = False

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "weights" / "yolo26s-seg.engine"
SOURCE_PATH = SCRIPT_DIR.parent / "video" / "bearing_wheel.mp4"
OUTPUT_PATH = SCRIPT_DIR / "runs" / "yoloe_output.mp4"
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
DEFAULT_CLASSES: list[str] = []
DEFAULT_PROJECT_MASKS = True
DEFAULT_SHOW = False
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOE segmentation inference on an image, video, or webcam.")
    parser.add_argument(
        "--model",
        default=str(MODEL_PATH),
        help="Path to a segmentation checkpoint, e.g. yolo26s-seg.pt or yoloe-26s-seg.pt",
    )
    parser.add_argument(
        "--source",
        default=str(SOURCE_PATH),
        help="Input source: image path, video path, directory, or webcam index like 0",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help="Where to save the annotated output. Use '' to disable saving.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Class names for true YOLOE checkpoints. Plain segmentation checkpoints ignore this.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.50, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="cuda or cpu")
    parser.add_argument(
        "--project-masks",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_PROJECT_MASKS,
        help="Blend the segmentation masks into the output",
    )
    parser.add_argument("--hide-boxes", action="store_true", help="Hide bounding boxes in the output")
    parser.add_argument("--hide-labels", action="store_true", help="Hide labels in the output")
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SHOW,
        help="Show a live preview window",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional safety limit for quick tests. 0 means process the whole source.",
    )
    return parser.parse_args()


def resolve_source(source: str) -> str | int:
    source = source.strip()
    if source.isdigit():
        return int(source)
    return source


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_model(model_path: str, class_names: list[str], device: str):
    try:
        from ultralytics import YOLO, YOLOE
    except ImportError:
        print("[error] ultralytics is not installed.")
        print("        Install it with: pip install ultralytics")
        sys.exit(1)

    model_hint = Path(model_path)
    if model_hint.suffix == ".pt" and not model_hint.exists() and not model_hint.is_absolute():
        model_hint = (Path.cwd() / model_hint).resolve()

    requested_device = str(device).lower()
    engine_backend = model_hint.suffix == ".engine"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        if engine_backend:
            print("[warn] PyTorch reports no CUDA device, but an engine file was requested.")
            print("       Trying TensorRT directly with device=cuda.")
        else:
            print("[warn] CUDA was requested but no CUDA device is available. Falling back to cpu.")
            requested_device = "cpu"

    if engine_backend and requested_device == "cpu":
        print(f"[error] TensorRT engine requires CUDA, but device={requested_device}.")
        print(f"        Engine file: {model_hint}")
        print("        This script will not fall back to ONNX/PT when an engine is requested.")
        print("        Run on a CUDA-capable device or pass a non-engine checkpoint explicitly.")
        sys.exit(1)

    if str(model_path).endswith(".pt") and not model_hint.exists():
        print(f"[error] YOLOE checkpoint not found: {model_hint}")
        print("        This environment is offline, so Ultralytics cannot auto-download missing weights.")
        print("        Put a segmentation checkpoint there or pass one with --model.")
        print("        Examples: yoloe-11s-seg.pt, yoloe-26s-seg.pt")
        sys.exit(1)

    selected_model = None
    model_kind = "YOLO"

    # Try YOLOE first only when the user asks for prompt classes or the filename looks like a YOLOE checkpoint.
    wants_yoloe = bool(class_names) or "yoloe" in model_hint.name.lower()
    if wants_yoloe:
        try:
            yoloe_model = YOLOE(model_path)
            if class_names:
                yoloe_model.set_classes(class_names)
            selected_model = yoloe_model
            model_kind = "YOLOE"
        except Exception as exc:
            print(f"[warn] YOLOE prompt setup unavailable for {model_hint.name}: {exc}")
            if class_names:
                print("[warn] falling back to plain segmentation inference without open-vocabulary classes")

    if selected_model is None:
        try:
            selected_model = YOLO(model_path)
        except Exception as exc:
            print(f"[error] failed to load segmentation model: {exc}")
            sys.exit(1)

    model_names = getattr(selected_model.model, "names", {})
    preview_names = []
    if isinstance(model_names, dict):
        preview_names = [str(model_names[k]) for k in list(model_names.keys())[:6]]

    print(f"[init] model={model_path}")
    print(f"[init] backend={model_kind} task={selected_model.task} device={requested_device}")
    if class_names and model_kind == "YOLOE":
        print(f"[init] classes={class_names}")
    elif class_names:
        print(f"[init] classes ignored for this checkpoint: {class_names}")
    else:
        print(f"[init] default classes from checkpoint: {preview_names}{' ...' if len(model_names) > 6 else ''}")
    return selected_model, requested_device


def annotate(result, args: argparse.Namespace, fps_text: str) -> np.ndarray:
    frame = result.plot(
        conf=True,
        boxes=not args.hide_boxes,
        labels=not args.hide_labels,
        masks=args.project_masks,
        color_mode="class",
    )

    count = 0 if result.boxes is None else len(result.boxes)
    cv2.putText(
        frame,
        f"{fps_text} | detections={count}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame


def resize_frame(frame):
    return cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))


def run_video(model, source, output_path: str, args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[error] cannot open source: {source}")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_path:
        out_path = Path(output_path)
        ensure_parent(out_path)
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            src_fps,
            (DISPLAY_WIDTH, DISPLAY_HEIGHT),
        )

    print(f"[run] source={source}")
    print(f"[run] size={DISPLAY_WIDTH}x{DISPLAY_HEIGHT} fps={src_fps:.2f} frames={total if total > 0 else 'unknown'}")
    print("[run] press Q to quit")

    frame_idx = 0
    started = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            frame = resize_frame(frame)
            infer_t0 = time.time()
            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )
            infer_ms = (time.time() - infer_t0) * 1000.0
            fps = frame_idx / max(time.time() - started, 1e-6)
            fps_text = f"fps={fps:.1f} infer={infer_ms:.0f}ms frame={frame_idx}"
            annotated = annotate(results[0], args, fps_text)

            if writer is not None:
                writer.write(annotated)

            if args.show:
                cv2.imshow("YOLOE Segmentation Detection", annotated)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            if args.max_frames and frame_idx >= args.max_frames:
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    if output_path:
        print(f"[done] saved annotated video -> {output_path}")


def run_image(model, source: str, output_path: str, args: argparse.Namespace) -> None:
    img = cv2.imread(source)
    if img is None:
        print(f"[error] cannot read image: {source}")
        sys.exit(1)
    img = resize_frame(img)

    infer_t0 = time.time()
    results = model.predict(
        source=img,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )
    infer_ms = (time.time() - infer_t0) * 1000.0
    annotated = annotate(results[0], args, f"infer={infer_ms:.0f}ms")

    if output_path:
        out_path = Path(output_path)
        if out_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            out_path = out_path.with_suffix(".jpg")
        ensure_parent(out_path)
        cv2.imwrite(str(out_path), annotated)
        print(f"[done] saved annotated image -> {out_path}")

    if args.show:
        cv2.imshow("YOLOE Segmentation Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()

    print(f"[init] source={args.source}")
    print(f"[init] output={args.output}")
    print(f"[init] display_size={DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
    source = resolve_source(args.source)
    model, args.device = load_model(args.model, args.classes, args.device)

    if isinstance(source, int):
        run_video(model, source, args.output, args)
        return

    suffix = Path(source).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        run_image(model, source, args.output, args)
    else:
        run_video(model, source, args.output, args)


if __name__ == "__main__":
    main()
