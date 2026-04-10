#!/usr/bin/env python3
"""Simple DINOv2 FPS checker for a folder of crop images."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

import stage2


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGES_DIR = SCRIPT_DIR / "stage1" / "run" / "crops"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check average DINOv2 FPS on a folder of images.")
    parser.add_argument(
        "--images-dir",
        default=str(DEFAULT_IMAGES_DIR),
        help="Folder containing crop images",
    )
    parser.add_argument("--device", default="cuda", help="Use cuda for GPU-only testing")
    parser.add_argument("--model-name", default="dinov2_vits14", help="DINOv2 model name")
    parser.add_argument("--max-items", type=int, default=0, help="Optional image limit")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup images before timing")
    return parser.parse_args()


def list_images(images_dir: Path) -> list[Path]:
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    images: list[Path] = []
    for pattern in patterns:
        images.extend(images_dir.glob(pattern))
    return sorted(images)


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        raise SystemExit(f"[fps] images folder not found: {images_dir}")

    image_paths = list_images(images_dir)
    if args.max_items > 0:
        image_paths = image_paths[: args.max_items]
    if not image_paths:
        raise SystemExit(f"[fps] no images found in: {images_dir}")

    torch, F, Image, transforms = stage2.require_stage2_deps()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "[fps] CUDA requested, but GPU is not available. "
            "This script is GPU-only and will not fall back to CPU."
        )

    load_start = time.perf_counter()
    model, tf, resolved_device = stage2.load_dino(args.model_name, args.device, torch, transforms)
    if args.device.startswith("cuda") and not str(resolved_device).startswith("cuda"):
        raise SystemExit(
            f"[fps] expected CUDA, but model loaded on {resolved_device}. "
            "This script is GPU-only and will not run on CPU."
        )
    load_time = time.perf_counter() - load_start

    warmup_count = min(max(0, args.warmup), len(image_paths))
    for path in image_paths[:warmup_count]:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        _ = stage2.get_embedding(image, model, tf, resolved_device, torch, F, Image)

    timed_paths = image_paths[warmup_count:] if warmup_count < len(image_paths) else image_paths
    if not timed_paths:
        timed_paths = image_paths

    total_images = 0
    total_seconds = 0.0

    print(f"[fps] images_dir={images_dir}")
    print(f"[fps] images={len(image_paths)} warmup={warmup_count} timed={len(timed_paths)}")
    print(f"[fps] device={resolved_device} model={args.model_name}")
    print(f"[fps] model_load_sec={load_time:.3f}")

    for path in timed_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[fps] skip unreadable image: {path}")
            continue

        start = time.perf_counter()
        embedding = stage2.get_embedding(image, model, tf, resolved_device, torch, F, Image)
        elapsed = time.perf_counter() - start
        if embedding is None:
            print(f"[fps] skip empty embedding: {path}")
            continue

        total_images += 1
        total_seconds += elapsed

    if total_images == 0:
        raise SystemExit("[fps] no valid images were processed")

    avg_sec = total_seconds / total_images
    avg_ms = avg_sec * 1000.0
    avg_fps = total_images / total_seconds if total_seconds > 0 else 0.0

    print(f"[fps] processed={total_images}")
    print(f"[fps] total_infer_sec={total_seconds:.3f}")
    print(f"[fps] avg_ms_per_image={avg_ms:.2f}")
    print(f"[fps] avg_fps={avg_fps:.2f}")


if __name__ == "__main__":
    main()
