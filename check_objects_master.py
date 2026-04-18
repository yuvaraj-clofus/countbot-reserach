#!/usr/bin/env python3
"""
Compare one master object image against every other image in objects/ using DINOv2.

Run:
    python3 check_objects_master.py
    python3 check_objects_master.py --master objects/object_0003_track_0030_frame_0309.jpg
"""

import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# -------- CONFIG --------
MODEL_NAME = "dinov2_vits14"
OBJECTS_DIR = Path("objects")
MATCH_THRESHOLD = 0.75
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEVICE = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def sync_device():
    """Synchronize accelerator work so timing is accurate."""
    if DEVICE.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        return

    if DEVICE == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def preprocess_image(image_bgr):
    """Resize and normalize one BGR image for DINOv2."""
    image = cv2.resize(image_bgr, (224, 224), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = (image - MEAN) / STD
    image = np.transpose(image, (2, 0, 1))
    tensor = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
    return tensor.to(DEVICE)


@torch.no_grad()
def get_embedding(model, image_path):
    """Load one image and return its normalized DINO embedding."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read image: {image_path}")

    tensor = preprocess_image(image)
    embedding = model(tensor)
    return F.normalize(embedding, dim=-1).squeeze(0)


def cosine_score(master_embedding, other_embedding):
    """Return cosine similarity score as float."""
    score = F.cosine_similarity(
        master_embedding.unsqueeze(0),
        other_embedding.unsqueeze(0),
        dim=-1,
    )
    return float(score.item())


def collect_object_images(folder):
    """Return sorted image paths from the objects folder."""
    if not folder.is_dir():
        raise FileNotFoundError(f"objects folder not found: {folder.resolve()}")

    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare one master object image against all other images in objects/."
    )
    parser.add_argument(
        "--master",
        type=str,
        default=None,
        help="Optional master image path. If omitted, the first image in objects/ is used.",
    )
    return parser.parse_args()


def resolve_master_image(master_arg, object_images):
    """Return the chosen master image path."""
    if master_arg:
        user_path = Path(master_arg).expanduser()
        candidates = [user_path, OBJECTS_DIR / user_path]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise SystemExit(
            f"[ERROR] master image not found: {master_arg}\n"
            f"Pass a valid path or a filename from {OBJECTS_DIR.resolve()}"
        )

    if not object_images:
        raise SystemExit(f"[ERROR] no images found in: {OBJECTS_DIR.resolve()}")
    return object_images[0].resolve()


def main():
    args = parse_args()
    object_images = collect_object_images(OBJECTS_DIR)
    if not object_images:
        raise SystemExit(f"[ERROR] no images found in: {OBJECTS_DIR.resolve()}")
    master_image = resolve_master_image(args.master, object_images)

    print(f"[DINO] Loading {MODEL_NAME} on {DEVICE}...")
    model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME, verbose=False)
    model.eval().to(DEVICE)
    print("[DINO] Ready.")
    print(f"[MASTER] {master_image.name}")
    print(f"[THRESHOLD] {MATCH_THRESHOLD:.2f}")
    print("-" * 72)

    sync_device()
    master_start = time.perf_counter()
    master_embedding = get_embedding(model, master_image)
    sync_device()
    master_time_sec = time.perf_counter() - master_start
    print(f"[MASTER EMBEDDING TIME] {master_time_sec:.4f} sec")
    print("-" * 72)

    checked_count = 0
    matched_count = 0
    total_compare_time_sec = 0.0

    for image_path in object_images:
        if image_path.resolve() == master_image.resolve():
            continue

        sync_device()
        compare_start = time.perf_counter()
        other_embedding = get_embedding(model, image_path)
        score = cosine_score(master_embedding, other_embedding)
        sync_device()
        compare_time_sec = time.perf_counter() - compare_start
        is_match = score >= MATCH_THRESHOLD

        checked_count += 1
        matched_count += int(is_match)
        total_compare_time_sec += compare_time_sec

        print(
            f"{image_path.name:<40} "
            f"score={score:.4f}  "
            f"match={is_match}  "
            f"time_sec={compare_time_sec:.4f}"
        )

    print("-" * 72)
    avg_compare_time_sec = (
        total_compare_time_sec / checked_count if checked_count > 0 else 0.0
    )
    print(
        f"checked={checked_count}  "
        f"matched={matched_count}  "
        f"not_matched={checked_count - matched_count}  "
        f"total_time_sec={total_compare_time_sec:.4f}  "
        f"avg_time_sec={avg_compare_time_sec:.4f}"
    )


if __name__ == "__main__":
    main()
