#!/usr/bin/env python3
"""
Benchmark DINOv2 processing time for one vertical strip split into 6 patches.

This script takes a 25 px wide vertical line and encodes 6 stacked patches of
25 x 25 pixels each.

"Train" here means building reference embeddings from the strip patches.
DINOv2 itself is not fine-tuned; it only encodes the patches.

Run commands:
    Train 6-patch mode:
        python check_dinov2_strip_train_time.py --image "1.jpg" --train-mode patches --show-preview

    Train full-image mode:
        python check_dinov2_strip_train_time.py --image "1.jpg" --train-mode full-image
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


DEFAULT_MODEL = "dinov2_vits14"
DEFAULT_TRAIN_MODE = "patches"
DEFAULT_PATCH_SIZE = 25
DEFAULT_NUM_PATCHES = 6
DEFAULT_STRIP_WIDTH = 25
DEFAULT_STRIP_X_FRACTION = 0.85
DEFAULT_PREVIEW_MS = 1500
_DINO_INPUT_SIZE = (224, 224)
_DINO_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_DINO_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
TRAIN_HELP_EXAMPLES = """
Examples:
  Train 6-patch mode:
    python check_dinov2_strip_train_time.py --image "1.jpg" --train-mode patches --show-preview

  Train full-image mode:
    python check_dinov2_strip_train_time.py --image "1.jpg" --train-mode full-image
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure DINOv2 time for either 6 vertical patches or the full image.",
        epilog=TRAIN_HELP_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="DINOv2 model name.")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, cuda:0.")
    parser.add_argument(
        "--train-mode",
        choices=["patches", "full-image"],
        default=DEFAULT_TRAIN_MODE,
        help="Use the existing 6-patch strip flow or encode the whole image as one training sample.",
    )
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE, help="Patch size in pixels.")
    parser.add_argument(
        "--num-patches",
        type=int,
        default=DEFAULT_NUM_PATCHES,
        help="Number of vertical patches to train. Default is 6.",
    )
    parser.add_argument("--strip-width", type=int, default=DEFAULT_STRIP_WIDTH, help="Strip width in pixels.")
    parser.add_argument("--strip-x", type=int, default=None, help="Left X coordinate of strip.")
    parser.add_argument(
        "--strip-x-fraction",
        type=float,
        default=DEFAULT_STRIP_X_FRACTION,
        help="Used when --strip-x is not provided.",
    )
    parser.add_argument(
        "--strip-top",
        type=int,
        default=None,
        help="Optional top Y for patch 1. Default spreads the 6 patches from top to bottom.",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs before timing.")
    parser.add_argument("--repeats", type=int, default=1, help="Timed runs.")
    parser.add_argument(
        "--save-embeddings",
        default="",
        help="Optional .pt output path for the final patch embeddings. Default is based on the image name.",
    )
    parser.add_argument(
        "--save-preview",
        default="",
        help="Optional image path to save a preview of the selected strip and patch guides.",
    )
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Show the selected training strip and its 6 patches in a window.",
    )
    parser.add_argument(
        "--preview-ms",
        type=int,
        default=DEFAULT_PREVIEW_MS,
        help="How long to show the preview window before it closes automatically.",
    )
    return parser.parse_args()


def sync_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def dino_tfm(rgb_patch: np.ndarray) -> torch.Tensor:
    pil_img = Image.fromarray(rgb_patch)
    if hasattr(Image, "Resampling"):
        pil_img = pil_img.resize(_DINO_INPUT_SIZE, Image.Resampling.BILINEAR)
    else:
        pil_img = pil_img.resize(_DINO_INPUT_SIZE, Image.BILINEAR)

    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
    return (tensor - _DINO_MEAN) / _DINO_STD


def load_dino(model_name: str, device: torch.device) -> tuple[torch.nn.Module, float]:
    sync_device(device)
    start = time.perf_counter()
    model = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)
    model.eval().to(device)
    sync_device(device)
    return model, time.perf_counter() - start


def choose_strip_region(
    image: np.ndarray,
    strip_width: int,
    strip_x: int | None,
    strip_x_fraction: float,
) -> int:
    img_h, img_w = image.shape[:2]
    if strip_width > img_w:
        raise ValueError(f"strip_width={strip_width} exceeds image width={img_w}")

    if strip_x is None:
        max_x = img_w - strip_width
        strip_x = int(round(max_x * strip_x_fraction))
    strip_x = min(max(strip_x, 0), img_w - strip_width)
    return strip_x


def choose_patch_tops(
    image: np.ndarray,
    patch_size: int,
    num_patches: int,
    strip_top: int | None,
) -> list[int]:
    img_h = image.shape[0]
    if patch_size > img_h:
        raise ValueError(
            f"patch_size={patch_size} exceeds image height={img_h}. Reduce --patch-size."
        )

    max_top = img_h - patch_size
    if strip_top is not None:
        start_top = min(max(strip_top, 0), max_top)
        end_top = max_top
        if num_patches == 1:
            return [start_top]
        tops = np.linspace(start_top, end_top, num_patches)
    else:
        if num_patches == 1:
            return [max_top // 2]
        tops = np.linspace(0, max_top, num_patches)

    patch_tops = [int(round(v)) for v in tops]
    patch_tops = [min(max(v, 0), max_top) for v in patch_tops]
    return patch_tops


def extract_spaced_patches(
    image: np.ndarray,
    x: int,
    patch_tops: list[int],
    width: int,
    patch_size: int,
) -> list[np.ndarray]:
    patches = []
    for top in patch_tops:
        patches.append(image[top : top + patch_size, x : x + width].copy())
    return patches


def default_embeddings_path(image_path: Path, train_mode: str) -> Path:
    if train_mode == "full-image":
        return image_path.with_name(f"{image_path.stem}_dinov2_full_image.pt")
    return image_path.with_name(f"{image_path.stem}_dinov2_6_patch_strip.pt")


@torch.no_grad()
def encode_patches(
    model: torch.nn.Module,
    patches_bgr: list[np.ndarray],
    device: torch.device,
) -> tuple[torch.Tensor, float, float, float]:
    sync_device(device)
    prep_start = time.perf_counter()
    tensors = [dino_tfm(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)) for patch in patches_bgr]
    batch = torch.stack(tensors).to(device)
    sync_device(device)
    prep_time = time.perf_counter() - prep_start

    sync_device(device)
    infer_start = time.perf_counter()
    feats = model(batch)
    feats = F.normalize(feats, dim=1)
    sync_device(device)
    infer_time = time.perf_counter() - infer_start

    total_time = prep_time + infer_time
    return feats, prep_time, infer_time, total_time


def build_patch_preview(
    image: np.ndarray,
    x: int,
    patch_tops: list[int],
    strip_width: int,
    patch_size: int,
    preview_max_height: int = 900,
) -> np.ndarray:
    preview = image.copy()
    num_patches = len(patch_tops)
    for idx, top in enumerate(patch_tops):
        cv2.rectangle(preview, (x, top), (x + strip_width, top + patch_size), (0, 255, 255), 2)
        cv2.putText(
            preview,
            str(idx + 1),
            (x + strip_width + 6, top + patch_size - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    strip_stack = []
    for idx, top in enumerate(patch_tops):
        patch = image[top : top + patch_size, x : x + strip_width].copy()
        zoom_w = max(200, strip_width * 8)
        zoom_h = patch_size * 4
        patch_zoom = cv2.resize(patch, (zoom_w, zoom_h), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(patch_zoom, (0, 0), (zoom_w - 1, zoom_h - 1), (0, 255, 255), 2)
        cv2.putText(
            patch_zoom,
            f"Patch {idx + 1}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        strip_stack.append(patch_zoom)

    strip = np.vstack(strip_stack)

    canvas_h = max(preview.shape[0], strip.shape[0])
    canvas_w = preview.shape[1] + strip.shape[1] + 20
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[: preview.shape[0], : preview.shape[1]] = preview
    x_off = preview.shape[1] + 20
    canvas[: strip.shape[0], x_off : x_off + strip.shape[1]] = strip

    cv2.putText(
        canvas,
        "Full image with trained strip",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Top-to-bottom trained patches",
        (x_off, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if canvas.shape[0] > preview_max_height:
        scale = preview_max_height / canvas.shape[0]
        canvas = cv2.resize(
            canvas,
            (int(canvas.shape[1] * scale), preview_max_height),
            interpolation=cv2.INTER_AREA,
        )
    return canvas


def build_full_image_preview(image: np.ndarray, preview_max_height: int = 900) -> np.ndarray:
    preview = image.copy()
    h, w = preview.shape[:2]
    cv2.rectangle(preview, (0, 0), (w - 1, h - 1), (0, 255, 255), 3)
    cv2.putText(
        preview,
        "Full image used for DINOv2 training",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if preview.shape[0] > preview_max_height:
        scale = preview_max_height / preview.shape[0]
        preview = cv2.resize(
            preview,
            (int(preview.shape[1] * scale), preview_max_height),
            interpolation=cv2.INTER_AREA,
        )
    return preview


def save_preview(preview: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), preview)


def show_preview(preview: np.ndarray, preview_ms: int) -> None:
    win_name = "DINOv2 trained strip preview"
    cv2.imshow(win_name, preview)
    cv2.waitKey(max(1, preview_ms))
    cv2.destroyWindow(win_name)


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.is_file():
        raise SystemExit(f"[error] image not found: {image_path}")
    if args.patch_size < 1 or args.num_patches < 1 or args.strip_width < 1:
        raise SystemExit("[error] patch-size, num-patches, and strip-width must be >= 1")
    if args.warmup < 0 or args.repeats < 1:
        raise SystemExit("[error] warmup must be >= 0 and repeats must be >= 1")

    total_start = time.perf_counter()

    read_start = time.perf_counter()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    read_time = time.perf_counter() - read_start
    if image is None:
        raise SystemExit(f"[error] failed to read image: {image_path}")

    select_start = time.perf_counter()
    strip_x = None
    patch_tops: list[int] = []
    num_patches = 1
    training_images: list[np.ndarray]
    if args.train_mode == "full-image":
        training_images = [image.copy()]
    else:
        strip_x = choose_strip_region(
            image=image,
            strip_width=args.strip_width,
            strip_x=args.strip_x,
            strip_x_fraction=args.strip_x_fraction,
        )
        patch_tops = choose_patch_tops(image, args.patch_size, args.num_patches, args.strip_top)
        training_images = extract_spaced_patches(image, strip_x, patch_tops, args.strip_width, args.patch_size)
        num_patches = len(training_images)
    select_time = time.perf_counter() - select_start

    device = resolve_device(args.device)
    model, model_load_time = load_dino(args.model_name, device)

    for _ in range(args.warmup):
        encode_patches(model, training_images, device)

    prep_times = []
    infer_times = []
    total_times = []
    last_embeddings = None
    for _ in range(args.repeats):
        last_embeddings, prep_time, infer_time, total_time = encode_patches(model, training_images, device)
        prep_times.append(prep_time)
        infer_times.append(infer_time)
        total_times.append(total_time)

    grand_total = time.perf_counter() - total_start
    total_training_time = sum(total_times)
    total_training_time_ms = total_training_time * 1000.0

    print(f"[info] image={image_path}")
    print(f"[info] image_size={image.shape[1]}x{image.shape[0]}")
    print(f"[info] device={device}")
    print(f"[info] model={args.model_name}")
    print(f"[info] train_mode={args.train_mode}")
    if args.train_mode == "full-image":
        print("[info] training_region=full_image")
    else:
        print(f"[info] strip_x={strip_x} strip_width={args.strip_width}")
        print(f"[info] patch_tops={patch_tops}")
        print(f"[info] patches={num_patches} patch_size={args.patch_size}x{args.patch_size}")
    print(f"[info] embedding_shape={tuple(last_embeddings.shape)}")
    print(f"[time] image_read_ms={read_time * 1000.0:.2f}")
    print(f"[time] strip_select_and_split_ms={select_time * 1000.0:.2f}")
    print(f"[time] model_load_ms={model_load_time * 1000.0:.2f}")
    print(f"[time] warmup_runs={args.warmup}")
    print(f"[time] timed_runs={args.repeats}")
    print(f"[time] avg_preprocess_ms={(sum(prep_times) / len(prep_times)) * 1000.0:.2f}")
    print(f"[time] avg_infer_ms={(sum(infer_times) / len(infer_times)) * 1000.0:.2f}")
    print(f"[time] avg_total_encode_ms={(sum(total_times) / len(total_times)) * 1000.0:.2f}")
    print(f"[time] total_timed_encode_ms={total_training_time_ms:.2f}")
    print(f"[time] trained_time_ms={total_training_time_ms:.2f}")
    print(f"[result] TOTAL TRAINING TIME: {total_training_time_ms:.2f} ms ({total_training_time:.3f} sec)")
    print(f"[time] grand_total_ms={grand_total * 1000.0:.2f}")

    preview = None
    if args.save_preview or args.show_preview:
        if args.train_mode == "full-image":
            preview = build_full_image_preview(image)
        else:
            preview = build_patch_preview(
                image=image,
                x=strip_x,
                patch_tops=patch_tops,
                strip_width=args.strip_width,
                patch_size=args.patch_size,
            )

    save_path = Path(args.save_embeddings) if args.save_embeddings else default_embeddings_path(image_path, args.train_mode)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "image_path": str(image_path),
        "train_mode": args.train_mode,
        "image_size": (image.shape[1], image.shape[0]),
        "device": str(device),
        "model": args.model_name,
        "embeddings": last_embeddings.cpu(),
    }
    if args.train_mode != "full-image":
        save_data.update(
            {
                "strip_x": strip_x,
                "strip_width": args.strip_width,
                "patch_tops": patch_tops,
                "patch_size": args.patch_size,
                "num_patches": num_patches,
            }
        )
    torch.save(save_data, save_path)
    print(f"[save] embeddings={save_path}")

    if args.save_preview:
        preview_path = Path(args.save_preview)
        save_preview(preview, preview_path)
        print(f"[save] preview={preview_path}")

    if args.show_preview:
        print(f"[preview] closing automatically in {args.preview_ms} ms")
        show_preview(preview, args.preview_ms)


if __name__ == "__main__":
    main()
