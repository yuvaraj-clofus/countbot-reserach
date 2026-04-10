#!/usr/bin/env python3
"""
Check DINOv2 inference match against saved training .pt files.

Supports:
    1. 6-patch vertical-strip embeddings
    2. full-image embeddings

The script prints match scores and total inference-match time for each mode.

Run commands:
    Inference match for 6-patch .pt:
        python check_dinov2_inference_match.py --image "1.jpg" --patches-pt "testing/slave/realimages/conveyor side camera view_dinov2_6_patch_strip.pt"

    Inference match for full-image .pt:
        python check_dinov2_inference_match.py --image "1.jpg" --full-image-pt "testing/slave/realimages/conveyor side camera view_dinov2_full_image.pt"

    Inference match for both modes:
        python check_dinov2_inference_match.py --image "1.jpg"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

from check_dinov2_strip_train_time import (
    default_embeddings_path,
    encode_patches,
    extract_spaced_patches,
    load_dino,
    resolve_device,
    sync_device,
)


DEFAULT_THRESHOLD = 0.90
INFER_HELP_EXAMPLES = """
Examples:
  6-patch inference match:
    python check_dinov2_inference_match.py --image "1.jpg" --patches-pt "testing/slave/realimages/conveyor side camera view_dinov2_6_patch_strip.pt"

  Full-image inference match:
    python check_dinov2_inference_match.py --image "1.jpg" --full-image-pt "testing/slave/realimages/conveyor side camera view_dinov2_full_image.pt"

  Run both modes automatically:
    python check_dinov2_inference_match.py --image "1.jpg"
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check inference match for saved DINOv2 patch/full-image .pt files.",
        epilog=INFER_HELP_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", required=True, help="Inference image path.")
    parser.add_argument(
        "--patches-pt",
        default="",
        help="Path to saved 6-patch training .pt. If omitted, the default sibling path is used when available.",
    )
    parser.add_argument(
        "--full-image-pt",
        default="",
        help="Path to saved full-image training .pt. If omitted, the default sibling path is used when available.",
    )
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, cuda, cuda:0.")
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Cosine similarity threshold for reporting a match.",
    )
    parser.add_argument("--warmup", type=int, default=0, help="Warmup runs before timed inference.")
    return parser.parse_args()


def infer_train_mode(data: dict, pt_path: Path) -> str:
    train_mode = data.get("train_mode")
    if train_mode in {"patches", "full-image"}:
        return train_mode
    if "patch_tops" in data:
        return "patches"
    if pt_path.name.endswith("_dinov2_full_image.pt") or "image_size" in data:
        return "full-image"
    raise ValueError(f"could not infer train mode from: {pt_path}")


def load_training_pt(pt_path: Path) -> dict:
    if not pt_path.is_file():
        raise FileNotFoundError(f"training pt not found: {pt_path}")
    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
    except TypeError:
        data = torch.load(pt_path, map_location="cpu")
    data["train_mode"] = infer_train_mode(data, pt_path)
    return data


def resolve_training_pt_paths(args: argparse.Namespace, image_path: Path) -> list[tuple[str, Path]]:
    pt_paths: list[tuple[str, Path]] = []
    explicit_selection = bool(args.patches_pt or args.full_image_pt)

    if args.patches_pt:
        pt_paths.append(("patches", Path(args.patches_pt)))
    elif not explicit_selection:
        patch_default = default_embeddings_path(image_path, "patches")
        if patch_default.is_file():
            pt_paths.append(("patches", patch_default))

    if args.full_image_pt:
        pt_paths.append(("full-image", Path(args.full_image_pt)))
    elif not explicit_selection:
        full_default = default_embeddings_path(image_path, "full-image")
        if full_default.is_file():
            pt_paths.append(("full-image", full_default))

    if not pt_paths:
        raise SystemExit(
            "[error] no training .pt files found. Pass --patches-pt and/or --full-image-pt."
        )
    return pt_paths


def load_image_size_from_path(image_path: str) -> tuple[int, int] | None:
    path = Path(image_path)
    if not path.is_file():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return (int(image.shape[1]), int(image.shape[0]))


def get_training_image_size(train_data: dict) -> tuple[int, int] | None:
    image_size = train_data.get("image_size")
    if image_size is not None and len(image_size) == 2:
        return (int(image_size[0]), int(image_size[1]))
    image_path = train_data.get("image_path")
    if image_path:
        return load_image_size_from_path(str(image_path))
    return None


def scale_coord(value: int, src_max: int, dst_max: int) -> int:
    if dst_max <= 0:
        return 0
    if src_max <= 0:
        return min(max(value, 0), dst_max)
    scaled = round((float(value) / float(src_max)) * float(dst_max))
    return min(max(int(scaled), 0), dst_max)


def get_inference_inputs(image, train_data: dict) -> tuple[list, dict]:
    train_mode = train_data["train_mode"]
    if train_mode == "full-image":
        return [image.copy()], {"training_region": "full_image"}

    infer_h, infer_w = image.shape[:2]
    strip_x = int(train_data["strip_x"])
    strip_width = int(train_data["strip_width"])
    patch_size = int(train_data["patch_size"])
    raw_patch_tops = [int(v) for v in train_data["patch_tops"]]
    train_image_size = get_training_image_size(train_data)

    if strip_width > infer_w or patch_size > infer_h:
        raise ValueError(
            f"inference image is too small for saved patch layout: "
            f"image={infer_w}x{infer_h}, strip_width={strip_width}, patch_size={patch_size}"
        )

    max_infer_x = infer_w - strip_width
    max_infer_top = infer_h - patch_size

    if train_image_size is not None:
        train_w, train_h = train_image_size
        max_train_x = max(1, train_w - strip_width)
        max_train_top = max(1, train_h - patch_size)
        strip_x = scale_coord(strip_x, max_train_x, max_infer_x)
        patch_tops = [scale_coord(v, max_train_top, max_infer_top) for v in raw_patch_tops]
    else:
        strip_x = min(max(strip_x, 0), max_infer_x)
        patch_tops = [min(max(v, 0), max_infer_top) for v in raw_patch_tops]

    patches = extract_spaced_patches(image, strip_x, patch_tops, strip_width, patch_size)
    for idx, patch in enumerate(patches):
        if patch.size == 0 or patch.shape[0] != patch_size or patch.shape[1] != strip_width:
            raise ValueError(
                f"empty/invalid patch at index={idx}: shape={getattr(patch, 'shape', None)} "
                f"strip_x={strip_x}, patch_top={patch_tops[idx]}, strip_width={strip_width}, patch_size={patch_size}"
            )
    return patches, {
        "strip_x": strip_x,
        "strip_width": strip_width,
        "patch_tops": patch_tops,
        "patch_size": patch_size,
        "train_image_size": train_image_size,
        "inference_image_size": (infer_w, infer_h),
    }


def compare_embeddings(
    inference_embeddings: torch.Tensor,
    trained_embeddings: torch.Tensor,
    threshold: float,
    train_mode: str,
) -> dict:
    if trained_embeddings.ndim == 1:
        trained_embeddings = trained_embeddings.unsqueeze(0)
    if inference_embeddings.ndim == 1:
        inference_embeddings = inference_embeddings.unsqueeze(0)

    if inference_embeddings.shape != trained_embeddings.shape:
        raise ValueError(
            f"embedding shape mismatch: inference={tuple(inference_embeddings.shape)} "
            f"trained={tuple(trained_embeddings.shape)}"
        )

    inference_embeddings = F.normalize(inference_embeddings, dim=1)
    trained_embeddings = F.normalize(trained_embeddings, dim=1)
    similarities = torch.sum(inference_embeddings * trained_embeddings, dim=1)
    matches = similarities >= threshold

    if train_mode == "patches":
        overall_match = bool(torch.all(matches).item())
        matched_count = int(matches.sum().item())
    else:
        overall_match = bool(matches[0].item())
        matched_count = int(matches.sum().item())

    return {
        "similarities": similarities.detach().cpu(),
        "matches": matches.detach().cpu(),
        "overall_match": overall_match,
        "matched_count": matched_count,
        "total_count": int(similarities.numel()),
        "avg_similarity": float(similarities.mean().item()),
        "min_similarity": float(similarities.min().item()),
        "max_similarity": float(similarities.max().item()),
    }


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.is_file():
        raise SystemExit(f"[error] image not found: {image_path}")
    if not 0.0 <= args.match_threshold <= 1.0:
        raise SystemExit("[error] --match-threshold must be between 0.0 and 1.0")
    if args.warmup < 0:
        raise SystemExit("[error] --warmup must be >= 0")

    overall_start = time.perf_counter()

    read_start = time.perf_counter()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    image_read_time = time.perf_counter() - read_start
    if image is None:
        raise SystemExit(f"[error] failed to read image: {image_path}")

    pt_paths = resolve_training_pt_paths(args, image_path)
    pt_records = []
    for _, pt_path in pt_paths:
        pt_records.append((pt_path, load_training_pt(pt_path)))

    model_names = {record.get("model", "dinov2_vits14") for _, record in pt_records}
    if len(model_names) != 1:
        raise SystemExit(
            f"[error] multiple model names found in .pt files: {sorted(model_names)}"
        )

    device = resolve_device(args.device)
    model_name = next(iter(model_names))
    model, model_load_time = load_dino(model_name, device)

    print(f"[info] image={image_path}")
    print(f"[info] image_size={image.shape[1]}x{image.shape[0]}")
    print(f"[info] device={device}")
    print(f"[info] model={model_name}")
    print(f"[info] match_threshold={args.match_threshold:.3f}")
    print(f"[time] image_read_ms={image_read_time * 1000.0:.2f}")
    print(f"[time] model_load_ms={model_load_time * 1000.0:.2f}")

    total_inference_match_time = 0.0
    for pt_path, train_data in pt_records:
        mode = train_data["train_mode"]
        mode_start = time.perf_counter()

        extract_start = time.perf_counter()
        inference_inputs, location_info = get_inference_inputs(image, train_data)
        extract_time = time.perf_counter() - extract_start

        for _ in range(args.warmup):
            encode_patches(model, inference_inputs, device)

        inference_embeddings, prep_time, infer_time, encode_time = encode_patches(
            model, inference_inputs, device
        )

        sync_device(device)
        compare_start = time.perf_counter()
        comparison = compare_embeddings(
            inference_embeddings=inference_embeddings,
            trained_embeddings=train_data["embeddings"].to(device),
            threshold=args.match_threshold,
            train_mode=mode,
        )
        sync_device(device)
        compare_time = time.perf_counter() - compare_start
        compare_time_ms = compare_time * 1000.0

        mode_total_time = time.perf_counter() - mode_start
        total_inference_match_time += mode_total_time

        print("")
        print(f"[{mode}] trained_pt={pt_path}")
        if mode == "full-image":
            print(f"[{mode}] training_region=full_image")
        else:
            print(f"[{mode}] strip_x={location_info['strip_x']} strip_width={location_info['strip_width']}")
            print(f"[{mode}] patch_tops={location_info['patch_tops']}")
            if location_info["train_image_size"] is not None:
                print(f"[{mode}] train_image_size={location_info['train_image_size']}")
                print(f"[{mode}] inference_image_size={location_info['inference_image_size']}")
        print(f"[{mode}] trained_embedding_shape={tuple(train_data['embeddings'].shape)}")
        print(f"[{mode}] inference_embedding_shape={tuple(inference_embeddings.shape)}")
        if mode == "patches":
            sim_text = ", ".join(f"{v:.4f}" for v in comparison["similarities"].tolist())
            print(f"[{mode}] per_patch_similarity=[{sim_text}]")
            print(
                f"[{mode}] matched_patches={comparison['matched_count']}/{comparison['total_count']}"
            )
        else:
            print(f"[{mode}] similarity={comparison['avg_similarity']:.4f}")
        print(f"[{mode}] avg_similarity={comparison['avg_similarity']:.4f}")
        print(f"[{mode}] min_similarity={comparison['min_similarity']:.4f}")
        print(f"[{mode}] match_result={comparison['overall_match']}")
        print(f"[{mode}] threshold={args.match_threshold:.3f}")
        print(f"[{mode}] extract_inputs_ms={extract_time * 1000.0:.2f}")
        print(f"[{mode}] preprocess_ms={prep_time * 1000.0:.2f}")
        print(f"[{mode}] infer_ms={infer_time * 1000.0:.2f}")
        print(f"[{mode}] encode_total_ms={encode_time * 1000.0:.2f}")
        print(f"[{mode}] compare_ms={compare_time_ms:.2f}")
        if mode == "patches":
            patch_count = max(1, comparison["total_count"])
            infer_time_ms = infer_time * 1000.0
            encode_time_ms = encode_time * 1000.0
            print(f"[{mode}] patch_match_total_ms={compare_time_ms:.2f}")
            print(f"[{mode}] avg_single_patch_infer_ms={infer_time_ms / patch_count:.2f}")
            print(f"[{mode}] avg_single_patch_encode_ms={encode_time_ms / patch_count:.2f}")
            print(f"[{mode}] avg_patch_match_ms={compare_time_ms / patch_count:.2f}")
            print(
                f"[{mode}] SINGLE PATCH INFERENCE TIME: {infer_time_ms / patch_count:.2f} ms "
                f"(average from {patch_count} patches)"
            )
            print(
                f"[{mode}] PATCH MATCH TIME: {compare_time_ms:.2f} ms total "
                f"({compare_time_ms / patch_count:.2f} ms per patch)"
            )
        else:
            print(f"[{mode}] FULL IMAGE MATCH TIME: {compare_time_ms:.2f} ms")
        print(f"[{mode}] TOTAL INFERENCE MATCH TIME: {mode_total_time * 1000.0:.2f} ms ({mode_total_time:.3f} sec)")

    overall_time = time.perf_counter() - overall_start
    print("")
    print(
        f"[result] TOTAL INFERENCE MATCH TIME ALL MODES: "
        f"{total_inference_match_time * 1000.0:.2f} ms ({total_inference_match_time:.3f} sec)"
    )
    print(
        f"[result] OVERALL END-TO-END TIME: {overall_time * 1000.0:.2f} ms ({overall_time:.3f} sec)"
    )


if __name__ == "__main__":
    main()
