#!/usr/bin/env python3
"""
Simple conveyor object detection using OpenCV MOG2 background subtraction.

What it does:
  - reads a video file or webcam
  - uses MOG2 to separate moving products from the conveyor background
  - cleans the mask with morphology
  - merges nearby blobs into a single product box
  - shows live preview with FPS and detection count
  - optionally saves the annotated output

Example:
  python3 detect.py
  python3 detect.py --source 0 --no-save
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = str(SCRIPT_DIR.parent / "video" / "bearing_wheel.mp4")
DEFAULT_OUTPUT = str(SCRIPT_DIR / "runs" / "mog2_output.mp4")


Detection = dict[str, float | int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect conveyor objects using MOG2 background subtraction.")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Video path or webcam index like 0")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output video path. Use empty string to disable saving.")
    parser.add_argument("--no-save", action="store_true", help="Disable output video writing")
    parser.add_argument("--no-show", action="store_true", help="Disable the live preview window")
    parser.add_argument("--history", type=int, default=300, help="MOG2 history length")
    parser.add_argument("--var-threshold", type=float, default=32.0, help="MOG2 variance threshold")
    parser.add_argument("--learning-rate", type=float, default=-1.0, help="Background update rate. -1 lets OpenCV choose")
    parser.add_argument("--warmup", type=int, default=20, help="Skip detections for the first N frames")
    parser.add_argument("--min-area", type=int, default=1400, help="Minimum contour area to keep")
    parser.add_argument("--max-area-ratio", type=float, default=0.20, help="Reject detections larger than this fraction of the frame")
    parser.add_argument("--min-width", type=int, default=28, help="Minimum raw detection width")
    parser.add_argument("--min-height", type=int, default=28, help="Minimum raw detection height")
    parser.add_argument("--min-fill-ratio", type=float, default=0.18, help="Minimum foreground fill inside a raw box")
    parser.add_argument("--min-solidity", type=float, default=0.30, help="Minimum contour solidity to reject loose shadow blobs")
    parser.add_argument("--min-diff", type=float, default=18.0, help="Minimum mean gray difference from learned background")
    parser.add_argument("--min-score", type=float, default=0.72, help="Only show detections with score at or above this value")
    parser.add_argument("--min-persist", type=int, default=2, help="Show a detection only after it appears for this many frames")
    parser.add_argument("--persist-iou", type=float, default=0.20, help="IoU needed to treat a detection as the same object across frames")
    parser.add_argument("--max-aspect-ratio", type=float, default=4.5, help="Reject very long thin blobs caused by glare or streaks")
    parser.add_argument("--edge-margin", type=int, default=8, help="Reject detections too close to the frame border")
    parser.add_argument("--roi-left", type=float, default=0.0, help="Ignore detections left of this frame fraction")
    parser.add_argument("--roi-right", type=float, default=1.0, help="Ignore detections right of this frame fraction")
    parser.add_argument("--roi-top", type=float, default=0.0, help="Ignore detections above this frame fraction")
    parser.add_argument("--roi-bottom", type=float, default=1.0, help="Ignore detections below this frame fraction")
    parser.add_argument("--box-pad", type=int, default=18, help="Padding added around each detected box")
    parser.add_argument("--merge-gap", type=int, default=35, help="Merge boxes if edges are this close")
    parser.add_argument("--mask-blur", type=int, default=5, help="Median blur size for removing dust noise. 0 disables it")
    parser.add_argument("--max-width", type=int, default=960, help="Preview max width. 0 keeps original size")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame limit for quick tests")
    return parser.parse_args()


def resolve_source(source: str) -> str | int:
    source = source.strip()
    if source.isdigit():
        return int(source)
    return source


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def clamp_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[int, int, int, int]:
    return max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)


def boxes_should_merge(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], gap: int) -> bool:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    if inter_w > 0 and inter_h > 0:
        return True

    gap_x = max(0, max(ax1, bx1) - min(ax2, bx2))
    gap_y = max(0, max(ay1, by1) - min(ay2, by2))
    overlap_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    overlap_h = max(0, min(ay2, by2) - max(ay1, by1))
    min_w = max(1, min(ax2 - ax1, bx2 - bx1))
    min_h = max(1, min(ay2 - ay1, by2 - by1))

    if gap_x <= gap and (overlap_h / min_h) >= 0.25:
        return True
    if gap_y <= gap and (overlap_w / min_w) >= 0.25:
        return True
    return False


def merge_boxes(boxes: list[tuple[int, int, int, int]], gap: int) -> list[tuple[int, int, int, int]]:
    if len(boxes) <= 1:
        return boxes

    merged = boxes[:]
    changed = True
    while changed:
        changed = False
        used = [False] * len(merged)
        next_boxes: list[tuple[int, int, int, int]] = []

        for i, box in enumerate(merged):
            if used[i]:
                continue

            used[i] = True
            mx1, my1, mx2, my2 = box

            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                if not boxes_should_merge((mx1, my1, mx2, my2), merged[j], gap):
                    continue

                bx1, by1, bx2, by2 = merged[j]
                mx1 = min(mx1, bx1)
                my1 = min(my1, by1)
                mx2 = max(mx2, bx2)
                my2 = max(my2, by2)
                used[j] = True
                changed = True

            next_boxes.append((mx1, my1, mx2, my2))

        merged = next_boxes

    return merged


def contour_solidity(contour: np.ndarray, area: float) -> float:
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 1.0:
        return 0.0
    return float(area / hull_area)


def box_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / max(1, area_a + area_b - inter)


def compute_detection_score(
    fill_ratio: float,
    min_fill_ratio: float,
    solidity: float,
    min_solidity: float,
    mean_diff: float,
    min_diff: float,
) -> float:
    fill_score = min(1.0, fill_ratio / max(min_fill_ratio, 1e-6))
    solidity_score = min(1.0, solidity / max(min_solidity, 1e-6))
    diff_score = min(1.0, mean_diff / max(min_diff, 1e-6))
    return max(0.0, min(1.0, (fill_score + solidity_score + diff_score) / 3.0))


def detect_boxes(
    frame: np.ndarray,
    subtractor: cv2.BackgroundSubtractor,
    learning_rate: float,
    min_area: int,
    max_area_ratio: float,
    min_width: int,
    min_height: int,
    min_fill_ratio: float,
    min_solidity: float,
    min_diff: float,
    max_aspect_ratio: float,
    edge_margin: int,
    roi_bounds: tuple[int, int, int, int],
    box_pad: int,
    merge_gap: int,
    mask_blur: int,
    kernels: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, list[Detection]]:
    height, width = frame.shape[:2]
    kernel_open, kernel_close, kernel_dilate = kernels
    roi_left, roi_top, roi_right, roi_bottom = roi_bounds

    fg_mask = subtractor.apply(frame, learningRate=learning_rate)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    if mask_blur >= 3:
        blur_size = mask_blur if mask_blur % 2 == 1 else mask_blur + 1
        fg_mask = cv2.medianBlur(fg_mask, blur_size)
    roi_mask = np.zeros_like(fg_mask)
    roi_mask[roi_top:roi_bottom, roi_left:roi_right] = 255
    fg_mask = cv2.bitwise_and(fg_mask, roi_mask)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)
    fg_mask = cv2.dilate(fg_mask, kernel_dilate, iterations=1)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = max_area_ratio * width * height
    bg_frame = subtractor.getBackgroundImage()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY) if bg_frame is not None else None

    detections: list[Detection] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0 or w < min_width or h < min_height:
            continue
        if x <= edge_margin or y <= edge_margin or (x + w) >= (width - edge_margin) or (y + h) >= (height - edge_margin):
            continue
        aspect_ratio = max(w, h) / max(1, min(w, h))
        if aspect_ratio > max_aspect_ratio:
            continue

        box_area = max(1, w * h)
        fill_ratio = area / box_area
        if fill_ratio < min_fill_ratio:
            continue

        solidity = contour_solidity(contour, area)
        if solidity < min_solidity:
            continue

        mean_diff = 255.0
        if bg_gray is not None:
            roi_mask = fg_mask[y:y + h, x:x + w] > 0
            if np.any(roi_mask):
                diff = cv2.absdiff(gray[y:y + h, x:x + w], bg_gray[y:y + h, x:x + w])
                mean_diff = float(diff[roi_mask].mean())
                if mean_diff < min_diff:
                    continue

        x1, y1, x2, y2 = clamp_box(x - box_pad, y - box_pad, x + w + box_pad, y + h + box_pad, width, height)
        score = compute_detection_score(fill_ratio, min_fill_ratio, solidity, min_solidity, mean_diff, min_diff)
        detections.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "score": score,
                "fill": fill_ratio,
                "solidity": solidity,
                "diff": mean_diff,
            }
        )

    merged_boxes = merge_boxes(
        [(int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])) for d in detections],
        merge_gap,
    )

    merged_detections: list[Detection] = []
    for x1, y1, x2, y2 in merged_boxes:
        group = [
            d for d in detections
            if not (int(d["x2"]) < x1 or int(d["x1"]) > x2 or int(d["y2"]) < y1 or int(d["y1"]) > y2)
        ]
        if group:
            merged_detections.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "score": max(float(d["score"]) for d in group),
                    "fill": max(float(d["fill"]) for d in group),
                    "solidity": max(float(d["solidity"]) for d in group),
                    "diff": max(float(d["diff"]) for d in group),
                }
            )
        else:
            merged_detections.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": 0.0, "fill": 0.0, "solidity": 0.0, "diff": 0.0})

    return fg_mask, merged_detections


def resize_for_display(frame: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0 or frame.shape[1] <= max_width:
        return frame
    scale = max_width / frame.shape[1]
    height = max(1, int(frame.shape[0] * scale))
    return cv2.resize(frame, (max_width, height), interpolation=cv2.INTER_AREA)


def compute_roi_bounds(width: int, height: int, args: argparse.Namespace) -> tuple[int, int, int, int]:
    left = int(max(0.0, min(1.0, args.roi_left)) * width)
    right = int(max(0.0, min(1.0, args.roi_right)) * width)
    top = int(max(0.0, min(1.0, args.roi_top)) * height)
    bottom = int(max(0.0, min(1.0, args.roi_bottom)) * height)
    right = max(left + 1, min(width, right))
    bottom = max(top + 1, min(height, bottom))
    return left, top, right, bottom


def update_persistent_detections(
    detections: list[Detection],
    previous: list[Detection],
    persist_iou: float,
) -> list[Detection]:
    updated: list[Detection] = []
    used_prev = [False] * len(previous)

    for det in detections:
        det_box = (int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"]))
        best_idx = -1
        best_iou = 0.0

        for idx, prev in enumerate(previous):
            if used_prev[idx]:
                continue
            prev_box = (int(prev["x1"]), int(prev["y1"]), int(prev["x2"]), int(prev["y2"]))
            iou = box_iou(det_box, prev_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        det_copy = dict(det)
        if best_idx >= 0 and best_iou >= persist_iou:
            det_copy["streak"] = int(previous[best_idx].get("streak", 1)) + 1
            used_prev[best_idx] = True
        else:
            det_copy["streak"] = 1
        updated.append(det_copy)

    return updated


def main() -> None:
    args = parse_args()
    source = resolve_source(args.source)
    save_output = bool(args.output) and not args.no_save

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[error] cannot open source: {source}")
        raise SystemExit(1)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    writer = None
    if save_output:
        output_path = Path(args.output)
        ensure_parent(output_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, src_fps, (frame_width, frame_height))

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=args.history,
        varThreshold=args.var_threshold,
        detectShadows=False,
    )
    kernels = (
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )
    roi_bounds = compute_roi_bounds(frame_width, frame_height, args)

    print(f"[init] source={source}")
    print(f"[init] output={args.output if save_output else 'disabled'}")
    print(f"[init] size={frame_width}x{frame_height} fps={src_fps:.2f} frames={total_frames or 'unknown'}")
    print(
        f"[init] history={args.history} var_threshold={args.var_threshold} "
        f"min_area={args.min_area} fill={args.min_fill_ratio} diff={args.min_diff} "
        f"score={args.min_score} persist={args.min_persist} box_pad={args.box_pad} merge_gap={args.merge_gap}"
    )
    print(f"[init] roi={roi_bounds[0]}:{roi_bounds[2]} x {roi_bounds[1]}:{roi_bounds[3]}")
    if not args.no_show:
        print("[run] press Q to quit")

    frame_idx = 0
    prev_time = time.perf_counter()
    previous_detections: list[Detection] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            if args.max_frames and frame_idx > args.max_frames:
                break

            if frame_idx <= args.warmup:
                subtractor.apply(frame, learningRate=args.learning_rate)
                detections: list[Detection] = []
                fg_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            else:
                fg_mask, detections = detect_boxes(
                    frame,
                    subtractor,
                    args.learning_rate,
                    args.min_area,
                    args.max_area_ratio,
                    args.min_width,
                    args.min_height,
                    args.min_fill_ratio,
                    args.min_solidity,
                    args.min_diff,
                    args.max_aspect_ratio,
                    args.edge_margin,
                    roi_bounds,
                    args.box_pad,
                    args.merge_gap,
                    args.mask_blur,
                    kernels,
                )
                detections = [d for d in detections if float(d["score"]) >= args.min_score]
                detections = update_persistent_detections(detections, previous_detections, args.persist_iou)
                previous_detections = detections

            if frame_idx <= args.warmup:
                previous_detections = []

            now = time.perf_counter()
            fps = 1.0 / max(1e-6, now - prev_time)
            prev_time = now

            annotated = frame.copy()
            roi_left, roi_top, roi_right, roi_bottom = roi_bounds
            cv2.rectangle(annotated, (roi_left, roi_top), (roi_right, roi_bottom), (255, 140, 0), 1)
            shown = 0
            for det in detections:
                if int(det.get("streak", 1)) < args.min_persist:
                    continue
                x1 = int(det["x1"])
                y1 = int(det["y1"])
                x2 = int(det["x2"])
                y2 = int(det["y2"])
                score = float(det["score"])
                shown += 1
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(
                    annotated,
                    f"object {score:.2f}",
                    (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 220, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                annotated,
                f"Frame {frame_idx}/{total_frames or '?'}  FPS={fps:.1f}  Det={shown}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if frame_idx <= args.warmup:
                cv2.putText(
                    annotated,
                    f"warming background model: {frame_idx}/{args.warmup}",
                    (12, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 180, 255),
                    2,
                    cv2.LINE_AA,
                )

            if writer is not None:
                writer.write(annotated)

            if not args.no_show:
                preview = resize_for_display(annotated, args.max_width)
                mask_small = resize_for_display(cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR), args.max_width)
                stacked = np.hstack([preview, mask_small]) if preview.shape[:2] == mask_small.shape[:2] else preview
                cv2.imshow("MOG2 Conveyor Detection", stacked)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    print("[run] stopped by user")
                    break

            if frame_idx % 30 == 0:
                print(f"Frame {frame_idx}/{total_frames or '?'}  FPS={fps:.1f}  Det={shown}")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    print(f"[done] processed {frame_idx} frames")


if __name__ == "__main__":
    main()
