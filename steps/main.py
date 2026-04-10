#!/usr/bin/env python3
"""Simple main loop with global frame thread."""

from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import stage1
import stage2


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = str(SCRIPT_DIR / "test.mp4")
DEFAULT_REFS = str(SCRIPT_DIR / "master-models" / "1bearing.pt")
DISPLAY_MAX_WIDTH = 960
DISPLAY_MAX_HEIGHT = 540

STAGE1_METHOD = "fastsam"
STAGE2_METHOD = "dinov2"

MOG2_BINARY_THRESHOLD = 100
MOG2_MIN_AREA = 80
MOG2_MIN_WIDTH = 20
MOG2_MIN_HEIGHT = 20
MOG2_BBOX_PADDING = 0
MOG2_INSIDE_MARGIN = 4
MOG2_MERGE_GAP = 35
MOG2_LINE_FRACTION = 0.5
MOG2_CENTER_MATCH_DISTANCE = 120

FASTSAM_MODEL_PATH = str(SCRIPT_DIR / "master-models" / "weights" / "FastSAM-s.engine")
FASTSAM_CONF_THRESHOLD = 0.40
FASTSAM_NMS_IOU = 0.45
FASTSAM_IMGSZ = 640
FASTSAM_MOTION_DIFF_THRESH = 30
FASTSAM_MIN_BOX_AREA = 500
FASTSAM_MAX_BOX_AREA_RATIO = 0.50
FASTSAM_MAX_BOX_WIDTH_RATIO = 0.70
FASTSAM_MAX_BOX_HEIGHT_RATIO = 0.70
FASTSAM_DARK_ROI_THRESH = 40
FASTSAM_MOTION_OVERLAP_THRESH = 0.20
FASTSAM_MERGE_BOX_IOU_THRESH = 0.10
FASTSAM_MERGE_BOX_GAP = 20
FASTSAM_MERGE_AXIS_OVERLAP = 0.35
STAGE1_LINE_FRACTION = 0.5
STAGE1_CENTER_MATCH_DISTANCE = 120

GLOBAL_FRAME = None
GLOBAL_FRAME_ID = -1
GLOBAL_RESULT = None
GLOBAL_RUNNING = True
CAPTURE_FINISHED = False
FRAME_LOCK = threading.Lock()
FRAME_QUEUE = queue.Queue(maxsize=8)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple stage1 -> stage2 pipeline")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="video path or webcam index like 0")
    parser.add_argument("--refs", default=DEFAULT_REFS, help="reference embedding .pt file")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--model-name", default="dinov2_vits14", help="stage2 model name")
    parser.add_argument("--threshold", type=float, default=0.78, help="match threshold")
    parser.add_argument("--stage1-method", default=STAGE1_METHOD, help="active stage1 method")
    parser.add_argument("--stage2-method", default=STAGE2_METHOD, help="active stage2 method")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True, help="show live window")
    parser.add_argument("--max-frames", type=int, default=0, help="optional processed frame limit")
    return parser.parse_args()


def resolve_source(source):
    value = str(source).strip()
    if value.isdigit():
        return int(value)
    return value


def capture_loop(cap, is_camera, source_fps):
    global GLOBAL_FRAME, GLOBAL_FRAME_ID, GLOBAL_RUNNING, CAPTURE_FINISHED

    while GLOBAL_RUNNING:
        ok, frame = cap.read()
        if not ok:
            CAPTURE_FINISHED = True
            break

        with FRAME_LOCK:
            GLOBAL_FRAME = frame
            GLOBAL_FRAME_ID += 1
            frame_id = GLOBAL_FRAME_ID

        if is_camera:
            try:
                while True:
                    FRAME_QUEUE.get_nowait()
            except queue.Empty:
                pass

            try:
                FRAME_QUEUE.put_nowait((frame_id, frame))
            except queue.Full:
                pass
            time.sleep(0.001)
            continue

        while GLOBAL_RUNNING:
            try:
                FRAME_QUEUE.put((frame_id, frame), timeout=0.05)
                break
            except queue.Full:
                continue


def get_next_frame(last_seen_id):
    while True:
        try:
            frame_id, frame = FRAME_QUEUE.get(timeout=0.05)
        except queue.Empty:
            if CAPTURE_FINISHED:
                return None, last_seen_id
            return None, last_seen_id

        if frame_id == last_seen_id:
            continue
        return frame.copy(), frame_id


def draw_stage1(frame, stage1_output):
    bboxes = stage1_output.get("bboxes") or []
    method = stage1_output.get("method", "-")
    line_x = stage1_output.get("line_x")

    cv2.putText(frame, f"Stage1={method}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    if line_x is not None:
        x = int(line_x)
        cv2.line(frame, (x, 0), (x, frame.shape[0] - 1), (255, 140, 0), 2)

    for bbox in bboxes:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 4, (0, 165, 255), -1)


def draw_stage2(frame, stage2_output, fps_value):
    if stage2_output is None:
        cv2.putText(frame, f"FPS={fps_value:.1f}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        return

    method = stage2_output.get("method", "-")
    label = stage2_output.get("label", "NONE")
    score = float(stage2_output.get("score", 0.0))

    if label == "MATCH":
        color = (0, 220, 0)
    elif label == "NOT_MATCH":
        color = (0, 0, 255)
    else:
        color = (255, 255, 0)

    cv2.putText(frame, f"Stage2={method}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{label} {score:.2f}", (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS={fps_value:.1f}", (12, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)


def fit_display_frame(frame):
    height, width = frame.shape[:2]
    scale = min(
        DISPLAY_MAX_WIDTH / max(1, width),
        DISPLAY_MAX_HEIGHT / max(1, height),
        1.0,
    )
    if scale >= 1.0:
        return frame

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def build_mog2_view(stage1_output, stage2_output):
    mask = stage1_output.get("data")
    bboxes = stage1_output.get("bboxes") or []
    line_x = stage1_output.get("line_x")
    method = stage1_output.get("method", "stage1")

    if mask is None:
        frame = stage1_output.get("frame")
        if frame is None:
            mog2_view = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            height, width = frame.shape[:2]
            mog2_view = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        mog2_view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    cv2.putText(mog2_view, f"{method.upper()} View", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    if line_x is not None:
        x = int(line_x)
        cv2.line(mog2_view, (x, 0), (x, mog2_view.shape[0] - 1), (255, 140, 0), 2)

    for bbox in bboxes:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(mog2_view, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(mog2_view, (cx, cy), 4, (0, 165, 255), -1)

    if stage2_output is not None:
        label = stage2_output.get("label", "NONE")
        score = float(stage2_output.get("score", 0.0))
        color = (0, 220, 0) if label == "MATCH" else (0, 0, 255) if label == "NOT_MATCH" else (255, 255, 0)
        cv2.putText(mog2_view, f"{label} {score:.2f}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    return mog2_view


def show_frames(original_frame, mog2_frame):
    original_fitted = fit_display_frame(original_frame)
    mog2_fitted = fit_display_frame(mog2_frame)
    cv2.imshow("Original Video", original_fitted)
    cv2.imshow("MOG2 Detection", mog2_fitted)
    key = cv2.waitKey(1) & 0xFF
    return key not in (ord("q"), ord("Q"))


def main():
    global GLOBAL_RESULT, GLOBAL_RUNNING, STAGE1_METHOD, STAGE2_METHOD, CAPTURE_FINISHED, GLOBAL_FRAME, GLOBAL_FRAME_ID

    args = parse_args()
    STAGE1_METHOD = args.stage1_method
    STAGE2_METHOD = args.stage2_method
    GLOBAL_RUNNING = True
    CAPTURE_FINISHED = False
    GLOBAL_FRAME = None
    GLOBAL_FRAME_ID = -1
    while True:
        try:
            FRAME_QUEUE.get_nowait()
        except queue.Empty:
            break

    stage1.reset_state()

    stage2.configure(
        refs_path=args.refs,
        device=args.device,
        model_name=args.model_name,
        threshold=args.threshold,
    )

    source = resolve_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source: {args.source}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    is_camera = isinstance(source, int)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    print(f"main: source={args.source}")
    print(f"main: stage1={STAGE1_METHOD}")
    print(f"main: stage2={STAGE2_METHOD}")
    print(f"main: refs={args.refs}")
    print(f"main: device={args.device}")
    print(f"main: display_max={DISPLAY_MAX_WIDTH}x{DISPLAY_MAX_HEIGHT}")
    if STAGE1_METHOD == "mog2":
        print(
            "main: mog2_settings="
            f"threshold={MOG2_BINARY_THRESHOLD} "
            f"min_area={MOG2_MIN_AREA} "
            f"min_width={MOG2_MIN_WIDTH} "
            f"min_height={MOG2_MIN_HEIGHT} "
            f"bbox_padding={MOG2_BBOX_PADDING} "
            f"inside_margin={MOG2_INSIDE_MARGIN} "
            f"merge_gap={MOG2_MERGE_GAP} "
            f"line_fraction={STAGE1_LINE_FRACTION} "
            f"center_match_distance={STAGE1_CENTER_MATCH_DISTANCE}"
        )
    elif STAGE1_METHOD == "fastsam":
        print(
            "main: fastsam_settings="
            f"model={FASTSAM_MODEL_PATH} "
            f"conf={FASTSAM_CONF_THRESHOLD} "
            f"iou={FASTSAM_NMS_IOU} "
            f"imgsz={FASTSAM_IMGSZ} "
            f"min_box_area={FASTSAM_MIN_BOX_AREA} "
            f"merge_gap={FASTSAM_MERGE_BOX_GAP} "
            f"line_fraction={STAGE1_LINE_FRACTION} "
            f"center_match_distance={STAGE1_CENTER_MATCH_DISTANCE}"
        )

    capture_thread = threading.Thread(target=capture_loop, args=(cap, is_camera, source_fps), daemon=True)
    capture_thread.start()

    processed_frames = 0
    last_seen_id = -1
    last_time = time.perf_counter()

    try:
        while GLOBAL_RUNNING:
            frame, last_seen_id = get_next_frame(last_seen_id)
            if frame is None:
                if CAPTURE_FINISHED:
                    break
                time.sleep(0.01)
                continue

            stage1_kwargs = {}
            if STAGE1_METHOD == "mog2":
                stage1_kwargs = {
                    "binary_threshold": MOG2_BINARY_THRESHOLD,
                    "min_area": MOG2_MIN_AREA,
                    "min_width": MOG2_MIN_WIDTH,
                    "min_height": MOG2_MIN_HEIGHT,
                    "bbox_padding": MOG2_BBOX_PADDING,
                    "inside_margin": MOG2_INSIDE_MARGIN,
                    "merge_gap": MOG2_MERGE_GAP,
                    "line_fraction": STAGE1_LINE_FRACTION,
                    "center_match_distance": STAGE1_CENTER_MATCH_DISTANCE,
                }
            elif STAGE1_METHOD == "fastsam":
                stage1_kwargs = {
                    "fastsam_model_path": FASTSAM_MODEL_PATH,
                    "fastsam_conf_threshold": FASTSAM_CONF_THRESHOLD,
                    "fastsam_nms_iou": FASTSAM_NMS_IOU,
                    "fastsam_imgsz": FASTSAM_IMGSZ,
                    "fastsam_motion_diff_thresh": FASTSAM_MOTION_DIFF_THRESH,
                    "fastsam_min_box_area": FASTSAM_MIN_BOX_AREA,
                    "fastsam_max_box_area_ratio": FASTSAM_MAX_BOX_AREA_RATIO,
                    "fastsam_max_box_width_ratio": FASTSAM_MAX_BOX_WIDTH_RATIO,
                    "fastsam_max_box_height_ratio": FASTSAM_MAX_BOX_HEIGHT_RATIO,
                    "fastsam_dark_roi_thresh": FASTSAM_DARK_ROI_THRESH,
                    "fastsam_motion_overlap_thresh": FASTSAM_MOTION_OVERLAP_THRESH,
                    "fastsam_merge_box_iou_thresh": FASTSAM_MERGE_BOX_IOU_THRESH,
                    "fastsam_merge_box_gap": FASTSAM_MERGE_BOX_GAP,
                    "fastsam_merge_axis_overlap": FASTSAM_MERGE_AXIS_OVERLAP,
                    "line_fraction": STAGE1_LINE_FRACTION,
                    "center_match_distance": STAGE1_CENTER_MATCH_DISTANCE,
                }

            stage1_output = stage1.run(
                frame,
                method=STAGE1_METHOD,
                bg_subtractor=bg_subtractor,
                **stage1_kwargs,
            )
            if stage1_output.get("triggered") and stage1_output.get("triggered_bbox") is not None:
                trigger_input = dict(stage1_output)
                trigger_input["bbox"] = stage1_output.get("triggered_bbox")
                trigger_input["center"] = stage1_output.get("triggered_center")
                GLOBAL_RESULT = stage2.run(trigger_input, method=STAGE2_METHOD)
            elif stage1_output.get("bbox") is not None:
                GLOBAL_RESULT = {
                    "type": "result",
                    "method": STAGE2_METHOD,
                    "label": "WAIT_LINE",
                    "score": 0.0,
                    "bbox": stage1_output.get("bbox"),
                }
            else:
                GLOBAL_RESULT = {
                    "type": "result",
                    "method": STAGE2_METHOD,
                    "label": "NO_OBJECT",
                    "score": 0.0,
                    "bbox": None,
                }

            now = time.perf_counter()
            fps_value = 1.0 / max(1e-6, now - last_time)
            last_time = now

            display_frame = frame.copy()
            draw_stage1(display_frame, stage1_output)
            draw_stage2(display_frame, GLOBAL_RESULT, fps_value)
            mog2_frame = build_mog2_view(stage1_output, GLOBAL_RESULT)

            if args.show and not show_frames(display_frame, mog2_frame):
                GLOBAL_RUNNING = False
                break

            processed_frames += 1
            if processed_frames % 30 == 0:
                det_count = len(stage1_output.get("bboxes") or [])
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                print(
                    f"Frame {processed_frames}/{total_frames or '?'}  "
                    f"FPS={fps_value:.1f}  Det={det_count}"
                )

            if args.max_frames and processed_frames >= args.max_frames:
                GLOBAL_RUNNING = False
                break
    finally:
        GLOBAL_RUNNING = False
        capture_thread.join(timeout=1.0)
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

    print("main: done")
    print(f"main: processed_frames={processed_frames}")
    print(f"main: last_result={GLOBAL_RESULT}")


if __name__ == "__main__":
    main()
