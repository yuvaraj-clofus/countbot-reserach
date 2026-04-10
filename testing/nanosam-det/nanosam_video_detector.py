#!/usr/bin/env python3
"""
NanoSAM Object Detector — Conveyor Belt Edition
================================================
What this script does:
  - Loads MobileSAM (vit_t) on Jetson Orin Nano GPU
  - Reads a video frame-by-frame, resizes to 640x480
  - Computes a motion mask (frame diff) to ignore the static black conveyor belt
  - Probes SAM only at motion-region points, skipping dark (conveyor) pixels
  - Filters out large/dark bounding boxes that cover the conveyor background
  - Removes child bounding boxes (boxes fully inside a larger box)
  - Keeps only detections with confidence >= CONF_THRESHOLD (0.85)
  - Runs SAM every PROCESS_EVERY_N frames and reuses result in between (speed)
  - Clears GPU cache each frame to avoid Jetson memory fragmentation
  - Overlays FPS + frame count + detection count on output

Tunable constants (all in one place below):
  CONF_THRESHOLD        — minimum SAM score to keep a detection
  POINTS_PER_SIDE       — grid density (4 = 16 pts, 6 = 36 pts, 8 = 64 pts — lower = faster)
  PROCESS_EVERY_N       — run SAM every N frames; reuse last result between (1 = every frame)
  MOTION_DIFF_THRESH    — pixel diff to count as motion (lower = more sensitive)
  DARK_PIXEL_THRESH     — brightness below which a pixel is treated as black conveyor
  MIN_BOX_AREA          — minimum bounding box pixel area (filters noise)
  MAX_BOX_AREA_RATIO    — max box area as fraction of frame (filters whole-frame boxes)
  MAX_BOX_WIDTH_RATIO   — max box width as fraction of frame width
  MAX_BOX_HEIGHT_RATIO  — max box height as fraction of frame height
  DARK_ROI_THRESH       — mean brightness below which a box ROI is treated as conveyor
  MOTION_OVERLAP_THRESH — fraction of box area that must overlap motion region to keep it
"""

import os

# ── Jetson GPU memory fix — must be set BEFORE torch is imported ─────────────
# Prevents NvMapMemAllocInternalTagged error 12 (CUDA memory fragmentation).
# Tells the CUDA caching allocator to split blocks at 128 MB max,
# so it never tries to grab one giant contiguous slab that Jetson can't give.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np
import torch

# Disable cuDNN auto-tuner — it caches large kernel configs that
# fragment Jetson unified memory and cause allocation failures.
torch.backends.cudnn.benchmark = False

# ── Tunable thresholds ──────────────────────────────────────────────────────
CONF_THRESHOLD        = 0.80   # SAM score gate — only keep detections above this
POINTS_PER_SIDE       = 4      # Grid points per side (4×4 = 16 pts per frame, fast on Jetson)
PROCESS_EVERY_N       = 1      # Run SAM every N frames (2 = ~2x faster preview)
MOTION_DIFF_THRESH    = 30     # Frame-diff threshold to count as motion (0-255)
DARK_PIXEL_THRESH     = 40     # Skip SAM point if pixel brightness < this (black conveyor)
MIN_BOX_AREA          = 500    # Ignore bounding boxes smaller than this (noise)
MAX_BOX_AREA_RATIO    = 0.50   # Ignore boxes covering > 50% of frame
MAX_BOX_WIDTH_RATIO   = 0.70   # Ignore boxes wider than 70% of frame width
MAX_BOX_HEIGHT_RATIO  = 0.70   # Ignore boxes taller than 70% of frame height
DARK_ROI_THRESH       = 40     # Ignore boxes whose interior is mostly dark (conveyor)
MOTION_OVERLAP_THRESH = 0.20   # At least 20% of box area must overlap motion region
# ────────────────────────────────────────────────────────────────────────────

try:
    from mobile_sam import sam_model_registry, SamPredictor
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "mobile_sam"], check=True)
    from mobile_sam import sam_model_registry, SamPredictor


# ── Device setup ─────────────────────────────────────────────────────────────

def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


# ── Model loading ─────────────────────────────────────────────────────────────

def load_predictor(device):
    checkpoint_path = "/home/vikbot/Documents/countbot/testing/edgesam_det/weights/mobile_sam.pt"
    try:
        sam = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        print("Loaded MobileSAM vit_t  →  SamPredictor")
        return predictor
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


# ── Child-box removal ─────────────────────────────────────────────────────────

def remove_child_boxes(bboxes):
    """Remove any bounding box fully contained inside another box."""
    if len(bboxes) <= 1:
        return bboxes
    keep = [True] * len(bboxes)
    for i, (ax1, ay1, ax2, ay2, _) in enumerate(bboxes):
        if not keep[i]:
            continue
        for j, (bx1, by1, bx2, by2, _) in enumerate(bboxes):
            if i == j or not keep[i]:
                continue
            if bx1 <= ax1 and by1 <= ay1 and bx2 >= ax2 and by2 >= ay2:
                keep[i] = False
                break
    return [b for b, k in zip(bboxes, keep) if k]


# ── Per-frame detection ───────────────────────────────────────────────────────

@torch.inference_mode()
def detect_objects_in_frame(frame, predictor, motion_mask=None):
    """
    Encode frame once, then probe SAM at each active grid point.
    Active = inside motion region AND pixel brightness >= DARK_PIXEL_THRESH.
    Returns filtered, deduplicated list of (x1, y1, x2, y2, conf).
    """
    height, width = frame.shape[:2]
    frame_area    = height * width
    frame_gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Encode image once — runs the heavy ViT encoder on GPU
    predictor.set_image(frame)

    # Build candidate point grid (POINTS_PER_SIDE × POINTS_PER_SIDE)
    step_y = height // POINTS_PER_SIDE
    step_x = width  // POINTS_PER_SIDE
    ys = np.arange(step_y // 2, height, step_y)
    xs = np.arange(step_x // 2, width,  step_x)
    point_grid = [(int(y), int(x)) for y in ys for x in xs]

    # Keep only points inside motion regions
    if motion_mask is not None:
        point_grid = [
            (py, px) for py, px in point_grid
            if 0 <= py < height and 0 <= px < width and motion_mask[py, px] > 0
        ]

    raw_boxes = []

    for py, px in point_grid:
        # Skip dark pixels — the black conveyor surface
        if frame_gray[py, px] < DARK_PIXEL_THRESH:
            continue

        try:
            masks, scores, _ = predictor.predict(
                point_coords=np.array([[px, py]]),
                point_labels=np.array([1]),
                multimask_output=False,
            )
        except Exception as e:
            print(f"\nSAM predict error: {e}")
            continue

        if scores[0] < CONF_THRESHOLD:
            continue

        mask = masks[0].astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_BOX_AREA or area > frame_area * MAX_BOX_AREA_RATIO:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            x1, y1, x2, y2 = x, y, x + bw, y + bh

            if bw > width * MAX_BOX_WIDTH_RATIO or bh > height * MAX_BOX_HEIGHT_RATIO:
                continue

            # Motion overlap — box must cover at least MOTION_OVERLAP_THRESH of moving area
            if motion_mask is not None:
                roi_m = motion_mask[y1:y2, x1:x2]
                if roi_m.size > 0 and np.count_nonzero(roi_m) / roi_m.size < MOTION_OVERLAP_THRESH:
                    continue

            # Darkness filter — skip if box interior is mostly the black conveyor
            roi_g = frame_gray[y1:y2, x1:x2]
            if roi_g.size > 0 and roi_g.mean() < DARK_ROI_THRESH:
                continue

            raw_boxes.append((x1, y1, x2, y2, float(scores[0])))

    # Free GPU cache to prevent memory fragmentation on Jetson
    torch.cuda.empty_cache()

    return remove_child_boxes(raw_boxes)


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_bounding_boxes(frame, bboxes):
    out = frame.copy()
    for x1, y1, x2, y2, conf in bboxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out, f"{conf:.2f}",
            (x1, max(y1 - 8, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2,
        )
    return out


# ── Main video loop ───────────────────────────────────────────────────────────

def process_video(video_path, output_path=None, max_frames=None):
    print(f"Processing video: {video_path}")

    device    = setup_device()
    predictor = load_predictor(device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return

    src_fps      = cap.get(cv2.CAP_PROP_FPS)
    src_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Source: {src_w}x{src_h} @ {src_fps:.1f} FPS  |  {total_frames} frames")
    print(f"Settings: conf={CONF_THRESHOLD}  grid={POINTS_PER_SIDE}x{POINTS_PER_SIDE}="
          f"{POINTS_PER_SIDE**2}pts  every={PROCESS_EVERY_N}fr  "
          f"motion_thresh={MOTION_DIFF_THRESH}")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, src_fps, (640, 480))
        print(f"Saving output → {output_path}")

    frame_count = 0
    prev_gray   = None
    fps_counter = 0
    tick_start  = cv2.getTickCount()
    last_bboxes = []   # reuse on skipped frames

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if max_frames and frame_count > max_frames:
                break

            frame = cv2.resize(frame, (640, 480))
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Motion mask
            motion_mask = None
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                _, motion_mask = cv2.threshold(diff, MOTION_DIFF_THRESH, 255, cv2.THRESH_BINARY)
                motion_mask = cv2.morphologyEx(
                    motion_mask, cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                )
            prev_gray = gray.copy()

            # Run SAM every PROCESS_EVERY_N frames
            if frame_count % PROCESS_EVERY_N == 0:
                last_bboxes = detect_objects_in_frame(frame, predictor, motion_mask)

            frame_out = draw_bounding_boxes(frame, last_bboxes)

            # Overlays
            fps_counter += 1
            elapsed  = (cv2.getTickCount() - tick_start) / cv2.getTickFrequency()
            live_fps = fps_counter / elapsed if elapsed > 0 else 0.0

            cv2.putText(frame_out, f"FPS: {live_fps:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_out, f"Frame: {frame_count}/{total_frames}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_out, f"Det: {len(last_bboxes)}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            print(f"Frame {frame_count}/{total_frames}  FPS={live_fps:.1f}  Det={len(last_bboxes)}", end="\r")

            if writer:
                writer.write(frame_out)

            cv2.imshow("NanoSAM Detection", frame_out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nStopped by user")
                break

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"\nDone — {frame_count} frames processed")


if __name__ == "__main__":
    video_path  = "/home/vikbot/Documents/countbot/testing/video/bearing_wheel.mp4"
    output_path = "/home/vikbot/Documents/countbot/testing/nanosam_output.mp4"

    process_video(video_path, output_path=output_path, max_frames=None)
