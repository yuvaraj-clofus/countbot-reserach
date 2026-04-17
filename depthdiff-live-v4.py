#!/usr/bin/env python3
"""
python3 depthdiff-live-v4.py --rotate 90

depthdiff-live-v3.py — Conveyor Object Counting with Depth-Anything V2
                       Direct depth threshold + two-line entry/exit counting.

Why v3 (vs v2):
    v2 used a reference bank + robust (MAD) standardization. That inverts
    on frames where a large object appears, because MAD widens and the
    standardized belt becomes as "far" from the reference as the object.
    For a fixed top-down camera on a flat belt the problem is much simpler:
    objects sit ABOVE the belt plane -> closer to camera -> higher predicted
    depth (DA-V2 is disparity-like). Direct per-frame normalization plus a
    threshold picks them out cleanly. No reference needed.

Pipeline per frame:
    1. Depth-Anything V2 on the ROI
    2. Min-max normalize depth to 0-255 (per frame)
    3. Threshold at user-adjustable level
    4. Morphology + temporal smoothing
    5. Contours -> centroids -> CentroidTracker
    6. Two-line ENTRY -> EXIT counting (rejects false positives between lines)

Controls:  q=quit   d=toggle debug stats   r=reset count+tracks
           (video-file mode) space=pause/resume
"""

import argparse
from collections import OrderedDict

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


# ── Config ────────────────────────────────────────────────────────────────────
GST_PIPELINE = (
    "v4l2src device=/dev/video0 ! "
    "videoconvert ! "
    "video/x-raw,format=BGR,width=640,height=480 ! "
    "appsink drop=1"
)

DEVICE = "mps" if torch.backends.mps.is_available() else (
         "cuda" if torch.cuda.is_available() else "cpu")

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

THRESH_INIT    = 150    # initial threshold on normalized depth 0-255
MIN_AREA       = 200    # minimum contour area to keep
TEMPORAL_ALPHA = 0.7    # temporal smoothing: weight for current mask vs prev

# Two-line counting — fractions of frame width
LINE_ENTRY_FRAC = 0.1
LINE_EXIT_FRAC  = 0.8

# Tracker settings
MAX_MATCH_DIST  = 120
MAX_DISAPPEARED = 15

# ROI crop fractions (run depth only on conveyor region)
ROI_LEFT  = 0
ROI_RIGHT = 1

DEBUG_STATS_EVERY = 30


# ── Load Depth-Anything V2 ───────────────────────────────────────────────────
print(f"[DEPTH] Loading {DEPTH_MODEL_ID} on {DEVICE}...")
processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).to(DEVICE)
model.eval()
print("[DEPTH] Ready.")


# ── CLI args ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Depth-Anything V2 conveyor counting v3 — direct thresholding.")

    # Source selection — exactly one of --video / --camera / (default GStreamer)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--video", "-v", type=str, default=None,
                     help="Path to a video file.")
    src.add_argument("--camera", "-c", type=int, default=None,
                     help="Camera index for plain cv2.VideoCapture (e.g. 0). "
                          "Use this for webcams / USB cams without GStreamer.")
    # If neither --video nor --camera given, falls back to the GStreamer pipeline.

    parser.add_argument("--start", "-s", type=str, default=None,
                        help="Start timestamp for video files, mm:ss or hh:mm:ss "
                             "(e.g. 01:30). Ignored for live sources.")
    parser.add_argument("--rotate", "-r", type=float, default=90.0,
                        help="Rotate each frame by N degrees clockwise before processing. "
                             "Default: 90.")
    parser.add_argument("--crop", type=str, default=None,
                        help="Crop each frame BEFORE depth estimation. "
                             "Format: x1,y1,x2,y2 as percentages (0-100) or "
                             "fractions (0.0-1.0). Example: '10,20,80,90' keeps "
                             "the rectangle from (10%%, 20%%) to (80%%, 90%%) of "
                             "the (rotated) frame. Applied after --rotate.")
    return parser.parse_args()


def parse_timestamp(ts):
    """Parse 'mm:ss' or 'hh:mm:ss' into seconds (float). Returns None on failure."""
    if not ts:
        return None
    try:
        parts = [float(p) for p in ts.strip().split(":")]
    except ValueError:
        return None
    if len(parts) == 2:
        mm, ss = parts
        return mm * 60 + ss
    if len(parts) == 3:
        hh, mm, ss = parts
        return hh * 3600 + mm * 60 + ss
    return None


def parse_crop(s):
    """
    Parse 'x1,y1,x2,y2' as either percent (0-100) or fraction (0.0-1.0).
    Auto-detects by checking if any value exceeds 1.0.
    Returns (x1, y1, x2, y2) as fractions in [0, 1], or None if s is falsy.
    Raises ValueError on bad input.
    """
    if not s:
        return None
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError(f"--crop expects 4 comma-separated values, got {len(parts)}")
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"--crop values must be numeric: {s!r}")
    # Auto-detect percent vs fraction
    if max(vals) > 1.0:
        vals = [v / 100.0 for v in vals]
    x1, y1, x2, y2 = vals
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(
            f"--crop out of range or flipped: x1={x1:.3f} y1={y1:.3f} "
            f"x2={x2:.3f} y2={y2:.3f}  (need 0 <= x1 < x2 <= 1 and same for y)")
    return (x1, y1, x2, y2)


def apply_crop(frame, crop_fracs):
    """Crop frame to the fraction rectangle (x1, y1, x2, y2). No-op if None."""
    if crop_fracs is None or frame is None:
        return frame
    h, w = frame.shape[:2]
    x1f, y1f, x2f, y2f = crop_fracs
    x1, y1 = int(round(x1f * w)), int(round(y1f * h))
    x2, y2 = int(round(x2f * w)), int(round(y2f * h))
    # Guard against zero-size crops from extreme fractions
    x2 = max(x2, x1 + 1)
    y2 = max(y2, y1 + 1)
    return frame[y1:y2, x1:x2]


def rotate_frame(frame, degrees):
    """Rotate frame clockwise by `degrees`. Handles arbitrary angles."""
    if frame is None:
        return frame
    d = degrees % 360
    if d == 0:
        return frame
    if d == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if d == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if d == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w = frame.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, -d, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]
    return cv2.warpAffine(frame, M, (new_w, new_h))


# ── Depth estimation ─────────────────────────────────────────────────────────

@torch.no_grad()
def compute_depth(frame_bgr):
    """Run Depth-Anything V2 on a frame; return float32 depth at frame size."""
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    inputs = processor(pil_img, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)

    depth = outputs.predicted_depth              # [1, Hm, Wm]
    depth = F.interpolate(depth.unsqueeze(1),    # -> [1, 1, Hm, Wm]
                          size=(h, w),
                          mode="bilinear",
                          align_corners=False).squeeze()
    return depth.detach().cpu().numpy().astype(np.float32)


def normalize_depth_to_uint8(depth):
    """Min-max normalize a depth map to 0-255 uint8 (per frame auto-scale).
    In DA-V2's disparity convention, brighter = closer to camera."""
    d_min = float(depth.min())
    d_max = float(depth.max())
    if d_max - d_min < 1e-6:
        return np.zeros(depth.shape, dtype=np.uint8)
    norm = 255.0 * (depth - d_min) / (d_max - d_min)
    return norm.astype(np.uint8)


# ── Simple centroid tracker ──────────────────────────────────────────────────

class CentroidTracker:
    """
    Tracks objects across frames by nearest-centroid matching.
    Each track holds: centroid history, whether it crossed the entry line,
    and whether it has been counted (crossed exit line after entry).
    """

    def __init__(self, max_dist=MAX_MATCH_DIST, max_disappeared=MAX_DISAPPEARED):
        self.next_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.crossed_entry = OrderedDict()
        self.counted = OrderedDict()
        self.prev_cx = OrderedDict()
        self.max_dist = max_dist
        self.max_disappeared = max_disappeared

    def reset(self):
        self.next_id = 0
        self.objects.clear()
        self.disappeared.clear()
        self.crossed_entry.clear()
        self.counted.clear()
        self.prev_cx.clear()

    def _register(self, cx, cy):
        oid = self.next_id
        self.objects[oid] = (cx, cy)
        self.disappeared[oid] = 0
        self.crossed_entry[oid] = False
        self.counted[oid] = False
        self.prev_cx[oid] = cx
        self.next_id += 1
        return oid

    def _deregister(self, oid):
        del self.objects[oid]
        del self.disappeared[oid]
        del self.crossed_entry[oid]
        del self.counted[oid]
        del self.prev_cx[oid]

    def update(self, centroids):
        if len(centroids) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            return self.objects

        input_centroids = np.array(centroids)

        if len(self.objects) == 0:
            for (cx, cy) in centroids:
                self._register(cx, cy)
            return self.objects

        object_ids = list(self.objects.keys())
        object_cents = np.array(list(self.objects.values()))

        D = np.linalg.norm(object_cents[:, None] - input_centroids[None, :], axis=2)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_dist:
                continue

            oid = object_ids[row]
            cx, cy = int(input_centroids[col][0]), int(input_centroids[col][1])
            self.prev_cx[oid] = self.objects[oid][0]
            self.objects[oid] = (cx, cy)
            self.disappeared[oid] = 0
            used_rows.add(row)
            used_cols.add(col)

        for row in range(len(object_ids)):
            if row not in used_rows:
                oid = object_ids[row]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)

        for col in range(len(input_centroids)):
            if col not in used_cols:
                cx, cy = int(input_centroids[col][0]), int(input_centroids[col][1])
                self._register(cx, cy)

        return self.objects

    def check_line_crossings(self, entry_x, exit_x):
        """Count objects that cross exit line AFTER crossing entry line."""
        new_counts = 0
        for oid in list(self.objects.keys()):
            cx, _ = self.objects[oid]
            px = self.prev_cx.get(oid, cx)

            if not self.crossed_entry[oid]:
                if px < entry_x <= cx or cx < entry_x <= px:
                    self.crossed_entry[oid] = True

            if self.crossed_entry[oid] and not self.counted[oid]:
                if px < exit_x <= cx or cx < exit_x <= px:
                    self.counted[oid] = True
                    new_counts += 1
        return new_counts


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rotate_deg = args.rotate

    try:
        crop_fracs = parse_crop(args.crop)
    except ValueError as e:
        print(f"ERROR: {e}")
        return

    if args.video:
        cap = cv2.VideoCapture(args.video)
        src_desc = f"file: {args.video}"
        is_live = False
    elif args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        src_desc = f"live camera index: {args.camera}"
        is_live = True
    else:
        cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
        src_desc = f"live GStreamer: {GST_PIPELINE}"
        is_live = True

    if not cap.isOpened():
        print(f"ERROR: Cannot open source ({src_desc})")
        return

    # Seek to --start timestamp before reading the first frame (file sources only)
    start_sec = parse_timestamp(args.start)
    if start_sec is not None:
        if is_live:
            print(f"[SEEK] --start ignored for live sources")
        else:
            ok = cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
            mm = int(start_sec // 60)
            ss = start_sec - mm * 60
            print(f"[SEEK] Skipping to {mm:02d}:{ss:06.3f}  "
                  f"({start_sec:.3f}s)  {'ok' if ok else 'FAILED'}")

    ret, first_frame = cap.read()
    if not ret:
        print("ERROR: Could not read first frame from source")
        cap.release()
        return
    first_frame = rotate_frame(first_frame, rotate_deg)
    first_frame = apply_crop(first_frame, crop_fracs)

    H, W = first_frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    entry_x = int(W * LINE_ENTRY_FRAC)
    exit_x  = int(W * LINE_EXIT_FRAC)

    roi_x0 = int(ROI_LEFT * W)
    roi_x1 = int(ROI_RIGHT * W)
    roi_w = roi_x1 - roi_x0

    print(f"Source     : {src_desc}")
    print(f"Rotate     : {rotate_deg}")
    if crop_fracs is not None:
        x1f, y1f, x2f, y2f = crop_fracs
        print(f"Crop       : ({x1f:.0%}, {y1f:.0%}) -> ({x2f:.0%}, {y2f:.0%})  "
              f"=> working frame {W}x{H}")
    else:
        print(f"Crop       : (none)")
    print(f"Video      : {W}x{H} @ {fps:.1f} fps (post-rotation/crop)")
    print(f"ROI        : x=[{roi_x0}, {roi_x1}]  ({roi_w}px wide)")
    print(f"Depth model: {DEPTH_MODEL_ID}  on  {DEVICE}")
    print(f"Entry line : x={entry_x}  ({LINE_ENTRY_FRAC:.0%} of width)")
    print(f"Exit  line : x={exit_x}  ({LINE_EXIT_FRAC:.0%} of width)")
    print(f"Threshold  : {THRESH_INIT}  (on normalized depth 0-255)\n")

    # After probing dimensions we rewind file sources back to the intended
    # start position (--start if given, else 0). Live sources aren't seekable.
    if args.video:
        if start_sec is not None:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── Threshold slider ──────────────────────────────────────────────────────
    win_name = "Depth-Anything V2 Counter v3"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    thresh_val = [THRESH_INIT]

    def on_thresh(val):
        thresh_val[0] = val

    cv2.createTrackbar("Threshold", win_name, thresh_val[0], 255, on_thresh)

    tracker = CentroidTracker()
    count = 0
    debug_mode = False
    prev_mask = None
    frame_n = 0
    loop_fps = 0.0
    compute_fps = 0.0
    loop_timer = cv2.getTickCount()
    tick_freq = cv2.getTickFrequency()
    video_delay_ms = max(1, int(round(1000.0 / fps))) if args.video else 1
    video_paused = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = rotate_frame(frame, rotate_deg)
        frame = apply_crop(frame, crop_fracs)
        frame_n += 1

        now = cv2.getTickCount()
        loop_fps = tick_freq / max(now - loop_timer, 1)
        loop_timer = now

        compute_start = cv2.getTickCount()

        video_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if not video_ms or video_ms <= 0:
            video_ms = (frame_n / fps) * 1000.0 if fps else 0.0
        video_sec = video_ms / 1000.0
        mm = int(video_sec // 60)
        ss = video_sec - mm * 60
        time_str = f"{mm:02d}:{ss:06.3f}"

        roi = frame[:, roi_x0:roi_x1]

        # 1) Depth estimation
        depth = compute_depth(roi)

        # 2) Normalize to 0-255 (brighter = closer in DA-V2 disparity convention)
        depth_u8 = normalize_depth_to_uint8(depth)

        # 3) Threshold: pixels brighter than `thresh` are anomalies (objects)
        cur_thresh = thresh_val[0]
        _, mask_roi = cv2.threshold(depth_u8, cur_thresh, 255, cv2.THRESH_BINARY)

        # 4) Morphology + temporal smoothing (same as v2)
        kernel = np.ones((5, 5), np.uint8)
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel, iterations=1)

        mask_float = mask_roi.astype(np.float32)
        if prev_mask is not None:
            mask_float = TEMPORAL_ALPHA * mask_float + (1.0 - TEMPORAL_ALPHA) * prev_mask
        prev_mask = mask_float
        mask_roi = (mask_float > 127).astype(np.uint8) * 255

        # 5) Place ROI mask into full-frame canvas
        mask_full = np.zeros((H, W), dtype=np.uint8)
        mask_full[:, roi_x0:roi_x1] = mask_roi

        contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        display = frame.copy()

        # Draw ROI bounds
        cv2.rectangle(display, (roi_x0, 0), (roi_x1, H), (100, 100, 100), 1)

        # Entry (yellow) and exit (blue) lines
        cv2.line(display, (entry_x, 0), (entry_x, H), (0, 255, 255), 2)
        cv2.line(display, (exit_x, 0), (exit_x, H), (255, 0, 0), 2)
        cv2.putText(display, "ENTRY", (entry_x + 5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display, "EXIT", (exit_x + 5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 6) Collect centroids from valid contours
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA]
        centroids = []
        for cnt in valid_contours:
            rect = cv2.minAreaRect(cnt)
            box_pts = cv2.boxPoints(rect).astype(int)
            cv2.drawContours(display, [box_pts], 0, (0, 255, 0), 2)

            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))

        # 7) Tracker update + line crossing check
        tracker.update(centroids)
        new_counts = tracker.check_line_crossings(entry_x, exit_x)
        count += new_counts
        if new_counts > 0:
            print(f"[COUNT {count}] +{new_counts} object(s) crossed entry->exit at frame {frame_n}")

        # Draw tracks with state colors
        for oid, (cx, cy) in tracker.objects.items():
            entered = tracker.crossed_entry[oid]
            counted = tracker.counted[oid]
            if counted:
                color = (0, 200, 0)
                label = f"#{oid} OK"
            elif entered:
                color = (0, 165, 255)
                label = f"#{oid} >"
            else:
                color = (180, 180, 180)
                label = f"#{oid}"
            cv2.circle(display, (cx, cy), 6, color, -1)
            cv2.putText(display, label, (cx + 10, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        compute_elapsed = cv2.getTickCount() - compute_start
        compute_fps = tick_freq / max(compute_elapsed, 1)

        if debug_mode and frame_n % DEBUG_STATS_EVERY == 0:
            print(f"  [STATS] frame={frame_n}  depth u8 "
                  f"min={depth_u8.min()} max={depth_u8.max()} "
                  f"mean={depth_u8.mean():.1f}  thresh={cur_thresh}  "
                  f"tracks={len(tracker.objects)}")

        # ── Build 3-panel display: detection | mask | colorized depth ────────
        mask_bgr = cv2.cvtColor(mask_full, cv2.COLOR_GRAY2BGR)

        # Colorized depth map (full frame) for debugging/tuning
        depth_full_u8 = np.zeros((H, W), dtype=np.uint8)
        depth_full_u8[:, roi_x0:roi_x1] = depth_u8
        depth_color = cv2.applyColorMap(depth_full_u8, cv2.COLORMAP_MAGMA)
        # Overlay threshold contour line on the colorized depth so you can see
        # where the cut falls
        _, thresh_line = cv2.threshold(depth_full_u8, cur_thresh, 255, cv2.THRESH_BINARY)
        thresh_contours, _ = cv2.findContours(thresh_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(depth_color, thresh_contours, -1, (0, 255, 255), 1)

        combined = np.hstack([display, mask_bgr, depth_color])

        max_w = 1400
        scale = min(1.0, max_w / combined.shape[1])
        if scale < 1.0:
            combined = cv2.resize(combined, (int(combined.shape[1] * scale),
                                             int(combined.shape[0] * scale)))

        # Banner
        banner_h = 70
        banner = np.zeros((banner_h, combined.shape[1], 3), dtype=np.uint8)
        line1 = (f"COUNT: {count}    Time: {time_str}    "
                 f"Frame: {frame_n}    Thresh: {cur_thresh}    "
                 f"Tracks: {len(tracker.objects)}")
        line2 = (f"FPS  src: {fps:.1f}    compute: {compute_fps:.1f}    "
                 f"loop: {loop_fps:.1f}    "
                 f"[ENTRY=yellow  EXIT=blue]  Depth: {DEPTH_MODEL_ID.split('/')[-1]}")
        cv2.putText(banner, line1, (15, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 80), 2, cv2.LINE_AA)
        cv2.putText(banner, line2, (15, 58), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 200, 255), 2, cv2.LINE_AA)
        combined = np.vstack([banner, combined])

        cv2.imshow(win_name, combined)

        if args.video:
            key = cv2.waitKey(0 if video_paused else video_delay_ms) & 0xFF
        else:
            key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            if args.video:
                video_paused = not video_paused
                print(f"[VIDEO {'PAUSED' if video_paused else 'PLAYING'}]")
        elif key == ord('r'):
            print("[RESET] count + tracks cleared.")
            count = 0
            tracker.reset()
            prev_mask = None
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"[DEBUG STATS {'ON' if debug_mode else 'OFF'}]")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n{'=' * 40}")
    print(f"Final count: {count}")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    main()
