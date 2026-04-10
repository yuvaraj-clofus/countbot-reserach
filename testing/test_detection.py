"""
test_detection.py — Live HSV detection + DINOv2 matching + line-crossing counter

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEPS
  1. Load stored DINOv2 embeddings from PT_FILE
  2. Open camera, apply HSV color thresholding each frame
  3. Track contour centroids across frames (CentroidTracker)
  4. When centroid crosses CROSSING_LINE (right→left):
       a. Extract bbox ROI from raw frame
       b. Generate DINOv2 embedding for the ROI
       c. Compare against stored embeddings (cosine similarity)
       d. If max similarity >= THRESHOLD  →  MATCH (good) else NO MATCH (bad)
  5. Display live preview with overlay, crossing line, counts

SUMMARY
  - Red vertical line     = crossing gate
  - Green bbox            = matched object (GOOD)
  - Red bbox              = no match (BAD)
  - Orange bbox           = tracked, not yet counted
  - GOOD / BAD displayed top-left
  - Press 'q' to quit

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIG  (edit values below before running)
"""

# ── CONFIG ────────────────────────────────────────────────────────────────────
PT_FILE       = "models/1bearing.pt"  # path to .pt model
# CAMERA_INDEX  = "/home/vikbot/Documents/countbot/testing/video/bearing_wheel.mp4"  # camera index (int) or video file path (str)
CAMERA_INDEX  = 0  # camera index (int) or video file path (str)
CAM_WIDTH     = 640        # capture resolution width
CAM_HEIGHT    = 480        # capture resolution height

# HSV background range — same scale as dashboard (H: 0-360, S/V: 0-100)
H_MIN, H_MAX  = 0,   360   # Hue   (0–360)
S_MIN, S_MAX  = 0,    100   # Sat   (0–100)
V_MIN, V_MAX  = 37,    100   # Val   (0–100)
MIN_CONTOUR   = 500        # minimum contour area in px²

CROSSING_LINE = 60         # 0–100 % of frame width (vertical gate line)
THRESHOLD     = 0.75       # cosine similarity threshold for a match (0.0–1.0)
# ─────────────────────────────────────────────────────────────────────────────

import math
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

def load_embeddings(pt_path):
    print(f"[init] loading embeddings from {pt_path}")
    data = torch.load(pt_path, map_location='cpu')
    embeddings = []
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, torch.Tensor):
                embeddings.append(F.normalize(v.cpu(), dim=-1))
    elif isinstance(data, torch.Tensor):
        embeddings = [F.normalize(data.cpu(), dim=-1)]
    if not embeddings:
        raise ValueError("No embeddings found in pt file")
    result = torch.stack(embeddings, dim=0)
    print(f"[init] {len(result)} reference embeddings loaded")
    return result


def load_dino():
    print("[init] loading DINOv2 model ...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    print("[init] DINOv2 ready")
    return model, tf


# ---------------------------------------------------------------------------
# Embedding + matching
# ---------------------------------------------------------------------------

def get_embedding(roi_bgr, model, tf):
    if roi_bgr is None or roi_bgr.size == 0:
        return None
    rgb    = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    tensor = tf(Image.fromarray(rgb)).unsqueeze(0)
    with torch.no_grad():
        emb = model(tensor)
    return F.normalize(emb, dim=-1).squeeze(0)


def is_match(roi_bgr, ref_embeddings, model, tf, threshold):
    emb = get_embedding(roi_bgr, model, tf)
    if emb is None:
        return False, 0.0
    cos = F.cosine_similarity(emb.unsqueeze(0), ref_embeddings, dim=-1)
    best = float(cos.max().item())
    return best >= threshold, round(best, 3)


# ---------------------------------------------------------------------------
# HSV segmentation — same scale as dashboard (H:0-360, S/V:0-100)
# ---------------------------------------------------------------------------

def segment_frame(frame, h_min, h_max, s_min, s_max, v_min, v_max, min_contour):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lo = np.array([int(h_min / 360 * 179),
                   int(s_min / 100 * 255),
                   int(v_min / 100 * 255)], dtype=np.uint8)
    hi = np.array([int(h_max / 360 * 179),
                   int(s_max / 100 * 255),
                   int(v_max / 100 * 255)], dtype=np.uint8)

    bg_mask = cv2.inRange(hsv, lo, hi)
    mask    = cv2.bitwise_not(bg_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= min_contour]

    output  = frame.copy()
    overlay = output.copy()
    cv2.drawContours(overlay, valid, -1, (203, 150, 255), -1)
    cv2.addWeighted(overlay, 0.35, output, 0.65, 0, output)
    cv2.drawContours(output, valid, -1, (60, 0, 140), 3)

    return output, mask, valid


# ---------------------------------------------------------------------------
# Centroid tracker (mirrors dashboard.py CentroidTracker)
# ---------------------------------------------------------------------------

class CentroidTracker:
    def __init__(self, max_disappeared=15, max_distance=80):
        self.next_id         = 0
        self.objects         = {}   # id -> (cx, cy)
        self.disappeared     = {}   # id -> frames missing
        self.max_disappeared = max_disappeared
        self.max_distance    = max_distance
        self.counted         = set()

    def register(self, centroid):
        self.objects[self.next_id]     = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, obj_id):
        self.objects.pop(obj_id, None)
        self.disappeared.pop(obj_id, None)

    def update(self, centroids):
        if not centroids:
            for obj_id in list(self.disappeared):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return dict(self.objects)

        if not self.objects:
            for c in centroids:
                self.register(c)
            return dict(self.objects)

        obj_ids   = list(self.objects.keys())
        obj_cents = list(self.objects.values())

        rows = []
        for oc in obj_cents:
            row = [math.hypot(oc[0] - nc[0], oc[1] - nc[1]) for nc in centroids]
            rows.append(row)

        used_rows, used_cols = set(), set()
        pairs = sorted(
            [(rows[r][c], r, c) for r in range(len(rows)) for c in range(len(centroids))],
            key=lambda x: x[0],
        )
        for dist, r, c in pairs:
            if r in used_rows or c in used_cols:
                continue
            if dist > self.max_distance:
                break
            obj_id = obj_ids[r]
            self.objects[obj_id]     = centroids[c]
            self.disappeared[obj_id] = 0
            used_rows.add(r)
            used_cols.add(c)

        for r, obj_id in enumerate(obj_ids):
            if r not in used_rows:
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

        for c in range(len(centroids)):
            if c not in used_cols:
                self.register(centroids[c])

        return dict(self.objects)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    ref_embeddings      = load_embeddings(PT_FILE)
    dino_model, dino_tf = load_dino()

    is_video = isinstance(CAMERA_INDEX, str)
    if is_video:
        if not os.path.isfile(CAMERA_INDEX):
            print(f"[error] video file not found: {CAMERA_INDEX}")
            return
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_ANY)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        print(f"[error] could not open: {CAMERA_INDEX}")
        return

    # ── Trackbar window ──────────────────────────────────────────────────────
    cv2.namedWindow("HSV Tune", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HSV Tune", 400, 280)
    def nothing(_): pass
    cv2.createTrackbar("H Min",       "HSV Tune", H_MIN,        360, nothing)
    cv2.createTrackbar("H Max",       "HSV Tune", H_MAX,        360, nothing)
    cv2.createTrackbar("S Min",       "HSV Tune", S_MIN,        100, nothing)
    cv2.createTrackbar("S Max",       "HSV Tune", S_MAX,        100, nothing)
    cv2.createTrackbar("V Min",       "HSV Tune", V_MIN,        100, nothing)
    cv2.createTrackbar("V Max",       "HSV Tune", V_MAX,        100, nothing)
    cv2.createTrackbar("Min Area",    "HSV Tune", MIN_CONTOUR, 5000, nothing)
    cv2.createTrackbar("Line % ",     "HSV Tune", CROSSING_LINE, 100, nothing)
    # ─────────────────────────────────────────────────────────────────────────

    tracker        = CentroidTracker(max_disappeared=20, max_distance=100)
    prev_positions = {}   # id -> x
    good_count     = 0
    bad_count      = 0

    print("[run] starting. press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop video
                continue
            break

        # Read trackbar values live
        h_min       = cv2.getTrackbarPos("H Min",    "HSV Tune")
        h_max       = cv2.getTrackbarPos("H Max",    "HSV Tune")
        s_min       = cv2.getTrackbarPos("S Min",    "HSV Tune")
        s_max       = cv2.getTrackbarPos("S Max",    "HSV Tune")
        v_min       = cv2.getTrackbarPos("V Min",    "HSV Tune")
        v_max       = cv2.getTrackbarPos("V Max",    "HSV Tune")
        min_area    = cv2.getTrackbarPos("Min Area", "HSV Tune")
        line_pct    = cv2.getTrackbarPos("Line % ",  "HSV Tune")

        fh, fw = frame.shape[:2]
        line_x = int(fw * line_pct / 100)

        output, mask, contours = segment_frame(
            frame,
            h_min, h_max,
            s_min, s_max,
            v_min, v_max,
            max(1, min_area),
        )

        # Draw crossing line
        cv2.line(output, (line_x, 0), (line_x, fh), (0, 0, 255), 2)

        # Collect centroids + raw ROIs
        centroids   = []
        cent_to_roi = {}
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2
            centroids.append((cx, cy))
            cent_to_roi[(cx, cy)] = frame[y:y + h, x:x + w]

        objects = tracker.update(centroids)

        # Draw boxes for all tracked objects
        for obj_id, (cx, cy) in objects.items():
            color = (0, 165, 255)   # orange = tracked, not yet counted
            cv2.circle(output, (cx, cy), 4, color, -1)

        # Check crossing (right → left: prev_x > line_x and now cx <= line_x)
        for obj_id, (cx, cy) in objects.items():
            prev_x = prev_positions.get(obj_id)
            if prev_x is not None and obj_id not in tracker.counted:
                if prev_x > line_x >= cx:
                    # Find closest ROI
                    best_roi, best_dist = None, float('inf')
                    for cent, roi in cent_to_roi.items():
                        d = math.hypot(cent[0] - cx, cent[1] - cy)
                        if d < best_dist:
                            best_dist, best_roi = d, roi

                    matched, score = is_match(best_roi, ref_embeddings, dino_model, dino_tf, THRESHOLD)

                    if matched:
                        good_count += 1
                        print(f"[count] MATCH {score} → GOOD: {good_count}  BAD: {bad_count}")
                    else:
                        bad_count += 1
                        print(f"[count] NO MATCH {score} → GOOD: {good_count}  BAD: {bad_count}")

                    tracker.counted.add(obj_id)

            prev_positions[obj_id] = cx

        # Prune dead tracks from prev_positions
        live_ids = set(objects.keys())
        for oid in list(prev_positions):
            if oid not in live_ids:
                del prev_positions[oid]

        # Count display
        cv2.rectangle(output, (0, 0), (220, 70), (0, 0, 0), -1)
        cv2.putText(output, f"GOOD: {good_count}", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(output, f"BAD:  {bad_count}", (8, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Mask preview (top-right corner)
        prev_w, prev_h = 120, 90
        mask_small = cv2.resize(mask, (prev_w, prev_h))
        output[0:prev_h, fw - prev_w:fw] = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)

        cv2.imshow("Detection", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            print(f"[hsv] H:{h_min}-{h_max}  S:{s_min}-{s_max}  V:{v_min}-{v_max}  "
                  f"MinArea:{min_area}  Line:{line_pct}%")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[done] GOOD: {good_count}  BAD: {bad_count}")
    print(f"[hsv]  H:{cv2.getTrackbarPos('H Min','HSV Tune')}-{cv2.getTrackbarPos('H Max','HSV Tune')}  "
          f"S:{cv2.getTrackbarPos('S Min','HSV Tune')}-{cv2.getTrackbarPos('S Max','HSV Tune')}  "
          f"V:{cv2.getTrackbarPos('V Min','HSV Tune')}-{cv2.getTrackbarPos('V Max','HSV Tune')}  "
          f"MinArea:{cv2.getTrackbarPos('Min Area','HSV Tune')}  "
          f"Line:{cv2.getTrackbarPos('Line % ','HSV Tune')}%")


if __name__ == '__main__':
    main()
