#!/usr/bin/env python3
"""
python3 dinodiff-live-v2-edit.py  --rotate 90 --entry-line 30,20,30,75 --exit-l
ine 80,20,80,75

dinodiff-live-v2-edit.py — Conveyor Object Detection using DINOv2
                       Two-line entry/exit counting to reject false positives.

Improvement over v1:
    Objects must cross the ENTRY line first, then the EXIT line to be counted.
    Random detections that pop up between the two lines are never registered
    and therefore never counted.

Controls:  q=quit  r=recapture reference  d=toggle debug stats
           (video-file mode) space=advance one frame
"""

import argparse
from collections import OrderedDict
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
GST_PIPELINE = (
    "v4l2src device=/dev/video0 ! "
    "videoconvert ! "
    "video/x-raw,format=BGR,width=640,height=480 ! "
    "appsink drop=1"
)

DEVICE = "mps" if torch.backends.mps.is_available() else (
         "cuda" if torch.cuda.is_available() else "cpu")

DINO_MODEL   = "dinov2_vits14"
PATCH_SIZE   = 14


def make_divisible_by(val, divisor):
    """Round val to nearest multiple of divisor."""
    return int(round(val / divisor)) * divisor


# Target resolution — computed from round numbers, divisible by 14
TARGET_W = make_divisible_by(640, PATCH_SIZE)  # 644
TARGET_H = make_divisible_by(480, PATCH_SIZE)  # 476

REF_FRAMES    = 100 # number of initial frames to capture as reference
TOPK          = 3      # top-k references for similarity (reduces flicker)
THRESH        = 0.5    # cosine-distance threshold for anomaly detection
MIN_AREA      = 200    # minimum contour area to keep
TEMPORAL_ALPHA = 0.7   # temporal smoothing: weight for current mask vs prev

# Two-line counting — fractions of frame width
LINE_ENTRY_FRAC = 0.3   # entry line (objects must cross this first)
LINE_EXIT_FRAC  = 0.8   # exit line  (crossing here after entry = counted)

# Tracker settings
MAX_MATCH_DIST  = 120   # max pixel distance to match centroid across frames
MAX_DISAPPEARED = 15    # frames before dropping an unmatched track

# ROI crop fractions (run DINO only on conveyor region)
ROI_LEFT  = 0
ROI_RIGHT = 1

# Object snapshot settings
SCRIPT_DIR = Path(__file__).resolve().parent
OBJECTS_DIR = SCRIPT_DIR / "objects"
CONFIRMED_DIR = SCRIPT_DIR / "confirmed"
UNCONFIRMED_DIR = SCRIPT_DIR / "unconfirmed"
OBJECT_CROP_PADDING = 10
OBJECT_EMBED_SIZE = 224
OBJECT_MATCH_THRESHOLD = 0.70

# Debug stats: print similarity stats every N frames
DEBUG_STATS_EVERY = 30


# ── Load DINOv2 ──────────────────────────────────────────────────────────────

print(f"[DINO] Loading {DINO_MODEL} on {DEVICE}...")
model = torch.hub.load("facebookresearch/dinov2", DINO_MODEL, verbose=False)
model.eval().to(DEVICE)
print("[DINO] Ready.")


# ── CLI args ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="DINOv2 conveyor anomaly detection v2 — two-line counting.")
    parser.add_argument("--video", "-v", type=str, default=None,
                        help="Path to video file. If omitted, uses the live GStreamer pipeline.")
    parser.add_argument("--rotate", "-r", type=float, default=0.0,
                        help="Rotate each frame by N degrees clockwise before processing.")
    parser.add_argument("--entry-line", type=str, default=None,
                        help="Entry line segment as x1,y1,x2,y2 in percentages "
                             "(0-100) or fractions (0.0-1.0). "
                             "Default: 30,0,30,100")
    parser.add_argument("--exit-line", type=str, default=None,
                        help="Exit line segment as x1,y1,x2,y2 in percentages "
                             "(0-100) or fractions (0.0-1.0). "
                             "Default: 80,0,80,100")
    return parser.parse_args()


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


def parse_line_segment(spec, default_x_frac):
    """
    Parse x1,y1,x2,y2 as either percent (0-100) or fraction (0.0-1.0).
    Returns line endpoints as fractions in [0, 1].
    """
    if not spec:
        return (default_x_frac, 0.0, default_x_frac, 1.0)

    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 4:
        raise ValueError(f"line expects 4 comma-separated values, got {len(parts)}")
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"line values must be numeric: {spec!r}")

    if max(vals) > 1.0:
        vals = [v / 100.0 for v in vals]

    x1, y1, x2, y2 = vals
    if not all(0.0 <= v <= 1.0 for v in vals):
        raise ValueError(
            f"line values out of range: x1={x1:.3f} y1={y1:.3f} "
            f"x2={x2:.3f} y2={y2:.3f}"
        )
    if x1 == x2 and y1 == y2:
        raise ValueError("line start and end cannot be the same point")
    return (x1, y1, x2, y2)


def line_to_pixels(line_fracs, width, height):
    """Convert fractional line endpoints to pixel coordinates."""
    x1f, y1f, x2f, y2f = line_fracs
    x1 = int(round(x1f * (width - 1)))
    y1 = int(round(y1f * (height - 1)))
    x2 = int(round(x2f * (width - 1)))
    y2 = int(round(y2f * (height - 1)))
    return ((x1, y1), (x2, y2))


def line_label_pos(line_pixels, width, height):
    """Choose a label anchor near the line start while keeping it on-screen."""
    (x1, y1), _ = line_pixels
    label_x = min(max(x1 + 5, 5), max(width - 120, 5))
    label_y = min(max(y1 + 25, 20), max(height - 10, 20))
    return (label_x, label_y)


def _orientation(a, b, c):
    """Return orientation of ordered triplet (a, b, c)."""
    val = ((b[1] - a[1]) * (c[0] - b[0])) - ((b[0] - a[0]) * (c[1] - b[1]))
    if val > 0:
        return 1
    if val < 0:
        return -1
    return 0


def _on_segment(a, b, c):
    """Return True if point b lies on segment ac."""
    return (
        min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and
        min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
    )


def segments_intersect(p1, p2, q1, q2):
    """Return True if segments p1-p2 and q1-q2 intersect."""
    if p1 == p2:
        return False

    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and _on_segment(p1, q1, p2):
        return True
    if o2 == 0 and _on_segment(p1, q2, p2):
        return True
    if o3 == 0 and _on_segment(q1, p1, q2):
        return True
    if o4 == 0 and _on_segment(q1, p2, q2):
        return True

    return False


# ── Image preprocessing ─────────────────────────────────────────────────────

MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


def preprocess(frame_bgr):
    """Resize to target dims (divisible by 14), normalize for DINOv2."""
    img = cv2.resize(frame_bgr, (TARGET_W, TARGET_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    tensor = tensor.to(DEVICE)
    tensor = (tensor - MEAN) / STD
    return tensor


def preprocess_object_crop(crop_bgr):
    """Resize and normalize one object crop for DINOv2 object matching."""
    img = cv2.resize(
        crop_bgr,
        (OBJECT_EMBED_SIZE, OBJECT_EMBED_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    tensor = tensor.to(DEVICE)
    tensor = (tensor - MEAN) / STD
    return tensor


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_patch_tokens(frame_bgr):
    """
    Run DINOv2 on a frame and return L2-normalized patch tokens.
    Returns: (num_patches, embed_dim) tensor
    """
    tensor = preprocess(frame_bgr)
    features = model.forward_features(tensor)
    tokens = features["x_norm_patchtokens"]  # (1, N, D)
    tokens = tokens.squeeze(0)                # (N, D)
    tokens = F.normalize(tokens, dim=1)
    return tokens


@torch.no_grad()
def extract_object_embedding(crop_bgr):
    """Return one normalized DINO embedding for an object crop."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    tensor = preprocess_object_crop(crop_bgr)
    embedding = model(tensor)
    return F.normalize(embedding, dim=-1).squeeze(0)


def compare_object_embeddings(master_embedding, object_embedding):
    """Return cosine similarity between two object embeddings."""
    if master_embedding is None or object_embedding is None:
        return None

    score = F.cosine_similarity(
        master_embedding.unsqueeze(0),
        object_embedding.unsqueeze(0),
        dim=-1,
    )
    return float(score.item())


def get_patch_grid_size():
    """Return (grid_h, grid_w) for the configured target resolution."""
    return TARGET_H // PATCH_SIZE, TARGET_W // PATCH_SIZE


# ── Reference bank ────────────────────────────────────────────────────────────

@torch.no_grad()
def build_reference_bank(rois):
    """
    Build a reference bank from multiple empty-belt ROI crops.
    Returns: (num_patches, num_frames, embed_dim) tensor
    """
    all_tokens = []
    for i, roi in enumerate(rois):
        tokens = extract_patch_tokens(roi)
        tokens = F.normalize(tokens, dim=1)
        all_tokens.append(tokens)
        print(f"  [REF] Frame {i+1}/{len(rois)} embedded")
    bank = torch.stack(all_tokens, dim=0).permute(1, 0, 2)
    return bank


@torch.no_grad()
def compute_anomaly_map(live_tokens, ref_bank):
    """
    Compare live patch tokens against the reference bank.
    Returns:
        sim_map:  (grid_h, grid_w)  similarity values [0, 1]
        diff_map: (grid_h, grid_w)  anomaly score = 1 - sim
    """
    grid_h, grid_w = get_patch_grid_size()
    live_tokens = F.normalize(live_tokens, dim=1)
    sims = torch.sum(live_tokens.unsqueeze(1) * ref_bank, dim=2)
    k = min(TOPK, sims.shape[1])
    topk_vals, _ = sims.topk(k=k, dim=1)
    mean_sims = topk_vals.mean(dim=1)
    sim_map = mean_sims.reshape(grid_h, grid_w).cpu().numpy()
    diff_map = 1.0 - sim_map
    return sim_map, diff_map


# ── Visualization ─────────────────────────────────────────────────────────────

def make_heatmap(diff_map, orig_h, orig_w, thresh):
    """Convert anomaly diff_map to a color heatmap anchored to thresh."""
    norm = np.clip(diff_map / (2.0 * max(thresh, 1e-6)), 0, 1)
    heatmap = (norm * 255).astype(np.uint8)
    heatmap = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap


def ensure_output_dir(path):
    """Create an output directory if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def crop_object_image(frame_bgr, bbox, padding=OBJECT_CROP_PADDING):
    """Crop a padded object bounding box from the frame."""
    if frame_bgr is None or bbox is None:
        return None

    frame_h, frame_w = frame_bgr.shape[:2]
    x, y, w, h = [int(v) for v in bbox]
    if w <= 0 or h <= 0:
        return None

    x0 = max(x - padding, 0)
    y0 = max(y - padding, 0)
    x1 = min(x + w + padding, frame_w)
    y1 = min(y + h + padding, frame_h)

    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return crop.copy()


def object_filename(object_index, track_id, frame_n):
    """Build a stable filename for an exited object crop."""
    filename = f"object_{object_index:04d}_track_{track_id:04d}_frame_{frame_n:04d}.jpg"
    return filename


def save_crop_image(crop, folder, object_index, track_id, frame_n):
    """Save one already-cropped object image to disk and return the saved path."""
    if crop is None or crop.size == 0:
        return None

    filename = object_filename(object_index, track_id, frame_n)
    out_path = Path(folder) / filename
    ok = cv2.imwrite(str(out_path), crop)
    if not ok:
        return None
    return str(out_path)


def save_object_image(frame_bgr, bbox, folder, object_index, track_id, frame_n):
    """Save one object crop to disk and return the saved path."""
    crop = crop_object_image(frame_bgr, bbox)
    return save_crop_image(crop, folder, object_index, track_id, frame_n)


def build_object_record(object_index, track_id, frame_n, time_sec, bbox,
                        object_path, decision_path, similarity, decision):
    """Build metadata for confirmed/unconfirmed object arrays."""
    return {
        "object_index": object_index,
        "track_id": track_id,
        "frame_n": frame_n,
        "time_sec": round(float(time_sec), 4),
        "bbox": tuple(int(v) for v in bbox) if bbox is not None else None,
        "object_path": object_path,
        "decision_path": decision_path,
        "similarity": similarity,
        "decision": decision,
    }


# ── Simple centroid tracker ───────────────────────────────────────────────────

class CentroidTracker:
    """
    Tracks objects across frames by nearest-centroid matching.
    Each track holds: centroid history, whether it crossed the entry line,
    and whether it has been counted (crossed exit line after entry).
    """

    def __init__(self, max_dist=MAX_MATCH_DIST, max_disappeared=MAX_DISAPPEARED):
        self.next_id = 0
        self.objects = OrderedDict()       # id -> (cx, cy)
        self.boxes = OrderedDict()         # id -> (x, y, w, h)
        self.disappeared = OrderedDict()   # id -> frames since last seen
        self.crossed_entry = OrderedDict() # id -> True if crossed entry line
        self.counted = OrderedDict()       # id -> True if already counted
        self.prev_pos = OrderedDict()      # id -> previous (cx, cy) for crossing detection
        self.max_dist = max_dist
        self.max_disappeared = max_disappeared

    def reset(self):
        self.next_id = 0
        self.objects.clear()
        self.boxes.clear()
        self.disappeared.clear()
        self.crossed_entry.clear()
        self.counted.clear()
        self.prev_pos.clear()

    def _register(self, cx, cy, bbox):
        oid = self.next_id
        self.objects[oid] = (cx, cy)
        self.boxes[oid] = bbox
        self.disappeared[oid] = 0
        self.crossed_entry[oid] = False
        self.counted[oid] = False
        self.prev_pos[oid] = (cx, cy)
        self.next_id += 1
        return oid

    def _deregister(self, oid):
        del self.objects[oid]
        del self.boxes[oid]
        del self.disappeared[oid]
        del self.crossed_entry[oid]
        del self.counted[oid]
        del self.prev_pos[oid]

    def update(self, detections):
        """
        detections: list of {"centroid": (cx, cy), "bbox": (x, y, w, h)}.
        Returns dict of {object_id: (cx, cy)} for all active tracks.
        """
        # No detections — mark all existing objects as disappeared
        if len(detections) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            return self.objects

        input_centroids = np.array([det["centroid"] for det in detections], dtype=np.int32)

        # No existing objects — register all
        if len(self.objects) == 0:
            for det in detections:
                cx, cy = det["centroid"]
                self._register(cx, cy, det["bbox"])
            return self.objects

        # Match existing objects to new centroids by nearest distance
        object_ids = list(self.objects.keys())
        object_cents = np.array(list(self.objects.values()))

        # Pairwise distance matrix
        D = np.linalg.norm(object_cents[:, None] - input_centroids[None, :], axis=2)

        # Greedy matching: sort all (row, col) pairs by distance
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
            self.prev_pos[oid] = self.objects[oid]
            self.objects[oid] = (cx, cy)
            self.boxes[oid] = detections[col]["bbox"]
            self.disappeared[oid] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Handle unmatched existing objects
        for row in range(len(object_ids)):
            if row not in used_rows:
                oid = object_ids[row]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)

        # Handle unmatched new centroids — register them
        for col in range(len(input_centroids)):
            if col not in used_cols:
                cx, cy = int(input_centroids[col][0]), int(input_centroids[col][1])
                self._register(cx, cy, detections[col]["bbox"])

        return self.objects

    def check_line_crossings(self, entry_line, exit_line):
        """
        Check which tracked objects crossed the entry or exit lines this frame.
        Returns a list of newly counted object IDs.
        """
        new_count_ids = []

        for oid in list(self.objects.keys()):
            current_pos = self.objects[oid]
            prev_pos = self.prev_pos.get(oid, current_pos)

            if not self.crossed_entry[oid]:
                if segments_intersect(prev_pos, current_pos, entry_line[0], entry_line[1]):
                    self.crossed_entry[oid] = True

            if self.crossed_entry[oid] and not self.counted[oid]:
                if segments_intersect(prev_pos, current_pos, exit_line[0], exit_line[1]):
                    self.counted[oid] = True
                    new_count_ids.append(oid)

        return new_count_ids


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    rotate_deg = args.rotate
    try:
        entry_line_fracs = parse_line_segment(args.entry_line, LINE_ENTRY_FRAC)
        exit_line_fracs = parse_line_segment(args.exit_line, LINE_EXIT_FRAC)
    except ValueError as e:
        print(f"ERROR: {e}")
        return

    if args.video:
        cap = cv2.VideoCapture(args.video)
        src_desc = f"file: {args.video}"
    else:
        cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
        src_desc = f"live GStreamer: {GST_PIPELINE}"

    if not cap.isOpened():
        print(f"ERROR: Cannot open source ({src_desc})")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("ERROR: Could not read first frame from source")
        cap.release()
        return
    first_frame = rotate_frame(first_frame, rotate_deg)

    H, W = first_frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    grid_h, grid_w = get_patch_grid_size()

    entry_line = line_to_pixels(entry_line_fracs, W, H)
    exit_line = line_to_pixels(exit_line_fracs, W, H)

    # ROI pixel bounds
    roi_x0 = int(ROI_LEFT * W)
    roi_x1 = int(ROI_RIGHT * W)
    roi_w = roi_x1 - roi_x0

    print(f"Source: {src_desc}")
    print(f"Rotate: {rotate_deg}")
    print(f"Video : {W}x{H} @ {fps:.1f} fps (post-rotation)")
    print(f"ROI   : x=[{roi_x0}, {roi_x1}]  ({roi_w}px wide)")
    print(f"DINO input : {TARGET_W}x{TARGET_H}  ->  patch grid {grid_w}x{grid_h}")
    print(f"Ref frames : {REF_FRAMES}")
    print(f"Entry line : {entry_line[0]} -> {entry_line[1]}")
    print(f"Exit  line : {exit_line[0]} -> {exit_line[1]}")
    print(f"Threshold  : {THRESH}")

    # ── Step 1: Capture reference frames ──────────────────────────────────────
    print(f"\n[REF] Capturing first {REF_FRAMES} frames as empty-belt reference...")
    ref_rois = [first_frame[:, roi_x0:roi_x1]]
    while len(ref_rois) < REF_FRAMES:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Video ended before capturing enough reference frames")
            return
        frame = rotate_frame(frame, rotate_deg)
        ref_rois.append(frame[:, roi_x0:roi_x1])

    ref_bank = build_reference_bank(ref_rois)
    print(f"[REF] Bank ready: {ref_bank.shape}  (patches x refs x dim)\n")
    ensure_output_dir(OBJECTS_DIR)
    ensure_output_dir(CONFIRMED_DIR)
    ensure_output_dir(UNCONFIRMED_DIR)
    print(f"[SAVE] Object crops will be stored in: {Path(OBJECTS_DIR).resolve()}")
    print(f"[SAVE] Confirmed crops will be stored in: {Path(CONFIRMED_DIR).resolve()}")
    print(f"[SAVE] Unconfirmed crops will be stored in: {Path(UNCONFIRMED_DIR).resolve()}")
    print(f"[MATCH] First saved object becomes master. Threshold={OBJECT_MATCH_THRESHOLD:.2f}")

    # ── Threshold slider ──────────────────────────────────────────────────────
    win_name = "DINOv2 Anomaly Detection v2"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    thresh_val = [int(THRESH * 100)]

    def on_thresh(val):
        thresh_val[0] = val

    cv2.createTrackbar("Threshold", win_name, thresh_val[0], 100, on_thresh)

    # ── Step 2: Process video ─────────────────────────────────────────────────
    tracker = CentroidTracker()
    count = 0
    saved_object_count = 0
    master_object = None
    master_embedding = None
    all_objects = []
    confirmed_objects = []
    unconfirmed_objects = []
    object_decisions = {}  # oid -> "confirmed" | "unconfirmed"
    debug_mode = False
    prev_mask = None
    frame_n = 0
    loop_fps = 0.0
    compute_fps = 0.0
    loop_timer = cv2.getTickCount()
    tick_freq = cv2.getTickFrequency()

    frame_clock = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = rotate_frame(frame, rotate_deg)
        frame_n += 1

        # Auto-skip frames to match real-time (mirrors live drop=1)
        if args.video and fps > 0:
            elapsed_sec = (cv2.getTickCount() - frame_clock) / tick_freq
            frames_behind = int(elapsed_sec * fps) - 1
            for _ in range(max(0, frames_behind)):
                if not cap.grab():
                    break
                frame_n += 1
        frame_clock = cv2.getTickCount()

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

        live_tokens = extract_patch_tokens(roi)
        sim_map, diff_map = compute_anomaly_map(live_tokens, ref_bank)

        cur_thresh = thresh_val[0] / 100.0
        diff_roi = cv2.resize(diff_map, (roi_w, H), interpolation=cv2.INTER_LINEAR)
        mask_roi = (diff_roi > cur_thresh).astype(np.uint8) * 255

        kernel = np.ones((5, 5), np.uint8)
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel, iterations=1)

        mask_float = mask_roi.astype(np.float32)
        if prev_mask is not None:
            mask_float = TEMPORAL_ALPHA * mask_float + (1.0 - TEMPORAL_ALPHA) * prev_mask
        prev_mask = mask_float
        mask_roi = (mask_float > 127).astype(np.uint8) * 255

        mask_full = np.zeros((H, W), dtype=np.uint8)
        mask_full[:, roi_x0:roi_x1] = mask_roi

        contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        display = frame.copy()

        # Draw ROI bounds
        cv2.rectangle(display, (roi_x0, 0), (roi_x1, H), (100, 100, 100), 1)

        # Draw entry line (yellow) and exit line (blue)
        cv2.line(display, entry_line[0], entry_line[1], (0, 255, 255), 2)
        cv2.line(display, exit_line[0], exit_line[1], (255, 0, 0), 2)

        # Label the lines
        cv2.putText(display, "ENTRY", line_label_pos(entry_line, W, H),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display, "EXIT", line_label_pos(exit_line, W, H),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Collect centroids from valid contours
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA]
        detections = []
        for cnt in valid_contours:
            rect = cv2.minAreaRect(cnt)
            box_pts = cv2.boxPoints(rect).astype(int)
            cv2.drawContours(display, [box_pts], 0, (0, 255, 0), 2)

            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                detections.append({
                    "centroid": (cx, cy),
                    "bbox": (x, y, w_box, h_box),
                })

        # Update tracker
        tracker.update(detections)

        # Check line crossings, count, and save object crops on exit
        new_count_ids = tracker.check_line_crossings(entry_line, exit_line)
        count += len(new_count_ids)
        if new_count_ids:
            print(f"[COUNT {count}] +{len(new_count_ids)} object(s) crossed entry->exit at frame {frame_n}")

        for oid in new_count_ids:
            bbox = tracker.boxes.get(oid)
            crop = crop_object_image(frame, bbox)
            saved_object_count += 1
            object_path = save_crop_image(
                crop=crop,
                folder=OBJECTS_DIR,
                object_index=saved_object_count,
                track_id=oid,
                frame_n=frame_n,
            )
            if not object_path:
                print(f"[SAVE ERROR] Could not store crop for track #{oid} at frame {frame_n}")
                continue

            object_embedding = extract_object_embedding(crop)
            if object_embedding is None:
                print(f"[MATCH ERROR] Could not embed crop for track #{oid} at frame {frame_n}")
                continue

            if master_embedding is None:
                decision = "confirmed"
                similarity = 1.0
                decision_dir = CONFIRMED_DIR
                master_embedding = object_embedding
                print(f"[MASTER] Track #{oid} set as master object from {object_path}")
            else:
                similarity = compare_object_embeddings(master_embedding, object_embedding)
                is_match = similarity is not None and similarity >= OBJECT_MATCH_THRESHOLD
                decision = "confirmed" if is_match else "unconfirmed"
                decision_dir = CONFIRMED_DIR if is_match else UNCONFIRMED_DIR

            decision_path = save_crop_image(
                crop=crop,
                folder=decision_dir,
                object_index=saved_object_count,
                track_id=oid,
                frame_n=frame_n,
            )
            if not decision_path:
                print(f"[SAVE ERROR] Could not store {decision} crop for track #{oid} at frame {frame_n}")
                continue

            record = build_object_record(
                object_index=saved_object_count,
                track_id=oid,
                frame_n=frame_n,
                time_sec=video_sec,
                bbox=bbox,
                object_path=object_path,
                decision_path=decision_path,
                similarity=round(float(similarity), 4) if similarity is not None else None,
                decision=decision,
            )
            object_decisions[oid] = decision
            all_objects.append(record)
            if decision == "confirmed":
                confirmed_objects.append(record)
                if master_object is None:
                    master_object = record
            else:
                unconfirmed_objects.append(record)

            print(
                f"[SAVE {saved_object_count}] Track #{oid} object={object_path}  "
                f"{decision}={decision_path}  score={record['similarity']:.4f}"
            )

        # Draw tracked objects with IDs and state
        for oid, (cx, cy) in tracker.objects.items():
            entered = tracker.crossed_entry[oid]
            counted = tracker.counted[oid]

            decision = object_decisions.get(oid)
            if counted and decision == "confirmed":
                color = (0, 200, 0)     # green — confirmed
                label = f"#{oid} CONFIRMED"
            elif counted and decision == "unconfirmed":
                color = (0, 0, 220)     # red — unconfirmed
                label = f"#{oid} UNCONFIRMED"
            elif entered:
                color = (0, 165, 255)   # orange — entered, awaiting exit
                label = f"#{oid} >"
            else:
                color = (180, 180, 180) # gray — not yet entered
                label = f"#{oid}"

            cv2.circle(display, (cx, cy), 6, color, -1)
            cv2.putText(display, label, (cx + 10, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        compute_elapsed = cv2.getTickCount() - compute_start
        compute_fps = tick_freq / max(compute_elapsed, 1)

        if debug_mode and frame_n % DEBUG_STATS_EVERY == 0:
            print(f"  [STATS] frame={frame_n}  sim  mean={sim_map.mean():.3f}  "
                  f"min={sim_map.min():.3f}  max={sim_map.max():.3f}  "
                  f"thresh={cur_thresh:.2f}  tracks={len(tracker.objects)}")

        # Build display: detection | B&W mask | heatmap overlay
        mask_bgr = cv2.cvtColor(mask_full, cv2.COLOR_GRAY2BGR)
        diff_full = np.zeros((H, W), dtype=np.float32)
        diff_full[:, roi_x0:roi_x1] = diff_roi
        heatmap = make_heatmap(diff_full, H, W, cur_thresh)
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
        combined = np.hstack([display, mask_bgr, overlay])

        max_w = 1400
        scale = min(1.0, max_w / combined.shape[1])
        if scale < 1.0:
            combined = cv2.resize(combined, (int(combined.shape[1] * scale),
                                             int(combined.shape[0] * scale)))

        # Banner
        banner_h = 70
        banner = np.zeros((banner_h, combined.shape[1], 3), dtype=np.uint8)
        line1 = (f"COUNT: {count}    CONFIRMED: {len(confirmed_objects)}    "
                 f"UNCONFIRMED: {len(unconfirmed_objects)}    "
                 f"Time: {time_str}    Frame: {frame_n}    "
                 f"Thresh: {cur_thresh:.2f}    Tracks: {len(tracker.objects)}")
        line2 = (f"FPS  src: {fps:.1f}    compute: {compute_fps:.1f}    "
                 f"loop: {loop_fps:.1f}    "
                 f"[ENTRY=yellow  EXIT=blue]")
        cv2.putText(banner, line1, (15, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 80), 2, cv2.LINE_AA)
        cv2.putText(banner, line2, (15, 58), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 200, 255), 2, cv2.LINE_AA)
        combined = np.vstack([banner, combined])

        cv2.imshow(win_name, combined)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            print("[REF] Recapturing reference frames...")
            ref_rois = []
            while len(ref_rois) < REF_FRAMES:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = rotate_frame(frame, rotate_deg)
                ref_rois.append(frame[:, roi_x0:roi_x1])
            if ref_rois:
                ref_bank = build_reference_bank(ref_rois)
                print(f"[REF] Bank updated: {ref_bank.shape}")
            count = 0
            tracker.reset()
            master_object = None
            master_embedding = None
            all_objects.clear()
            confirmed_objects.clear()
            unconfirmed_objects.clear()
            object_decisions.clear()
            prev_mask = None
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"[DEBUG STATS {'ON' if debug_mode else 'OFF'}]")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n{'=' * 40}")
    print(f"Final count: {count}")
    print(f"Confirmed objects: {len(confirmed_objects)}")
    print(f"Unconfirmed objects: {len(unconfirmed_objects)}")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    main()
