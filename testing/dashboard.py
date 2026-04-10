"""
dashboard.py — Single-file Flask web dashboard for CountBot.
Depends on: flask, opencv-python, numpy
"""

import os
import glob
import threading
import time
import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from flask import Flask, Response, redirect, render_template_string, request, jsonify, url_for

# ---------------------------------------------------------------------------
# App & shared state
# ---------------------------------------------------------------------------

app = Flask(__name__)

lock = threading.Lock()

state = {
    "folder": "",
    "video_index": 0,
    "reference_file": None,
    "running": False,
    "good": 0,
    "bad": 0,
    # HSV background range sliders
    "h_min": 0,
    "h_max": 30,
    "s_min": 0,
    "s_max": 60,
    "v_min": 0,
    "v_max": 80,
    "compare_threshold": 15,
    "line_pos": 60,
    "min_size": 500,
    "roi_rect": None,
    "training": False,
}

# Latest JPEG frame bytes (or None)
frame_buffer = None
frame_lock = threading.Lock()

# Preview frame captured during setup
preview_frame_bytes = None

# DINOv2 embedding cache / model cache
_dino_model = None
_dino_transform = None
_loaded_ref_embeddings_path = None
_loaded_ref_embeddings_tensor = None

# Background capture thread handle
capture_thread = None
stop_event = threading.Event()

# Training internal state
training_last_frame = None
training_saved_rois = []

# Raw frame shared from capture_loop so training page doesn't need its own cap
training_raw_frame = None
training_raw_lock  = threading.Lock()

# Persistent cap for training page when main loop is not running
_train_cap = None
_train_cap_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Auto-detect HSV parameters from a plain conveyor frame
# ---------------------------------------------------------------------------

def auto_detect_hsv(frame):
    """
    Analyse a plain conveyor belt frame and return the background HSV range.
    Returns (h_min, h_max, s_min, s_max, v_min, v_max) all 0-100/360 scale.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:, :, 0].astype(np.float32)
    s_ch = hsv[:, :, 1].astype(np.float32)
    v_ch = hsv[:, :, 2].astype(np.float32)

    # BG H centre: peak of hue histogram
    h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    bg_h_cv = int(np.argmax(h_hist))
    h_std   = float(np.std(h_ch))
    tol     = max(8, int(h_std * 1.5))
    h_min   = int(max(0,   bg_h_cv - tol) / 179 * 360)
    h_max   = int(min(179, bg_h_cv + tol) / 179 * 360)

    # S range: cover mean ± 2σ
    s_mean, s_std = float(np.mean(s_ch)), float(np.std(s_ch))
    s_min = max(0,   int((s_mean - 2 * s_std) / 255 * 100))
    s_max = min(100, int((s_mean + 2 * s_std) / 255 * 100))

    # V range: same
    v_mean, v_std = float(np.mean(v_ch)), float(np.std(v_ch))
    v_min = max(0,   int((v_mean - 2 * v_std) / 255 * 100))
    v_max = min(100, int((v_mean + 2 * v_std) / 255 * 100))

    return h_min, h_max, s_min, s_max, v_min, v_max

# ---------------------------------------------------------------------------
# HSV segmentation (mirrors colorthes.py)
# ---------------------------------------------------------------------------

def segment_frame(frame, h_min, h_max, s_min, s_max, v_min, v_max, min_size=500):
    """
    Exclude the background HSV range and return foreground contours.
    Pixels INSIDE [h_min..h_max, s_min..s_max, v_min..v_max] = background.
    Everything OUTSIDE = foreground object.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Convert 0-360/0-100 ranges to OpenCV scale
    lo = np.array([int(h_min / 360 * 179),
                   int(s_min / 100 * 255),
                   int(v_min / 100 * 255)], dtype=np.uint8)
    hi = np.array([int(h_max / 360 * 179),
                   int(s_max / 100 * 255),
                   int(v_max / 100 * 255)], dtype=np.uint8)

    bg_mask  = cv2.inRange(hsv, lo, hi)          # white = background
    mask     = cv2.bitwise_not(bg_mask)           # white = foreground

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = frame.copy()
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_size]

    # Light pink semi-transparent fill over segmented shape
    overlay = output.copy()
    cv2.drawContours(overlay, valid_contours, -1, (203, 150, 255), -1)
    cv2.addWeighted(overlay, 0.35, output, 0.65, 0, output)

    # Dark pink contour border
    cv2.drawContours(output, valid_contours, -1, (60, 0, 140), 3)

    for cnt in valid_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 0),   6)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)

    return output, mask, valid_contours


# ---------------------------------------------------------------------------
# HSV histogram comparison
# ---------------------------------------------------------------------------

def _hsv_hist(img_bgr):
    """Compute a normalised 3-channel HSV histogram for comparison."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [50], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])
    for h in (h_hist, s_hist, v_hist):
        cv2.normalize(h, h)
    return np.concatenate([h_hist, s_hist, v_hist]).flatten()


def compare_roi_to_reference(roi_bgr, ref_images, compare_thr=0.5):
    """Return True if any reference image matches roi_bgr above compare_thr."""
    if roi_bgr is None or roi_bgr.size == 0 or not ref_images:
        return False
    roi_hist = _hsv_hist(roi_bgr)
    for ref in ref_images:
        ref_hist = _hsv_hist(ref)
        score = cv2.compareHist(
            roi_hist.reshape(-1, 1).astype(np.float32),
            ref_hist.reshape(-1, 1).astype(np.float32),
            cv2.HISTCMP_CORREL,
        )
        if score >= compare_thr:
            return True
    return False


def list_reference_files(folder):
    """Return sorted .pt files from folder."""
    if not folder or not os.path.isdir(folder):
        return []
    files = []
    for name in sorted(os.listdir(folder)):
        if os.path.splitext(name)[1].lower() == ".pt":
            files.append(name)
    return files


def ensure_dino_model():
    global _dino_model, _dino_transform
    if _dino_model is None:
        _dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        _dino_model.eval()
        _dino_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return _dino_model, _dino_transform


def load_reference_embeddings(folder, filename):
    """Load .pt embedding dictionary and return normalized tensor stack."""
    if not folder or not filename:
        return None
    path = os.path.join(folder, filename)
    if not os.path.isfile(path):
        return None

    try:
        data = torch.load(path, map_location="cpu")
    except Exception:
        return None

    embeddings = []
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, torch.Tensor):
                embeddings.append(F.normalize(v.to("cpu"), dim=-1))
    elif isinstance(data, torch.Tensor):
        # If a tensor was stored directly
        embeddings = [F.normalize(data.to("cpu"), dim=-1)]

    if not embeddings:
        return None

    return torch.stack(embeddings, dim=0)


def get_embedding_from_roi(roi_bgr):
    if roi_bgr is None or roi_bgr.size == 0:
        return None
    model, transform_obj = ensure_dino_model()
    dst_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(dst_rgb)
    tensor = transform_obj(pil).unsqueeze(0)
    with torch.no_grad():
        emb = model(tensor)
    emb = F.normalize(emb, dim=-1).squeeze(0)
    return emb


def compare_roi_to_reference_embedding(roi_bgr, ref_embeddings_tensor, compare_thr=0.5):
    if roi_bgr is None or roi_bgr.size == 0 or ref_embeddings_tensor is None or ref_embeddings_tensor.numel() == 0:
        return False
    roi_embedding = get_embedding_from_roi(roi_bgr)
    if roi_embedding is None:
        return False

    cos = F.cosine_similarity(
        roi_embedding.unsqueeze(0),
        ref_embeddings_tensor,
        dim=-1,
    )
    return float(cos.max().item()) >= float(compare_thr)


def load_reference_images(folder, filename):
    """Load the selected reference image (and optionally every image in the folder)."""
    images = []
    if filename:
        path = os.path.join(folder, filename)
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
    return images


# ---------------------------------------------------------------------------
# Simple nearest-centroid tracker
# ---------------------------------------------------------------------------

class CentroidTracker:
    def __init__(self, max_disappeared=15, max_distance=80):
        self.next_id    = 0
        self.objects    = {}   # id -> centroid (x, y)
        self.disappeared = {}  # id -> frames missing
        self.max_disappeared = max_disappeared
        self.max_distance    = max_distance
        # Track which IDs have already been counted
        self.counted    = set()

    def register(self, centroid):
        self.objects[self.next_id]      = centroid
        self.disappeared[self.next_id]  = 0
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

        # Distance matrix
        rows = []
        for oc in obj_cents:
            row = []
            for nc in centroids:
                d = math.hypot(oc[0] - nc[0], oc[1] - nc[1])
                row.append(d)
            rows.append(row)

        # Greedy match (closest pair first)
        used_rows = set()
        used_cols = set()
        pairs = sorted(
            [(rows[r][c], r, c) for r in range(len(rows)) for c in range(len(centroids))],
            key=lambda x: x[0]
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

        # Unmatched existing objects
        for r, obj_id in enumerate(obj_ids):
            if r not in used_rows:
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

        # Unmatched new centroids → register
        for c in range(len(centroids)):
            if c not in used_cols:
                self.register(centroids[c])

        return dict(self.objects)


# ---------------------------------------------------------------------------
# Capture / processing thread
# ---------------------------------------------------------------------------

def capture_loop(video_index, stop_evt):
    global frame_buffer, state, _loaded_ref_embeddings_path, _loaded_ref_embeddings_tensor

    cap = cv2.VideoCapture(video_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        with lock:
            state["running"] = False
        return

    tracker = CentroidTracker(max_disappeared=20, max_distance=100)

    # Previous centroid positions for crossing detection
    prev_positions = {}  # id -> x

    while not stop_evt.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        with training_raw_lock:
            training_raw_frame = frame

        # Read slider values under lock
        with lock:
            h_min    = state["h_min"]
            h_max    = state["h_max"]
            s_min    = state["s_min"]
            s_max    = state["s_max"]
            v_min    = state["v_min"]
            v_max    = state["v_max"]
            cmp_thr  = state["compare_threshold"] / 100.0
            line_pos = state["line_pos"]
            min_size = state["min_size"]
            folder   = state["folder"]
            ref_file = state["reference_file"]

        # Load reference embeddings once per selected pt file
        ref_embeddings = None
        if folder and ref_file:
            ref_path = os.path.join(folder, ref_file)
            if ref_path != _loaded_ref_embeddings_path:
                _loaded_ref_embeddings_tensor = load_reference_embeddings(folder, ref_file)
                _loaded_ref_embeddings_path = ref_path
            ref_embeddings = _loaded_ref_embeddings_tensor
        else:
            _loaded_ref_embeddings_path = None
            _loaded_ref_embeddings_tensor = None

        # Apply ROI cropping if set by training mode
        roi_rect = state.get("roi_rect")
        cropped_frame = frame
        roi_offset = (0, 0)
        if roi_rect:
            x0, y0, w0, h0 = roi_rect
            x0 = max(0, min(x0, frame.shape[1] - 1))
            y0 = max(0, min(y0, frame.shape[0] - 1))
            w0 = max(1, min(w0, frame.shape[1] - x0))
            h0 = max(1, min(h0, frame.shape[0] - y0))
            cropped_frame = frame[y0:y0+h0, x0:x0+w0]
            roi_offset = (x0, y0)

        # Segment
        output, mask, contours = segment_frame(cropped_frame, h_min, h_max, s_min, s_max, v_min, v_max, min_size)

        fh, fw = cropped_frame.shape[:2]
        line_x = int(fw * line_pos / 100)

        # Draw counting line
        cv2.line(output, (line_x, 0), (line_x, fh), (0, 0, 255), 2)

        # Collect centroids from valid contours
        centroids = []
        rois      = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2
            centroids.append((cx, cy))
            rois.append(frame[y:y+h, x:x+w])

        # Build centroid->ROI mapping before tracker scrambles the order
        cent_to_roi = {c: r for c, r in zip(centroids, rois)}

        objects = tracker.update(centroids)

        # Detect line crossings (left → right)
        ref_images = load_reference_images(folder, ref_file)

        for obj_id, (cx, cy) in objects.items():
            prev_x = prev_positions.get(obj_id)
            if prev_x is not None and obj_id not in tracker.counted:
                if prev_x > line_x >= cx:
                    # Find closest ROI centroid
                    best_roi  = None
                    best_dist = float("inf")
                    for cent, roi in cent_to_roi.items():
                        d = math.hypot(cent[0] - cx, cent[1] - cy)
                        if d < best_dist:
                            best_dist = d
                            best_roi  = roi

                    is_good = compare_roi_to_reference_embedding(best_roi, ref_embeddings, cmp_thr)

                    with lock:
                        if is_good:
                            state["good"] += 1
                        else:
                            state["bad"] += 1

                    tracker.counted.add(obj_id)

            prev_positions[obj_id] = cx

        # Prune prev_positions for deregistered objects
        live_ids = set(objects.keys())
        for oid in list(prev_positions.keys()):
            if oid not in live_ids:
                del prev_positions[oid]

        # Overlay mask preview (top-left corner, 160×120)
        preview_w, preview_h = 160, 120
        mask_preview = cv2.resize(mask, (preview_w, preview_h))
        mask_bgr = cv2.cvtColor(mask_preview, cv2.COLOR_GRAY2BGR)
        output[0:preview_h, 0:preview_w] = mask_bgr

        # Encode JPEG
        ret2, jpeg = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ret2:
            with frame_lock:
                frame_buffer = jpeg.tobytes()

    cap.release()
    with lock:
        state["running"] = False


# ---------------------------------------------------------------------------
# MJPEG generator
# ---------------------------------------------------------------------------

def generate_mjpeg():
    while True:
        with frame_lock:
            data = frame_buffer

        if data is None:
            # Send a placeholder grey frame
            placeholder = np.full((480, 640, 3), 36, dtype=np.uint8)
            cv2.putText(
                placeholder,
                "No signal — press Start",
                (80, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (140, 140, 140),
                2,
                cv2.LINE_AA,
            )
            _, jpeg = cv2.imencode(".jpg", placeholder)
            data = jpeg.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
        )
        time.sleep(0.033)  # ~30 fps cap on the stream side


# ---------------------------------------------------------------------------
# HTML templates
# ---------------------------------------------------------------------------

PAGE1_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CountBot — Setup</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #1a1a1a;
    color: #e0e0e0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
  }
  .card {
    background: #242424;
    border-radius: 12px;
    padding: 40px 48px;
    width: 420px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
  }
  h1 { font-size: 1.6rem; font-weight: 600; margin-bottom: 8px; color: #fff; }
  .subtitle { color: #888; font-size: 0.85rem; margin-bottom: 32px; }
  label { display: block; font-size: 0.8rem; color: #aaa; margin-bottom: 6px; letter-spacing: .04em; }
  input[type=text], input[type=number] {
    width: 100%;
    background: #1a1a1a;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    color: #e0e0e0;
    font-size: 0.95rem;
    padding: 10px 14px;
    outline: none;
    transition: border-color .2s;
    margin-bottom: 22px;
  }
  input:focus { border-color: #4caf50; }
  button {
    width: 100%;
    background: #4caf50;
    color: #fff;
    border: none;
    border-radius: 6px;
    font-size: 1rem;
    padding: 12px;
    cursor: pointer;
    font-weight: 600;
    transition: background .2s;
  }
  button:hover { background: #43a047; }
</style>
</head>
<body>
<div class="card">
  <h1>CountBot</h1>
  <p class="subtitle">Configure your session before starting.</p>
  <form method="POST" action="/setup">
    <label>EMBEDDING FOLDER PATH</label>
    <input type="text" name="folder" placeholder="/home/user/embeddings" value="{{ folder }}" required>
    <label>CAMERA INDEX</label>
    <input type="number" name="video_index" value="{{ video_index }}" min="0" step="1">
    <button type="submit">Continue &rarr;</button>
  </form>
</div>
</body>
</html>"""


PREVIEW_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CountBot — Camera Preview</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #1a1a1a;
    color: #e0e0e0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    gap: 24px;
    padding: 24px;
  }
  h1 { font-size: 1.3rem; font-weight: 600; color: #fff; }
  p  { font-size: 0.85rem; color: #888; margin-top: 4px; }
  .frame-box {
    background: #242424;
    border-radius: 12px;
    overflow: hidden;
    max-width: 860px;
    width: 100%;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
  }
  .frame-box img { width: 100%; display: block; }
  .params {
    background: #242424;
    border-radius: 12px;
    padding: 20px 28px;
    max-width: 860px;
    width: 100%;
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 12px;
  }
  .param {
    text-align: center;
    background: #1a1a1a;
    border-radius: 8px;
    padding: 14px 8px;
  }
  .param-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: .06em;
    color: #666;
    margin-bottom: 6px;
  }
  .param-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #4caf50;
  }
  .actions {
    display: flex;
    gap: 14px;
    max-width: 860px;
    width: 100%;
  }
  .btn {
    flex: 1;
    padding: 13px 0;
    border: none;
    border-radius: 8px;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    transition: background .2s;
  }
  .btn-back { background: #333; color: #aaa; }
  .btn-back:hover { background: #3a3a3a; }
  .btn-go   { background: #4caf50; color: #fff; }
  .btn-go:hover { background: #43a047; }
</style>
</head>
<body>
  <div>
    <h1>Camera Preview — Auto-detected HSV</h1>
    <p>Frame captured from the conveyor. Detected background parameters shown below.</p>
  </div>

  <div class="frame-box">
    <img src="/frame_preview" alt="Captured frame">
  </div>

  <div class="params">
    <div class="param">
      <div class="param-label">H Min</div>
      <div class="param-value">{{ h_min }}</div>
    </div>
    <div class="param">
      <div class="param-label">H Max</div>
      <div class="param-value">{{ h_max }}</div>
    </div>
    <div class="param">
      <div class="param-label">S Min</div>
      <div class="param-value">{{ s_min }}</div>
    </div>
    <div class="param">
      <div class="param-label">S Max</div>
      <div class="param-value">{{ s_max }}</div>
    </div>
    <div class="param">
      <div class="param-label">V Min</div>
      <div class="param-value">{{ v_min }}</div>
    </div>
    <div class="param">
      <div class="param-label">V Max</div>
      <div class="param-value">{{ v_max }}</div>
    </div>
  </div>

  <div class="actions">
    <button class="btn btn-back" onclick="history.back()">&#8592; Back</button>
    <button class="btn btn-go" onclick="location.href='/main'">Continue to Dashboard &rarr;</button>
  </div>
</body>
</html>"""


PAGE_TRAIN_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CountBot — Train</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #1a1a1a; color: #e0e0e0; font-family: 'Segoe UI', system-ui, sans-serif; }

  .container {
    display: flex;
    flex-direction: row;
    min-height: 100vh;
    gap: 14px;
    padding: 14px;
  }

  /* Sidebar LEFT */
  .right {
    order: 1;
    width: 300px;
    min-width: 300px;
    background: #242424;
    border-radius: 12px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    overflow-y: auto;
  }

  /* Frame RIGHT */
  .left {
    order: 2;
    flex: 1;
    background: #242424;
    border-radius: 12px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  #canvas {
    border: 1px solid #3a3a3a;
    border-radius: 8px;
    cursor: crosshair;
  }

  .panel {
    padding: 10px;
    border-radius: 8px;
    background: #1a1a1a;
    margin-top: 4px;
    width: 100%;
    font-size: 0.82rem;
  }

  .button {
    border: none;
    font-weight: 600;
    border-radius: 8px;
    cursor: pointer;
    padding: 10px;
    color: #fff;
    background: #4caf50;
    width: 100%;
  }
  .button.warn  { background: #1976d2; }
  .button.back  { background: #555; }

  /* HSV sliders */
  .sliders-section { display: flex; flex-direction: column; gap: 6px; }
  .sliders-section h3 {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: .07em;
    color: #666;
    margin-bottom: 2px;
  }
  .slider-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .slider-name {
    width: 56px;
    min-width: 56px;
    font-size: 0.78rem;
    color: #aaa;
    text-align: right;
  }
  .slider-wrap {
    flex: 1;
    position: relative;
    height: 20px;
    display: flex;
    align-items: center;
  }
  .slider-wrap::before {
    content: '';
    position: absolute;
    left: 0; right: 0;
    height: 4px;
    background: #3a3a3a;
    border-radius: 2px;
    pointer-events: none;
  }
  .slider-fill {
    position: absolute;
    left: 0;
    height: 4px;
    background: #4caf50;
    border-radius: 2px;
    pointer-events: none;
    width: 0%;
  }
  input[type=range] {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 4px;
    background: transparent;
    outline: none;
    cursor: pointer;
    position: relative;
    z-index: 1;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 14px; height: 14px;
    border-radius: 50%;
    background: #fff;
    border: 2px solid #4caf50;
    box-shadow: 0 0 3px rgba(0,0,0,0.4);
    cursor: pointer;
  }
  input[type=range]::-moz-range-thumb {
    width: 14px; height: 14px;
    border-radius: 50%;
    background: #fff;
    border: 2px solid #4caf50;
    cursor: pointer;
  }
  .slider-value {
    width: 36px;
    min-width: 36px;
    text-align: right;
    font-size: 0.78rem;
    color: #ffdc3c;
    font-variant-numeric: tabular-nums;
  }
  .divider { border: none; border-top: 1px solid #333; margin: 4px 0; }
</style>
</head>
<body>
<div class="container">

  <!-- LEFT SIDEBAR -->
  <div class="right">
    <button class="button" id="btn-refresh">Refresh Frame</button>
    <button class="button" id="btn-set-roi">Set ROI &amp; Detect HSV</button>

    <!-- HSV Sliders -->
    <div class="sliders-section" id="hsv-panel">
      <h3>Background HSV Range</h3>

      <div class="slider-row">
        <span class="slider-name">H Min</span>
        <div class="slider-wrap">
          <div class="slider-fill" id="fill-h_min"></div>
          <input type="range" id="sl-h_min" min="0" max="360" value="0"
                 oninput="hsvChanged('h_min', this)">
        </div>
        <span class="slider-value" id="val-h_min">0</span>
      </div>

      <div class="slider-row">
        <span class="slider-name">H Max</span>
        <div class="slider-wrap">
          <div class="slider-fill" id="fill-h_max"></div>
          <input type="range" id="sl-h_max" min="0" max="360" value="30"
                 oninput="hsvChanged('h_max', this)">
        </div>
        <span class="slider-value" id="val-h_max">30</span>
      </div>

      <div class="slider-row">
        <span class="slider-name">S Min</span>
        <div class="slider-wrap">
          <div class="slider-fill" id="fill-s_min"></div>
          <input type="range" id="sl-s_min" min="0" max="100" value="0"
                 oninput="hsvChanged('s_min', this)">
        </div>
        <span class="slider-value" id="val-s_min">0</span>
      </div>

      <div class="slider-row">
        <span class="slider-name">S Max</span>
        <div class="slider-wrap">
          <div class="slider-fill" id="fill-s_max"></div>
          <input type="range" id="sl-s_max" min="0" max="100" value="60"
                 oninput="hsvChanged('s_max', this)">
        </div>
        <span class="slider-value" id="val-s_max">60</span>
      </div>

      <div class="slider-row">
        <span class="slider-name">V Min</span>
        <div class="slider-wrap">
          <div class="slider-fill" id="fill-v_min"></div>
          <input type="range" id="sl-v_min" min="0" max="100" value="0"
                 oninput="hsvChanged('v_min', this)">
        </div>
        <span class="slider-value" id="val-v_min">0</span>
      </div>

      <div class="slider-row">
        <span class="slider-name">V Max</span>
        <div class="slider-wrap">
          <div class="slider-fill" id="fill-v_max"></div>
          <input type="range" id="sl-v_max" min="0" max="100" value="80"
                 oninput="hsvChanged('v_max', this)">
        </div>
        <span class="slider-value" id="val-v_max">80</span>
      </div>

      <div class="slider-row">
        <span class="slider-name">Min Size</span>
        <div class="slider-wrap">
          <div class="slider-fill" id="fill-min_size" style="background:#7eb8e8;"></div>
          <input type="range" id="sl-min_size" min="0" max="10000" value="500"
                 oninput="hsvChanged('min_size', this)">
        </div>
        <span class="slider-value" id="val-min_size" style="color:#7eb8e8;">500</span>
      </div>
    </div>

    <hr class="divider">

    <button class="button" id="btn-save">Save Sample &nbsp;[s]</button>
    <button class="button warn" id="btn-train">Create PT &nbsp;[t]</button>
    <div class="panel" id="train-status">Status: ready</div>
    <div class="panel" id="stats">Samples: 0</div>
    <button class="button back" onclick="window.location.href='/main'">Back</button>
  </div>

  <!-- RIGHT FRAME -->
  <div class="left">
    <h2 id="frame-title">Live Segmentation</h2>
    <canvas id="canvas" width="640" height="680"></canvas>
    <div class="panel" id="coords">ROI: not set — drag on canvas to select object</div>
  </div>

</div>

<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

let drag = false;
let sx=0, sy=0, ex=0, ey=0;
let roi = null;
let samples = 0;
let segmentMode = true;    // always show segmentation overlay like the dashboard

let _scale = 1;
let _offsetX = 0;
let _offsetY = 0;

const HSV_MAX = { h_min:360, h_max:360, s_min:100, s_max:100, v_min:100, v_max:100, min_size:5000 };

// Initialise fill bars
(function(){
  Object.keys(HSV_MAX).forEach(name => {
    const el   = document.getElementById('sl-' + name);
    const fill = document.getElementById('fill-' + name);
    if(el && fill) fill.style.width = (parseInt(el.value)/HSV_MAX[name]*100).toFixed(2)+'%';
  });
})();

let hsvTimer = null;
function hsvChanged(name, el){
  const val = parseInt(el.value, 10);
  document.getElementById('val-' + name).textContent = val;
  const fill = document.getElementById('fill-' + name);
  fill.style.width = (val / HSV_MAX[name] * 100).toFixed(2) + '%';
  clearTimeout(hsvTimer);
  hsvTimer = setTimeout(() => {
    const body = {};
    body[name] = val;
    fetch('/slider', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body)
    }).then(() => { if(segmentMode) drawImage(); });
  }, 80);
}

function setSliders(d){
  ['h_min','h_max','s_min','s_max','v_min','v_max'].forEach(k => {
    if(d[k] !== undefined){
      const el   = document.getElementById('sl-' + k);
      const fill = document.getElementById('fill-' + k);
      const lbl  = document.getElementById('val-' + k);
      el.value = d[k];
      lbl.textContent = d[k];
      fill.style.width = (d[k] / HSV_MAX[k] * 100).toFixed(2) + '%';
    }
  });
}

function drawImage(){
  const img = new Image();
  img.onload = ()=>{
    const canvasW = canvas.width;
    const canvasH = canvas.height;
    ctx.clearRect(0, 0, canvasW, canvasH);

    const scale  = Math.min(canvasW / img.width, canvasH / img.height);
    const drawW  = img.width  * scale;
    const drawH  = img.height * scale;
    const offsetX = (canvasW - drawW) / 2;
    const offsetY = (canvasH - drawH) / 2;

    ctx.drawImage(img, offsetX, offsetY, drawW, drawH);

    // In raw mode, overlay the drawn ROI box
    if(!segmentMode && roi){
      ctx.strokeStyle = '#4caf50';
      ctx.lineWidth = 2;
      ctx.strokeRect(
        offsetX + roi.x * scale,
        offsetY + roi.y * scale,
        roi.w * scale,
        roi.h * scale
      );
    }

    _scale   = scale;
    _offsetX = offsetX;
    _offsetY = offsetY;
  };
  // In segment mode the server draws contours; in raw mode show plain frame
  img.src = (segmentMode ? '/train_segmented_frame' : '/train_frame') + '?ts=' + Date.now();
}

canvas.addEventListener('mousedown', e=>{
  const r = canvas.getBoundingClientRect();
  sx = (e.clientX - r.left - _offsetX) / _scale;
  sy = (e.clientY - r.top  - _offsetY) / _scale;
  drag = true;
});

canvas.addEventListener('mousemove', e=>{
  if(!drag) return;
  const r = canvas.getBoundingClientRect();
  ex = (e.clientX - r.left - _offsetX) / _scale;
  ey = (e.clientY - r.top  - _offsetY) / _scale;

  drawImage();
  ctx.strokeStyle = '#ffdc3c';
  ctx.lineWidth = 2;
  ctx.strokeRect(
    _offsetX + Math.min(sx,ex) * _scale,
    _offsetY + Math.min(sy,ey) * _scale,
    Math.abs(ex-sx) * _scale,
    Math.abs(ey-sy) * _scale
  );
});

canvas.addEventListener('mouseup', ()=>{
  if(!drag) return;
  drag = false;
  roi = {
    x: Math.floor(Math.min(sx,ex)),
    y: Math.floor(Math.min(sy,ey)),
    w: Math.floor(Math.abs(ex-sx)),
    h: Math.floor(Math.abs(ey-sy))
  };
  document.getElementById('coords').innerText =
    'ROI: '+roi.x+','+roi.y+' / '+roi.w+'×'+roi.h+'  — click "Set ROI & Detect HSV"';
  drawImage();
});

function setStatus(t){
  document.getElementById('train-status').innerText = 'Status: ' + t;
}

// ── Buttons ──────────────────────────────────────────────────────────────────

document.getElementById('btn-refresh').onclick = ()=>{
  segmentMode = true;
  document.getElementById('frame-title').textContent = 'Live Segmentation';
  drawImage();
  setStatus('refreshed — draw ROI around object, then click Set ROI');
};

document.getElementById('btn-set-roi').onclick = ()=>{
  if(!roi){ setStatus('draw ROI first'); return; }
  fetch('/train_set_roi', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(roi)
  }).then(r=>r.json()).then(d=>{
    if(!d.ok){ setStatus('error: '+d.error); return; }
    setSliders(d);   // populate sliders with auto-detected values
    segmentMode = true;
    document.getElementById('frame-title').textContent = 'Live Segmentation — adjust sliders';
    document.getElementById('coords').innerText =
      'Object segmented — adjust HSV sliders, then press [s] to save';
    drawImage();
    setStatus('ROI set — adjust sliders then save');
  });
};

function doSave(){
  fetch('/train_save', {method:'POST'})
    .then(r=>r.json()).then(d=>{
      if(d.ok){
        samples = d.count;
        document.getElementById('stats').innerText = 'Samples: ' + samples;
        setStatus('saved sample #' + samples);
      } else {
        setStatus('save error: ' + (d.error||'?'));
      }
    });
}

function doTrain(){
  const name = prompt('Output filename (no extension):');
  if(!name) return;
  setStatus('running DINOv2…');
  fetch('/train_run', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({filename:name})
  }).then(r=>r.json()).then(d=>{
    if(d.ok) setStatus('saved: ' + d.path + '  ('+d.count+' embeddings)');
    else setStatus('error: ' + (d.error||'?'));
  });
}

document.getElementById('btn-save').onclick  = doSave;
document.getElementById('btn-train').onclick = doTrain;

// ── Keyboard shortcuts ────────────────────────────────────────────────────────
document.addEventListener('keydown', e=>{
  if(e.target.tagName === 'INPUT') return;
  if(e.key === 's' || e.key === 'S') doSave();
  if(e.key === 't' || e.key === 'T') doTrain();
});

// Auto-refresh frame continuously so video keeps moving
setInterval(() => { drawImage(); }, 200);
window.onload = drawImage;

// Release the training camera when leaving the page
window.addEventListener('beforeunload', () => {
  navigator.sendBeacon('/train_close');
});
</script>
</body>
</html>
"""


PAGE2_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CountBot — Dashboard</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #1a1a1a;
    color: #e0e0e0;
    font-family: 'Segoe UI', system-ui, sans-serif;
    display: flex;
    height: 100vh;
    overflow: hidden;
  }

  /* ---- Sidebar ---- */
  #sidebar {
    width: 260px;
    min-width: 260px;
    background: #242424;
    display: flex;
    flex-direction: column;
    border-right: 1px solid #2e2e2e;
    overflow: hidden;
  }
  #sidebar-header {
    padding: 18px 16px 10px;
    border-bottom: 1px solid #2e2e2e;
  }
  #sidebar-header h2 {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: #888;
    margin-bottom: 4px;
  }
  #sidebar-header .folder-path {
    font-size: 0.75rem;
    color: #555;
    word-break: break-all;
  }
  #file-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px 0;
  }
  #file-list::-webkit-scrollbar { width: 4px; }
  #file-list::-webkit-scrollbar-thumb { background: #3a3a3a; border-radius: 2px; }
  .file-item {
    padding: 9px 16px;
    cursor: pointer;
    font-size: 0.82rem;
    color: #bbb;
    border-left: 3px solid transparent;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    transition: background .15s, color .15s;
  }
  .file-item:hover { background: #2c2c2c; color: #e0e0e0; }
  .file-item.active {
    background: #1e2e1e;
    border-left-color: #4caf50;
    color: #4caf50;
  }

  #counts-box {
    padding: 16px;
    border-top: 1px solid #2e2e2e;
    border-bottom: 1px solid #2e2e2e;
    display: flex;
    gap: 12px;
  }
  .count-card {
    flex: 1;
    background: #1a1a1a;
    border-radius: 8px;
    padding: 12px 8px;
    text-align: center;
  }
  .count-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: .06em;
    margin-bottom: 6px;
  }
  .count-label.good { color: #4caf50; }
  .count-label.bad  { color: #f44336; }
  .count-value {
    font-size: 1.8rem;
    font-weight: 700;
    line-height: 1;
  }
  .count-value.good { color: #4caf50; }
  .count-value.bad  { color: #f44336; }

  #sidebar-buttons {
    padding: 14px 16px;
    display: flex;
    gap: 10px;
  }
  .btn {
    flex: 1;
    padding: 10px 0;
    border: none;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
    transition: background .2s, opacity .2s;
  }
  .btn-train { background: #1976d2; color: #fff; }
  .btn-train:hover:not(:disabled) { background: #1565c0; }
  .btn-start { background: #4caf50; color: #fff; }
  .btn-start:hover:not(:disabled) { background: #43a047; }
  .btn-stop  { background: #f44336; color: #fff; }
  .btn-stop:hover:not(:disabled)  { background: #e53935; }
  .btn:disabled { opacity: .4; cursor: default; }

  /* ---- Main area ---- */
  #main {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  #top-bar {
    background: #242424;
    border-bottom: 1px solid #2e2e2e;
    padding: 12px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  #top-bar h1 { font-size: 1.1rem; font-weight: 600; color: #fff; flex: 1; }
  .status-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #555;
    transition: background .3s;
  }
  .status-dot.running { background: #4caf50; box-shadow: 0 0 6px #4caf50aa; }
  .status-text { font-size: 0.8rem; color: #888; }

  #content {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  /* ---- Sliders ---- */
  #sliders-card {
    background: #242424;
    border-radius: 10px;
    padding: 18px 20px;
  }
  #sliders-card h3 {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: #666;
    margin-bottom: 14px;
  }
  .sliders-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px 20px;
  }
  .slider-row {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .slider-row.full-width {
    grid-column: 1 / -1;
    margin-top: 6px;
    padding-top: 10px;
    border-top: 1px solid #333;
  }
  .slider-name {
    width: 90px;
    min-width: 90px;
    font-size: 0.82rem;
    color: #aaa;
    text-align: right;
  }
  .slider-wrap {
    flex: 1;
    position: relative;
    height: 20px;
    display: flex;
    align-items: center;
  }
  /* track background */
  .slider-wrap::before {
    content: '';
    position: absolute;
    left: 0; right: 0;
    height: 4px;
    background: #3a3a3a;
    border-radius: 2px;
    pointer-events: none;
  }
  /* filled portion (updated via JS) */
  .slider-fill {
    position: absolute;
    left: 0;
    height: 4px;
    background: #4caf50;
    border-radius: 2px;
    pointer-events: none;
    width: 0%;
  }
  input[type=range] {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 4px;
    background: transparent;
    outline: none;
    cursor: pointer;
    position: relative;
    z-index: 1;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px; height: 16px;
    border-radius: 50%;
    background: #fff;
    border: 2px solid #4caf50;
    box-shadow: 0 0 4px rgba(0,0,0,0.4);
    cursor: pointer;
  }
  input[type=range]::-moz-range-thumb {
    width: 16px; height: 16px;
    border-radius: 50%;
    background: #fff;
    border: 2px solid #4caf50;
    box-shadow: 0 0 4px rgba(0,0,0,0.4);
    cursor: pointer;
  }
  .slider-value {
    width: 42px;
    min-width: 42px;
    text-align: right;
    font-size: 0.82rem;
    color: #ffdc3c;
    font-variant-numeric: tabular-nums;
  }

  /* ---- Video ---- */
  #video-card {
    background: #242424;
    border-radius: 10px;
    overflow: hidden;
    flex: 1;
    min-height: 300px;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  #video-card img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
  }
</style>
</head>
<body>

<!-- SIDEBAR -->
<div id="sidebar">
  <div id="sidebar-header">
    <h2>Reference Embeddings</h2>
    <div class="folder-path">{{ folder }}</div>
  </div>

  <div id="file-list">
    {% for f in files %}
    <div class="file-item {% if f == reference_file %}active{% endif %}"
         data-name="{{ f }}"
         onclick="selectFile(this, '{{ f }}')">
      {{ f }}
    </div>
    {% endfor %}
    {% if not files %}
    <div style="padding:16px;color:#555;font-size:0.8rem;">No images found.</div>
    {% endif %}
  </div>

  <div id="counts-box">
    <div class="count-card">
      <div class="count-label good">Good</div>
      <div class="count-value good" id="count-good">0</div>
    </div>
    <div class="count-card">
      <div class="count-label bad">Bad</div>
      <div class="count-value bad" id="count-bad">0</div>
    </div>
  </div>

  <div id="sidebar-buttons">
    <button class="btn btn-train" id="btn-train" onclick="window.open('/train', '_blank')">Train</button>
    <button class="btn btn-start" id="btn-start" onclick="doStart()">Start</button>
    <button class="btn btn-stop"  id="btn-stop"  onclick="doStop()" disabled>Stop</button>
  </div>
</div>

<!-- MAIN -->
<div id="main">
  <div id="top-bar">
    <h1>CountBot Dashboard</h1>
    <div class="status-dot" id="status-dot"></div>
    <span class="status-text" id="status-text">Idle</span>
  </div>

  <div id="content">
    <!-- Sliders -->
    <div id="sliders-card">
      <h3>Background HSV Range</h3>
      <div class="sliders-grid">

        <div class="slider-row">
          <span class="slider-name">H Min</span>
          <div class="slider-wrap">
            <div class="slider-fill" id="fill-h_min"></div>
            <input type="range" id="sl-h_min" min="0" max="360" value="{{ sliders.h_min }}"
                   oninput="sliderChanged('h_min', this)">
          </div>
          <span class="slider-value" id="val-h_min">{{ sliders.h_min }}</span>
        </div>

        <div class="slider-row">
          <span class="slider-name">H Max</span>
          <div class="slider-wrap">
            <div class="slider-fill" id="fill-h_max"></div>
            <input type="range" id="sl-h_max" min="0" max="360" value="{{ sliders.h_max }}"
                   oninput="sliderChanged('h_max', this)">
          </div>
          <span class="slider-value" id="val-h_max">{{ sliders.h_max }}</span>
        </div>

        <div class="slider-row">
          <span class="slider-name">S Min</span>
          <div class="slider-wrap">
            <div class="slider-fill" id="fill-s_min"></div>
            <input type="range" id="sl-s_min" min="0" max="100" value="{{ sliders.s_min }}"
                   oninput="sliderChanged('s_min', this)">
          </div>
          <span class="slider-value" id="val-s_min">{{ sliders.s_min }}</span>
        </div>

        <div class="slider-row">
          <span class="slider-name">S Max</span>
          <div class="slider-wrap">
            <div class="slider-fill" id="fill-s_max"></div>
            <input type="range" id="sl-s_max" min="0" max="100" value="{{ sliders.s_max }}"
                   oninput="sliderChanged('s_max', this)">
          </div>
          <span class="slider-value" id="val-s_max">{{ sliders.s_max }}</span>
        </div>

        <div class="slider-row">
          <span class="slider-name">V Min</span>
          <div class="slider-wrap">
            <div class="slider-fill" id="fill-v_min"></div>
            <input type="range" id="sl-v_min" min="0" max="100" value="{{ sliders.v_min }}"
                   oninput="sliderChanged('v_min', this)">
          </div>
          <span class="slider-value" id="val-v_min">{{ sliders.v_min }}</span>
        </div>

        <div class="slider-row">
          <span class="slider-name">V Max</span>
          <div class="slider-wrap">
            <div class="slider-fill" id="fill-v_max"></div>
            <input type="range" id="sl-v_max" min="0" max="100" value="{{ sliders.v_max }}"
                   oninput="sliderChanged('v_max', this)">
          </div>
          <span class="slider-value" id="val-v_max">{{ sliders.v_max }}</span>
        </div>

        <div class="slider-row">
          <span class="slider-name">Cmp Thr</span>
          <div class="slider-wrap">
            <div class="slider-fill" id="fill-compare_threshold"></div>
            <input type="range" id="sl-compare_threshold" min="0" max="100" value="{{ sliders.compare_threshold }}"
                   oninput="sliderChanged('compare_threshold', this)">
          </div>
          <span class="slider-value" id="val-compare_threshold">{{ sliders.compare_threshold }}</span>
        </div>

        <!-- Line position & Min size — full width rows -->
        <div class="slider-row full-width">
          <span class="slider-name" style="color:#e8a020;">Line Pos %</span>
          <div class="slider-wrap">
            <div class="slider-fill" id="fill-line_pos" style="background:#e8a020;"></div>
            <input type="range" id="sl-line_pos" min="10" max="90" value="{{ sliders.line_pos }}"
                   oninput="sliderChanged('line_pos', this)">
          </div>
          <span class="slider-value" id="val-line_pos" style="color:#e8a020;">{{ sliders.line_pos }}</span>
        </div>

        <div class="slider-row full-width">
          <span class="slider-name" style="color:#7eb8e8;">Min Size</span>
          <div class="slider-wrap">
            <div class="slider-fill" id="fill-min_size" style="background:#7eb8e8;"></div>
            <input type="range" id="sl-min_size" min="0" max="5000" value="{{ sliders.min_size }}"
                   oninput="sliderChanged('min_size', this)">
          </div>
          <span class="slider-value" id="val-min_size" style="color:#7eb8e8;">{{ sliders.min_size }}</span>
        </div>

      </div>
    </div>

    <!-- Video feed -->
    <div id="video-card">
      <img src="/video_feed" alt="Live feed">
    </div>
  </div>
</div>

<script>
// ---- Slider logic ----
const SLIDER_MAX = { h_min: 360, h_max: 360, s_min: 100, s_max: 100, v_min: 100, v_max: 100, compare_threshold: 100, line_pos: 90, min_size: 5000 };
let sliderTimer = null;

function sliderChanged(name, el) {
  const val = parseInt(el.value, 10);
  const max = SLIDER_MAX[name];
  document.getElementById('val-' + name).textContent = val;
  // Update fill bar
  const fill = document.getElementById('fill-' + name);
  fill.style.width = (val / max * 100).toFixed(2) + '%';

  // Debounce POST to server
  clearTimeout(sliderTimer);
  sliderTimer = setTimeout(() => {
    const body = {};
    body[name] = val;
    fetch('/slider', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
  }, 80);
}

// Initialise fill bars on page load
(function initFills() {
  const names = ['h_min', 'h_max', 's_min', 's_max', 'v_min', 'v_max', 'compare_threshold', 'line_pos', 'min_size'];
  names.forEach(name => {
    const el  = document.getElementById('sl-' + name);
    const max = SLIDER_MAX[name];
    const fill = document.getElementById('fill-' + name);
    fill.style.width = (parseInt(el.value, 10) / max * 100).toFixed(2) + '%';
  });
})();

// ---- Reference file selection ----
function selectFile(el, name) {
  document.querySelectorAll('.file-item').forEach(i => i.classList.remove('active'));
  el.classList.add('active');
  fetch('/set_reference', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({filename: name})
  });
}

// ---- Start / Stop ----
function doStart() {
  fetch('/start', {method: 'POST'}).then(() => pollStatus());
}
function doStop() {
  fetch('/stop', {method: 'POST'});
}

// ---- Status polling ----
function pollStatus() {
  fetch('/status')
    .then(r => r.json())
    .then(data => {
      document.getElementById('count-good').textContent = data.good;
      document.getElementById('count-bad').textContent  = data.bad;

      const dot  = document.getElementById('status-dot');
      const txt  = document.getElementById('status-text');
      const btnStart = document.getElementById('btn-start');
      const btnStop  = document.getElementById('btn-stop');

      if (data.running) {
        dot.classList.add('running');
        txt.textContent   = 'Running';
        btnStart.disabled = true;
        btnStop.disabled  = false;
      } else {
        dot.classList.remove('running');
        txt.textContent   = 'Idle';
        btnStart.disabled = false;
        btnStop.disabled  = true;
      }
    })
    .catch(() => {});
}

setInterval(pollStatus, 800);
pollStatus();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def list_images(folder):
    if not folder or not os.path.isdir(folder):
        return []
    files = []
    for name in sorted(os.listdir(folder)):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTS:
            files.append(name)
    return files


@app.route("/", methods=["GET"])
def page1():
    with lock:
        folder = state["folder"]
        video_index = state["video_index"]
    return render_template_string(PAGE1_HTML, folder=folder, video_index=video_index)


@app.route("/setup", methods=["POST"])
def setup():
    folder = request.form.get("folder", "").strip()
    try:
        video_index = int(request.form.get("video_index", 0))
    except ValueError:
        video_index = 0
    with lock:
        state["folder"]      = folder
        state["video_index"] = video_index

    # Capture one frame and auto-detect HSV
    global preview_frame_bytes
    cap = cv2.VideoCapture(video_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    ret, frame = cap.read()
    cap.release()

    if ret:
        h_min, h_max, s_min, s_max, v_min, v_max = auto_detect_hsv(frame)
        with lock:
            state["h_min"] = h_min
            state["h_max"] = h_max
            state["s_min"] = s_min
            state["s_max"] = s_max
            state["v_min"] = v_min
            state["v_max"] = v_max
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        preview_frame_bytes = jpeg.tobytes()

    return redirect(url_for("preview"))


@app.route("/frame_preview")
def frame_preview():
    global preview_frame_bytes
    if preview_frame_bytes is None:
        placeholder = np.full((480, 640, 3), 36, dtype=np.uint8)
        cv2.putText(placeholder, "No frame captured", (120, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (140, 140, 140), 2)
        _, jpeg = cv2.imencode(".jpg", placeholder)
        data = jpeg.tobytes()
    else:
        data = preview_frame_bytes
    return Response(data, mimetype="image/jpeg")


@app.route("/preview", methods=["GET"])
def preview():
    with lock:
        h_min = state["h_min"]
        h_max = state["h_max"]
        s_min = state["s_min"]
        s_max = state["s_max"]
        v_min = state["v_min"]
        v_max = state["v_max"]
    return render_template_string(PREVIEW_HTML,
        h_min=h_min, h_max=h_max,
        s_min=s_min, s_max=s_max,
        v_min=v_min, v_max=v_max)


@app.route("/main", methods=["GET"])
def page2():
    with lock:
        folder     = state["folder"]
        ref_file   = state["reference_file"]
        sliders    = {
            "h_min":              state["h_min"],
            "h_max":              state["h_max"],
            "s_min":              state["s_min"],
            "s_max":              state["s_max"],
            "v_min":              state["v_min"],
            "v_max":              state["v_max"],
            "compare_threshold":  state["compare_threshold"],
            "line_pos":           state["line_pos"],
            "min_size":           state["min_size"],
        }
    files = list_reference_files(folder)
    return render_template_string(
        PAGE2_HTML,
        folder=folder,
        files=files,
        reference_file=ref_file,
        sliders=sliders,
    )


@app.route("/set_reference", methods=["POST"])
def set_reference():
    data = request.get_json(force=True, silent=True) or {}
    filename = data.get("filename", "")
    with lock:
        state["reference_file"] = filename
    return jsonify(ok=True)


@app.route("/start", methods=["POST"])
def start():
    global capture_thread, stop_event, frame_buffer, _train_cap
    with lock:
        if state["running"]:
            return jsonify(ok=True, message="already running")
        video_index = state["video_index"]
        state["running"] = True

    stop_event = threading.Event()
    with frame_lock:
        frame_buffer = None

    # Release training cap so capture_loop can own the camera exclusively
    with _train_cap_lock:
        if _train_cap is not None:
            _train_cap.release()
            _train_cap = None

    capture_thread = threading.Thread(
        target=capture_loop,
        args=(video_index, stop_event),
        daemon=True,
        name="capture_loop",
    )
    capture_thread.start()
    return jsonify(ok=True)


@app.route("/stop", methods=["POST"])
def stop():
    global capture_thread
    stop_event.set()
    if capture_thread is not None:
        capture_thread.join(timeout=3.0)
        capture_thread = None
    with lock:
        state["running"] = False
    return jsonify(ok=True)


@app.route("/slider", methods=["POST"])
def slider():
    data = request.get_json(force=True, silent=True) or {}
    allowed = {"h_min", "h_max", "s_min", "s_max", "v_min", "v_max", "compare_threshold", "line_pos", "min_size"}
    with lock:
        for key, val in data.items():
            if key in allowed:
                try:
                    state[key] = int(val)
                except (ValueError, TypeError):
                    pass
    return jsonify(ok=True)


@app.route("/status", methods=["GET"])
def status():
    with lock:
        return jsonify(
            good=state["good"],
            bad=state["bad"],
            running=state["running"],
        )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/train_close", methods=["POST"])
def train_close():
    global _train_cap
    with _train_cap_lock:
        if _train_cap is not None:
            _train_cap.release()
            _train_cap = None
    return "", 204


@app.route("/train", methods=["GET"])
def train():
    with lock:
        if state.get("running"):
            return "<p style='color:#f55'>Stop conveyor first.</p><p><a href='/main'>Back</a></p>"
        return render_template_string(PAGE_TRAIN_HTML)


@app.route("/train_frame")
def train_frame():
    global training_last_frame, _train_cap

    with lock:
        running = state["running"]
        idx = state.get("video_index", 0)

    if running:
        # Main capture loop owns the camera — read the frame it already captured
        with training_raw_lock:
            frame = training_raw_frame
        if frame is None:
            placeholder = np.full((480, 640, 3), 36, dtype=np.uint8)
            cv2.putText(placeholder, "Starting...", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (140, 140, 140), 2)
            _, jpeg = cv2.imencode('.jpg', placeholder)
            return Response(jpeg.tobytes(), mimetype='image/jpeg')
    else:
        # Keep a persistent cap so we don't open/close the camera on every request
        with _train_cap_lock:
            if _train_cap is None or not _train_cap.isOpened():
                _train_cap = cv2.VideoCapture(idx)
                _train_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                _train_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            ret, frame = _train_cap.read() if _train_cap is not None else (False, None)
        if not ret or frame is None:
            placeholder = np.full((480, 640, 3), 36, dtype=np.uint8)
            cv2.putText(placeholder, "No frame", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (140, 140, 140), 2)
            _, jpeg = cv2.imencode('.jpg', placeholder)
            return Response(jpeg.tobytes(), mimetype='image/jpeg')

    training_last_frame = frame.copy()
    _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return Response(jpeg.tobytes(), mimetype='image/jpeg')


@app.route("/train_set_roi", methods=["POST"])
def train_set_roi():
    global training_last_frame
    data = request.get_json(force=True, silent=True) or {}
    x = int(data.get('x', 0))
    y = int(data.get('y', 0))
    w = int(data.get('w', 0))
    h = int(data.get('h', 0))

    if training_last_frame is None or w <= 0 or h <= 0:
        return jsonify(ok=False, error='invalid roi')

    h0 = max(0, min(y, training_last_frame.shape[0] - 1))
    w0 = max(0, min(x, training_last_frame.shape[1] - 1))
    h1 = min(training_last_frame.shape[0], h0 + h)
    w1 = min(training_last_frame.shape[1], w0 + w)
    crop = training_last_frame[h0:h1, w0:w1]
    if crop.size == 0:
        return jsonify(ok=False, error='empty crop')

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:, :, 0].astype(np.int32)
    s_ch = hsv[:, :, 1].astype(np.int32)
    v_ch = hsv[:, :, 2].astype(np.int32)

    h_min = int(np.clip(np.percentile(h_ch, 5) / 179 * 360, 0, 360))
    h_max = int(np.clip(np.percentile(h_ch, 95) / 179 * 360, 0, 360))
    s_min = int(np.clip(np.percentile(s_ch, 5) / 255 * 100, 0, 100))
    s_max = int(np.clip(np.percentile(s_ch, 95) / 255 * 100, 0, 100))
    v_min = int(np.clip(np.percentile(v_ch, 5) / 255 * 100, 0, 100))
    v_max = int(np.clip(np.percentile(v_ch, 95) / 255 * 100, 0, 100))

    with lock:
        state['roi_rect'] = [x, y, w, h]
        state['h_min'] = h_min
        state['h_max'] = h_max
        state['s_min'] = s_min
        state['s_max'] = s_max
        state['v_min'] = v_min
        state['v_max'] = v_max

    return jsonify(ok=True, h_min=h_min, h_max=h_max, s_min=s_min, s_max=s_max, v_min=v_min, v_max=v_max)


@app.route("/train_segmented_frame")
def train_segmented_frame():
    global training_last_frame, _train_cap

    with lock:
        running = state["running"]
        idx = state.get("video_index", 0)

    # 1. Main loop running → use its raw frame
    if running:
        with training_raw_lock:
            frame = training_raw_frame
    else:
        frame = None

    # 2. Main loop not running → grab from persistent cap
    if frame is None:
        with _train_cap_lock:
            if _train_cap is None or not _train_cap.isOpened():
                _train_cap = cv2.VideoCapture(idx)
                _train_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                _train_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            ret, frame = _train_cap.read() if _train_cap is not None else (False, None)
        if not ret or frame is None:
            placeholder = np.full((480, 640, 3), 36, dtype=np.uint8)
            cv2.putText(placeholder, "No frame", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (140, 140, 140), 2)
            _, jpeg = cv2.imencode('.jpg', placeholder)
            return Response(jpeg.tobytes(), mimetype='image/jpeg')

    # Keep training_last_frame in sync so Save still works
    training_last_frame = frame.copy()

    with lock:
        h_min    = state["h_min"]
        h_max    = state["h_max"]
        s_min    = state["s_min"]
        s_max    = state["s_max"]
        v_min    = state["v_min"]
        v_max    = state["v_max"]
        min_size = state["min_size"]
        roi      = state.get("roi_rect")

    annotated, _mask, _contours = segment_frame(
        frame, h_min, h_max, s_min, s_max, v_min, v_max, min_size
    )
    if roi:
        rx, ry, rw, rh = roi
        cv2.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)

    _, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return Response(jpeg.tobytes(), mimetype='image/jpeg')


@app.route("/train_save", methods=["POST"])
def train_save():
    global training_last_frame, training_saved_rois
    with lock:
        roi      = state.get('roi_rect')
        h_min    = state["h_min"]
        h_max    = state["h_max"]
        s_min    = state["s_min"]
        s_max    = state["s_max"]
        v_min    = state["v_min"]
        v_max    = state["v_max"]
        min_size = state["min_size"]
        folder   = state.get('folder', '').strip()

    if training_last_frame is None or not roi:
        return jsonify(ok=False, error='no roi set or frame')

    x, y, w, h = roi
    crop = training_last_frame[y:y+h, x:x+w]
    if crop.size == 0:
        return jsonify(ok=False, error='empty crop')

    # Build segmented crop: apply contour mask so background is black
    annotated, mask, _contours = segment_frame(
        training_last_frame, h_min, h_max, s_min, s_max, v_min, v_max, min_size
    )
    mask_crop = mask[y:y+h, x:x+w]
    seg_crop = crop.copy()
    seg_crop[mask_crop == 0] = 0

    # Save annotated frame image to folder
    count = len(training_saved_rois)
    if folder and os.path.isdir(folder):
        img_path = os.path.join(folder, f'train_sample_{count:04d}.jpg')
        cv2.imwrite(img_path, annotated)

    training_saved_rois.append(seg_crop)
    return jsonify(ok=True, count=len(training_saved_rois))


@app.route("/train_run", methods=["POST"])
def train_run():
    data = request.get_json(force=True, silent=True) or {}
    name = data.get('filename', '').strip()
    if not name:
        return jsonify(ok=False, error='filename required')

    with lock:
        folder = state.get('folder', '').strip()
    if not folder or not os.path.isdir(folder):
        return jsonify(ok=False, error='invalid folder')

    if not training_saved_rois:
        return jsonify(ok=False, error='no saved samples')

    model, _ = ensure_dino_model()
    embeddings = {}
    for i, roi in enumerate(training_saved_rois):
        emb = get_embedding_from_roi(roi)
        if emb is not None:
            embeddings[f'sample_{i}'] = emb

    out_path = os.path.join(folder, f'{name}.pt')
    torch.save(embeddings, out_path)

    return jsonify(ok=True, path=out_path, count=len(embeddings))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)