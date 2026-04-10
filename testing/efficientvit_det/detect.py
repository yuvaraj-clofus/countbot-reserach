#!/usr/bin/env python3
"""
EfficientViT-SAM Object Detector — Conveyor Belt Edition
=========================================================
What this script does:
  - Loads EfficientViT-SAM (l0) on GPU with Jetson memory fixes
  - Patches torchvision.ops.batched_nms with pure-PyTorch NMS to fix
    the broken C++ torchvision extension on this Jetson
  - Runs SAM in a background thread so display loop never blocks
  - Reads video frame-by-frame, resizes to 640x480
  - Computes motion mask to ignore static black conveyor belt
  - Filters oversized / dark bounding boxes (conveyor background)
  - Removes child bounding boxes (boxes fully inside a larger box)
  - Keeps only detections with predicted_iou >= CONF_THRESHOLD
  - Overlays FPS + frame count + detection count on output

Tunable constants:
  CONF_THRESHOLD        — minimum SAM predicted_iou to keep a detection
  STAB_THRESHOLD        — minimum stability score to keep a detection
  SAM_SKIP              — run SAM every N frames
  MOTION_DIFF_THRESH    — pixel diff to count as motion (0-255)
  MIN_BOX_AREA          — minimum bounding box area in pixels
  MAX_BOX_AREA_RATIO    — max box area as fraction of frame
  MAX_BOX_WIDTH_RATIO   — max box width as fraction of frame width
  MAX_BOX_HEIGHT_RATIO  — max box height as fraction of frame height
  DARK_ROI_THRESH       — mean brightness below = conveyor box, skip
  MOTION_OVERLAP_THRESH — min fraction of box area inside motion region
"""

import os

# ── Must be set BEFORE torch is imported ─────────────────────────────────────
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
# ─────────────────────────────────────────────────────────────────────────────

import sys
import threading
import cv2
import numpy as np
import torch

torch.backends.cudnn.benchmark = False

# ── Patch torchvision batched_nms BEFORE efficientvit is imported ─────────────
# The C++ torchvision extension is broken on this Jetson (undefined symbol).
# EfficientViT's mask generator internally calls batched_nms via:
#   from torchvision.ops.boxes import batched_nms
# We must patch both the module attribute AND the boxes submodule before any
# efficientvit import so all code paths get the pure-PyTorch version.

def _pytorch_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)
    keep  = []
    while order.numel() > 0:
        i = int(order[0])
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        ix1  = x1[rest].clamp(min=float(x1[i]))
        iy1  = y1[rest].clamp(min=float(y1[i]))
        ix2  = x2[rest].clamp(max=float(x2[i]))
        iy2  = y2[rest].clamp(max=float(y2[i]))
        inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
        iou   = inter / (areas[i] + areas[rest] - inter)
        order = rest[(iou <= iou_threshold).nonzero(as_tuple=False).squeeze(1)]
    return torch.tensor(keep, dtype=torch.long)

def _batched_nms_pure(boxes: torch.Tensor, scores: torch.Tensor,
                      idxs: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    offsets   = idxs.to(boxes.dtype) * (boxes.max() + 1)
    boxes_off = boxes + offsets[:, None]
    return _pytorch_nms(boxes_off, scores, iou_threshold)

# Patch at every level torchvision exposes it
import torchvision.ops as _tvops
import torchvision.ops.boxes as _tvboxes
_tvops.batched_nms   = _batched_nms_pure
_tvboxes.batched_nms = _batched_nms_pure
print("[init] patched torchvision batched_nms → pure-PyTorch NMS")
# ─────────────────────────────────────────────────────────────────────────────

# ── Tunable thresholds ──────────────────────────────────────────────────────
SOURCE                = "/home/vikbot/Documents/countbot/testing/video/bearing_wheel.mp4"
SAM_MODEL             = "efficientvit-sam-l0"
SAM_WEIGHTS           = "weights/efficientvit_sam_l0.pt"

CONF_THRESHOLD        = 0.50   # predicted_iou gate
STAB_THRESHOLD        = 0.50   # stability score gate
SAM_SKIP              = 3      # run SAM every N frames
MOTION_DIFF_THRESH    = 30     # frame-diff threshold to count as motion
MIN_BOX_AREA          = 500    # ignore boxes smaller than this (px²)
MAX_BOX_AREA_RATIO    = 0.50   # ignore boxes > 50% of frame area
MAX_BOX_WIDTH_RATIO   = 0.70   # ignore boxes > 70% of frame width
MAX_BOX_HEIGHT_RATIO  = 0.70   # ignore boxes > 70% of frame height
DARK_ROI_THRESH       = 40     # ignore boxes whose interior mean < this
MOTION_OVERLAP_THRESH = 0.20   # min 20% of box must overlap motion region
# ────────────────────────────────────────────────────────────────────────────


# ── Model loader ──────────────────────────────────────────────────────────────

def load_mask_generator():
    try:
        from efficientvit.sam_model_zoo import create_efficientvit_sam_model
        from efficientvit.models.efficientvit.sam import EfficientViTSamAutomaticMaskGenerator
    except ImportError:
        print("[error] efficientvit not installed.")
        print("  git clone https://github.com/mit-han-lab/efficientvit /tmp/efficientvit")
        print("  pip install -e /tmp/efficientvit")
        sys.exit(1)

    # After efficientvit is imported it may have captured the original batched_nms
    # inside its own module namespace — patch those too.
    try:
        import segment_anything.utils.amg as _amg
        _amg.batched_nms = _batched_nms_pure
        print("[init] patched segment_anything.utils.amg.batched_nms")
    except Exception:
        pass
    try:
        import mobile_sam.utils.amg as _mamg
        _mamg.batched_nms = _batched_nms_pure
        print("[init] patched mobile_sam.utils.amg.batched_nms")
    except Exception:
        pass

    weights_abs = os.path.join(os.path.dirname(os.path.abspath(__file__)), SAM_WEIGHTS)
    kwargs = {}
    if os.path.isfile(weights_abs):
        kwargs["weight_url"] = weights_abs
        print(f"[init] using local weights: {weights_abs}")
    else:
        print("[init] weights not found locally — will auto-download")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = create_efficientvit_sam_model(SAM_MODEL, pretrained=True, **kwargs)
    model.eval().to(device)
    print(f"[init] EfficientViT-SAM loaded on {device}")

    return EfficientViTSamAutomaticMaskGenerator(
        model,
        pred_iou_thresh=CONF_THRESHOLD,
        stability_score_thresh=STAB_THRESHOLD,
    ), device


# ── Child-box removal ─────────────────────────────────────────────────────────

def remove_child_boxes(bboxes):
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


# ── Background SAM worker ─────────────────────────────────────────────────────

class SamWorker:
    def __init__(self, mask_gen):
        self.mask_gen  = mask_gen
        self._lock     = threading.Lock()
        self._result   = []
        self._pending  = None
        self._running  = False
        self._done_evt = threading.Event()  # fires after each completed run

    @property
    def busy(self):
        return self._running

    def get_result(self):
        with self._lock:
            return list(self._result)

    def wait_first(self, timeout=60):
        """Block until first SAM result arrives (or timeout seconds)."""
        self._done_evt.wait(timeout=timeout)

    def submit(self, frame_rgb, motion_mask):
        if self._running:
            with self._lock:
                self._pending = (frame_rgb, motion_mask)
            return
        with self._lock:
            self._pending = None
        self._running = True
        threading.Thread(target=self._run, args=(frame_rgb, motion_mask), daemon=True).start()

    def _run(self, frame_rgb, motion_mask):
        height, width = frame_rgb.shape[:2]
        frame_area    = height * width
        frame_gray    = cv2.cvtColor(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)

        print("\n[thread] generate() starting ...")
        try:
            raw_masks = self.mask_gen.generate(frame_rgb)
            print(f"[thread] generate() done — {len(raw_masks)} raw masks")
        except Exception as e:
            print(f"\n[thread] generate() FAILED: {e}")
            import traceback; traceback.print_exc()
            raw_masks = []

        if raw_masks:
            ious  = [round(m["predicted_iou"],   3) for m in raw_masks[:5]]
            stabs = [round(m["stability_score"], 3) for m in raw_masks[:5]]
            print(f"[sam] iou sample={ious}  stab sample={stabs}")

        n_conf = n_area = n_size = n_motion = n_dark = 0
        results = []
        for m in raw_masks:
            if m["predicted_iou"]   < CONF_THRESHOLD:  n_conf   += 1; continue
            if m["stability_score"] < STAB_THRESHOLD:  n_conf   += 1; continue
            bx, by, bw, bh = [int(v) for v in m["bbox"]]
            x1, y1, x2, y2 = bx, by, bx + bw, by + bh
            area = bw * bh
            if area < MIN_BOX_AREA or area > frame_area * MAX_BOX_AREA_RATIO: n_area   += 1; continue
            if bw > width * MAX_BOX_WIDTH_RATIO or bh > height * MAX_BOX_HEIGHT_RATIO: n_size += 1; continue
            if motion_mask is not None:
                roi_m = motion_mask[y1:y2, x1:x2]
                if roi_m.size > 0 and np.count_nonzero(roi_m) / roi_m.size < MOTION_OVERLAP_THRESH:
                    n_motion += 1; continue
            roi_g = frame_gray[y1:y2, x1:x2]
            if roi_g.size > 0 and roi_g.mean() < DARK_ROI_THRESH: n_dark += 1; continue
            results.append((x1, y1, x2, y2, float(m["predicted_iou"])))

        print(f"[filter] dropped conf:{n_conf} area:{n_area} size:{n_size} "
              f"motion:{n_motion} dark:{n_dark} | kept:{len(results)}")

        results = remove_child_boxes(results)

        with self._lock:
            self._result  = results
            pending       = self._pending
            self._pending = None

        self._done_evt.set()   # signal first result ready
        self._running = False

        if pending is not None:
            frame_rgb, motion_mask = pending
            self._running = True
            threading.Thread(target=self._run, args=(frame_rgb, motion_mask), daemon=True).start()


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw(frame, bboxes):
    out = frame.copy()
    for x1, y1, x2, y2, conf in bboxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{conf:.2f}", (x1, max(y1 - 8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return out


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    mask_gen, device = load_mask_generator()
    sam = SamWorker(mask_gen)

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print(f"[error] cannot open: {SOURCE}")
        sys.exit(1)

    src_fps      = cap.get(cv2.CAP_PROP_FPS)
    src_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[init] source: {src_w}x{src_h} @ {src_fps:.1f} FPS  |  {total_frames} frames")
    print(f"[init] conf={CONF_THRESHOLD}  stab={STAB_THRESHOLD}  "
          f"skip={SAM_SKIP}fr  motion={MOTION_DIFF_THRESH}  device={device}")
    print("[run] press Q to quit")

    frame_count = 0
    prev_gray   = None
    fps_counter = 0
    tick_start  = cv2.getTickCount()

    # Read and submit first frame immediately so SAM warms up
    ret, frame = cap.read()
    if ret:
        frame_count = 1
        frame = cv2.resize(frame, (640, 480))
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sam.submit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None)
        print("[run] waiting for first SAM result (may take a few seconds on first run) ...")
        sam.wait_first(timeout=120)
        print("[run] first result ready — starting display loop")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame = cv2.resize(frame, (640, 480))
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            motion_mask = None
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                _, motion_mask = cv2.threshold(diff, MOTION_DIFF_THRESH, 255, cv2.THRESH_BINARY)
                motion_mask = cv2.morphologyEx(
                    motion_mask, cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                )
            prev_gray = gray.copy()

            if frame_count % SAM_SKIP == 0:
                sam.submit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), motion_mask)

            bboxes = sam.get_result()
            out    = draw(frame, bboxes)

            fps_counter += 1
            elapsed  = (cv2.getTickCount() - tick_start) / cv2.getTickFrequency()
            live_fps = fps_counter / elapsed if elapsed > 0 else 0.0

            cv2.putText(out, f"FPS: {live_fps:.1f}",       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(out, f"Frame: {frame_count}/{total_frames}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(out, f"Det: {len(bboxes)}",         (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            if sam.busy:
                cv2.putText(out, "SAM...", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            print(f"Frame {frame_count}/{total_frames}  FPS={live_fps:.1f}  Det={len(bboxes)}", end="\r")

            cv2.imshow("EfficientViT-SAM Detection", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[run] stopped by user")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[done] processed {frame_count} frames")


if __name__ == "__main__":
    main()
