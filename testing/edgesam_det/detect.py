#!/usr/bin/env python3
"""
EdgeSAM / MobileSAM — Segment Anything Detector (Conveyor Belt)
================================================================
What this script does:
  - Patches torchvision.ops.batched_nms with pure-PyTorch NMS so
    SamAutomaticMaskGenerator works on this Jetson (broken C++ ext)
  - Loads MobileSAM (vit_t) on GPU
  - Uses SamAutomaticMaskGenerator — segments ALL objects automatically
    in one GPU pass per frame (true "segment anything" approach)
  - Runs SAM every SAM_SKIP frames in a background thread
  - Computes motion mask to ignore the static black conveyor belt
  - Filters oversized / dark bounding boxes (conveyor background)
  - Removes child bounding boxes (boxes fully inside a larger box)
  - Keeps only detections with predicted_iou >= CONF_THRESHOLD
  - Overlays FPS + frame count + detection count on output
  - Saves output video + shows live preview (Q to quit)

Tunable constants:
  CONF_THRESHOLD        — min SAM predicted_iou  (0.0–1.0)
  STAB_THRESHOLD        — min stability score    (0.0–1.0)
  SAM_SKIP              — run SAM every N frames
  MOTION_DIFF_THRESH    — pixel diff to count as motion (0-255)
  MIN_BOX_AREA          — min bounding box area in pixels
  MAX_BOX_AREA_RATIO    — max box area as fraction of frame
  MAX_BOX_WIDTH_RATIO   — max box width fraction
  MAX_BOX_HEIGHT_RATIO  — max box height fraction
  DARK_ROI_THRESH       — mean brightness below = conveyor, skip
  MOTION_OVERLAP_THRESH — min fraction of box inside motion region
"""

import os

# Must be before torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import sys
import threading
import cv2
import numpy as np
import torch

torch.backends.cudnn.benchmark = False

# ── Pure-PyTorch NMS — patches the broken C++ torchvision op ─────────────────
# SamAutomaticMaskGenerator calls torchvision.ops.batched_nms internally.
# On this Jetson the C++ torchvision extension is broken (undefined symbol).
# We replace it everywhere before mobile_sam is imported.

def _pt_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)
    keep  = []
    while order.numel() > 0:
        i = int(order[0]);  keep.append(i)
        if order.numel() == 1: break
        rest = order[1:]
        inter = ((x2[rest].clamp(max=float(x2[i])) - x1[rest].clamp(min=float(x1[i]))).clamp(0) *
                 (y2[rest].clamp(max=float(y2[i])) - y1[rest].clamp(min=float(y1[i]))).clamp(0))
        iou   = inter / (areas[i] + areas[rest] - inter)
        order = rest[(iou <= iou_threshold).nonzero(as_tuple=False).squeeze(1)]
    return torch.tensor(keep, dtype=torch.long)

def _batched_nms(boxes: torch.Tensor, scores: torch.Tensor,
                 idxs: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    offsets = idxs.to(boxes.dtype) * (boxes.max() + 1)
    return _pt_nms(boxes + offsets[:, None], scores, iou_threshold)

import torchvision.ops       as _tvops
import torchvision.ops.boxes as _tvboxes
_tvops.batched_nms   = _batched_nms
_tvboxes.batched_nms = _batched_nms
print("[init] patched torchvision.batched_nms → pure-PyTorch NMS")

# Now safe to import mobile_sam
try:
    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
except ImportError:
    print("[error] mobile_sam not installed.  pip install mobile_sam")
    sys.exit(1)

# Patch inside mobile_sam's own amg module (it imports batched_nms by name)
try:
    import mobile_sam.utils.amg as _amg
    _amg.batched_nms = _batched_nms
    print("[init] patched mobile_sam.utils.amg.batched_nms")
except Exception:
    pass

# ── Tunable thresholds ──────────────────────────────────────────────────────
SOURCE                = "/home/vikbot/Documents/countbot/testing/video/bearing_wheel.mp4"
OUTPUT                = "/home/vikbot/Documents/countbot/testing/edgesam_output.mp4"
SAM_CHECKPOINT        = "weights/mobile_sam.pt"
SAM_TYPE              = "vit_t"

CONF_THRESHOLD        = 0.85   # SAM predicted_iou gate
STAB_THRESHOLD        = 0.85   # stability score gate
SAM_SKIP              = 3      # run SAM every N frames (background thread)
SAM_POINTS_PER_SIDE   = 8     # grid density: 8×8=64 pts (default 32²=1024 → Jetson OOM)
SAM_INPUT_SIZE        = (320, 240)  # resize frame before SAM to cut GPU memory ~4×
MOTION_DIFF_THRESH    = 30     # frame-diff to count as motion
MIN_BOX_AREA          = 200    # ignore boxes smaller than this (px² at SAM resolution)
MAX_BOX_AREA_RATIO    = 0.50   # ignore boxes > 50% of frame
MAX_BOX_WIDTH_RATIO   = 0.70   # ignore boxes > 70% of frame width
MAX_BOX_HEIGHT_RATIO  = 0.70   # ignore boxes > 70% of frame height
DARK_ROI_THRESH       = 40     # ignore boxes whose interior mean < this
MOTION_OVERLAP_THRESH = 0.20   # min 20% of box must overlap motion region
# ────────────────────────────────────────────────────────────────────────────


# ── Model loader ──────────────────────────────────────────────────────────────

def load_generator():
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    abs_ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), SAM_CHECKPOINT)
    if not os.path.isfile(abs_ckpt):
        print(f"[error] checkpoint not found: {abs_ckpt}")
        sys.exit(1)

    sam = sam_model_registry[SAM_TYPE](checkpoint=abs_ckpt)
    sam.to(device).eval()

    gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side        = SAM_POINTS_PER_SIDE,   # 8×8=64 vs default 32²=1024
        pred_iou_thresh        = CONF_THRESHOLD,
        stability_score_thresh = STAB_THRESHOLD,
        min_mask_region_area   = MIN_BOX_AREA,
    )
    print(f"[init] MobileSAM {SAM_TYPE} → SamAutomaticMaskGenerator on {device} "
          f"(points_per_side={SAM_POINTS_PER_SIDE}, input={SAM_INPUT_SIZE})")
    return gen, device


# ── Child-box removal ─────────────────────────────────────────────────────────

def remove_child_boxes(bboxes):
    if len(bboxes) <= 1:
        return bboxes
    keep = [True] * len(bboxes)
    for i, (ax1, ay1, ax2, ay2, _) in enumerate(bboxes):
        if not keep[i]: continue
        for j, (bx1, by1, bx2, by2, _) in enumerate(bboxes):
            if i == j or not keep[i]: continue
            if bx1 <= ax1 and by1 <= ay1 and bx2 >= ax2 and by2 >= ay2:
                keep[i] = False; break
    return [b for b, k in zip(bboxes, keep) if k]


# ── Background SAM worker ─────────────────────────────────────────────────────

class SamWorker:
    def __init__(self, gen):
        self.gen      = gen
        self._lock    = threading.Lock()
        self._result  = []
        self._pending = None
        self._running = False
        self._ready   = threading.Event()

    @property
    def busy(self): return self._running

    def get_result(self):
        with self._lock: return list(self._result)

    def wait_first(self, timeout=60):
        self._ready.wait(timeout=timeout)

    def submit(self, frame_rgb, motion_mask):
        if self._running:
            with self._lock: self._pending = (frame_rgb, motion_mask)
            return
        with self._lock: self._pending = None
        self._running = True
        threading.Thread(target=self._run, args=(frame_rgb, motion_mask), daemon=True).start()

    def _run(self, frame_rgb, motion_mask):
        # display resolution
        disp_h, disp_w = frame_rgb.shape[:2]

        # Downsample for SAM — cuts GPU memory ~4× vs 640×480
        sam_w, sam_h = SAM_INPUT_SIZE
        sam_frame = cv2.resize(
            cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), (sam_w, sam_h)
        )
        sam_rgb   = cv2.cvtColor(sam_frame, cv2.COLOR_BGR2RGB)
        sam_gray  = cv2.cvtColor(sam_frame, cv2.COLOR_BGR2GRAY)
        sam_area  = sam_h * sam_w

        # Scale factors to map SAM boxes → display coords
        sx = disp_w / sam_w
        sy = disp_h / sam_h

        # Downsample motion mask to SAM resolution for overlap test
        motion_sam = None
        if motion_mask is not None:
            motion_sam = cv2.resize(motion_mask, (sam_w, sam_h),
                                    interpolation=cv2.INTER_NEAREST)

        print("\n[thread] SAM generate() ...", end=" ", flush=True)
        try:
            with torch.inference_mode():
                raw = self.gen.generate(sam_rgb)
            torch.cuda.empty_cache()
            print(f"{len(raw)} masks")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback; traceback.print_exc()
            raw = []

        results = []
        n_conf = n_area = n_size = n_motion = n_dark = 0
        for m in raw:
            if m["predicted_iou"]   < CONF_THRESHOLD:  n_conf += 1; continue
            if m["stability_score"] < STAB_THRESHOLD:  n_conf += 1; continue

            # bbox is in SAM (downsampled) coords
            bx, by, bw, bh = [int(v) for v in m["bbox"]]
            area = bw * bh
            if area < MIN_BOX_AREA or area > sam_area * MAX_BOX_AREA_RATIO: n_area += 1; continue
            if bw > sam_w * MAX_BOX_WIDTH_RATIO or bh > sam_h * MAX_BOX_HEIGHT_RATIO: n_size += 1; continue

            if motion_sam is not None:
                roi_m = motion_sam[by:by+bh, bx:bx+bw]
                if roi_m.size > 0 and np.count_nonzero(roi_m) / roi_m.size < MOTION_OVERLAP_THRESH:
                    n_motion += 1; continue

            roi_g = sam_gray[by:by+bh, bx:bx+bw]
            if roi_g.size > 0 and roi_g.mean() < DARK_ROI_THRESH: n_dark += 1; continue

            # Scale bbox up to display resolution
            x1 = int(bx * sx);  y1 = int(by * sy)
            x2 = int((bx+bw) * sx); y2 = int((by+bh) * sy)
            results.append((x1, y1, x2, y2, float(m["predicted_iou"])))

        results = remove_child_boxes(results)
        print(f"[filter] conf:{n_conf} area:{n_area} size:{n_size} "
              f"motion:{n_motion} dark:{n_dark} → kept:{len(results)}")

        with self._lock:
            self._result = results
            pending = self._pending
            self._pending = None

        self._ready.set()
        self._running = False

        if pending is not None:
            self._running = True
            threading.Thread(target=self._run, args=pending, daemon=True).start()


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw(frame, bboxes):
    out = frame.copy()
    for x1, y1, x2, y2, conf in bboxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{conf:.2f}", (x1, max(y1 - 8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    gen, device = load_generator()
    sam = SamWorker(gen)

    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print(f"[error] cannot open: {SOURCE}"); sys.exit(1)

    src_fps      = cap.get(cv2.CAP_PROP_FPS)
    src_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[init] {src_w}x{src_h} @ {src_fps:.1f}fps  {total_frames} frames  device={device}")
    print(f"[init] conf={CONF_THRESHOLD}  stab={STAB_THRESHOLD}  skip={SAM_SKIP}")

    writer = None
    if OUTPUT:
        writer = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (640, 480))
        print(f"[init] saving → {OUTPUT}")

    # Warm-up: submit first frame and wait for result before starting display
    ret, frame = cap.read()
    if not ret:
        print("[error] empty video"); sys.exit(1)
    frame = cv2.resize(frame, (640, 480))
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sam.submit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None)
    print(f"[run] waiting for first SAM result (input={SAM_INPUT_SIZE} points={SAM_POINTS_PER_SIDE}²={SAM_POINTS_PER_SIDE**2}) ...")
    sam.wait_first(timeout=120)

    frame_count = 1
    fps_counter = 0
    tick_start  = cv2.getTickCount()
    print("[run] press Q to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame_count += 1
            frame = cv2.resize(frame, (640, 480))
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Motion mask
            diff = cv2.absdiff(prev_gray, gray)
            _, motion_mask = cv2.threshold(diff, MOTION_DIFF_THRESH, 255, cv2.THRESH_BINARY)
            motion_mask = cv2.morphologyEx(
                motion_mask, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            prev_gray = gray.copy()

            if frame_count % SAM_SKIP == 0:
                sam.submit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), motion_mask)

            bboxes = sam.get_result()
            out    = draw(frame, bboxes)

            fps_counter += 1
            elapsed  = (cv2.getTickCount() - tick_start) / cv2.getTickFrequency()
            live_fps = fps_counter / elapsed if elapsed > 0 else 0.0

            cv2.putText(out, f"FPS: {live_fps:.1f}",                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(out, f"Frame: {frame_count}/{total_frames}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(out, f"Det: {len(bboxes)}",                  (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            if sam.busy:
                cv2.putText(out, "SAM...", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            print(f"Frame {frame_count}/{total_frames}  FPS={live_fps:.1f}  Det={len(bboxes)}", end="\r")

            if writer: writer.write(out)
            cv2.imshow("EdgeSAM — Segment Anything", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[run] stopped"); break

    finally:
        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()
        print(f"\n[done] {frame_count} frames")


if __name__ == "__main__":
    main()
