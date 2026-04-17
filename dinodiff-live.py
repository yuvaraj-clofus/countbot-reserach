#!/usr/bin/env python3
"""
dinodiff.py — Conveyor Object Detection using DINOv2 (Reference-Based Anomaly Segmentation)

Approach (from dinotask.md):
    1. Capture first N frames when belt is empty → reference embeddings
    2. Extract DINOv2 patch embeddings per frame at ~640×480 (divisible by 14)
    3. Patch-wise cosine similarity against reference bank (top-k mean)
    4. Threshold to produce binary mask / similarity heatmap

Controls:  q=quit  r=recapture reference  d=toggle debug stats
"""

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

REF_FRAMES    = 40    # number of initial frames to capture as reference
TOPK          = 3     # top-k references for similarity (reduces flicker)
THRESH        = 0.4   # cosine-distance threshold for anomaly detection
MIN_AREA      = 200   # minimum contour area to keep
LINE_X_FRAC   = 0.85  # counting line position as fraction of frame width
TEMPORAL_ALPHA = 0.7   # temporal smoothing: weight for current mask vs prev

# ROI crop fractions (run DINO only on conveyor region)
ROI_LEFT  = 0.2
ROI_RIGHT = 0.8

# Debug stats: print similarity stats every N frames
DEBUG_STATS_EVERY = 30


# ── Load DINOv2 ──────────────────────────────────────────────────────────────

print(f"[DINO] Loading {DINO_MODEL} on {DEVICE}...")
model = torch.hub.load("facebookresearch/dinov2", DINO_MODEL, verbose=False)
model.eval().to(DEVICE)
print("[DINO] Ready.")

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
        tokens = extract_patch_tokens(roi)  # (N, D)
        tokens = F.normalize(tokens, dim=1)
        all_tokens.append(tokens)
        print(f"  [REF] Frame {i+1}/{len(rois)} embedded")
    # Stack: (num_frames, N, D) -> permute to (N, num_frames, D)
    bank = torch.stack(all_tokens, dim=0).permute(1, 0, 2)
    return bank


@torch.no_grad()
def compute_anomaly_map(live_tokens, ref_bank):
    """
    Compare live patch tokens against the reference bank.

    For each patch, compute top-k mean cosine similarity to reference frames.
    Returns:
        sim_map:  (grid_h, grid_w)  similarity values [0, 1]
        diff_map: (grid_h, grid_w)  anomaly score = 1 - sim
    """
    grid_h, grid_w = get_patch_grid_size()

    # Ensure live tokens are normalized
    live_tokens = F.normalize(live_tokens, dim=1)

    # live_tokens: (N, D), ref_bank: (N, num_refs, D)
    # (N, 1, D) * (N, num_refs, D) -> sum over D -> (N, num_refs)
    sims = torch.sum(live_tokens.unsqueeze(1) * ref_bank, dim=2)  # (N, num_refs)

    # Top-k mean instead of max — reduces flickering
    k = min(TOPK, sims.shape[1])
    topk_vals, _ = sims.topk(k=k, dim=1)
    mean_sims = topk_vals.mean(dim=1)  # (N,)

    sim_map = mean_sims.reshape(grid_h, grid_w).cpu().numpy()
    diff_map = 1.0 - sim_map

    return sim_map, diff_map


# ── Visualization ─────────────────────────────────────────────────────────────

def make_heatmap(diff_map, orig_h, orig_w, thresh):
    """Convert anomaly diff_map to a color heatmap anchored to thresh.
    Below thresh → blue/green (belt), above thresh → red (anomaly)."""
    norm = np.clip(diff_map / (2.0 * max(thresh, 1e-6)), 0, 1)
    heatmap = (norm * 255).astype(np.uint8)
    heatmap = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(GST_PIPELINE, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print(f"ERROR: Cannot open GStreamer pipeline: {GST_PIPELINE}")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    grid_h, grid_w = get_patch_grid_size()
    line_x = int(W * LINE_X_FRAC)

    # ROI pixel bounds
    roi_x0 = int(ROI_LEFT * W)
    roi_x1 = int(ROI_RIGHT * W)
    roi_w = roi_x1 - roi_x0

    print(f"Video : {W}x{H} @ {fps:.1f} fps")
    print(f"ROI   : x=[{roi_x0}, {roi_x1}]  ({roi_w}px wide)")
    print(f"DINO input : {TARGET_W}x{TARGET_H}  →  patch grid {grid_w}x{grid_h}")
    print(f"Ref frames : {REF_FRAMES}")
    print(f"Top-k      : {TOPK}")
    print(f"Threshold  : {THRESH}")

    # ── Step 1: Capture reference frames (empty belt ROIs) ────────────────────
    print(f"\n[REF] Capturing first {REF_FRAMES} frames as empty-belt reference...")
    ref_rois = []
    while len(ref_rois) < REF_FRAMES:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Video ended before capturing enough reference frames")
            return
        ref_rois.append(frame[:, roi_x0:roi_x1])

    ref_bank = build_reference_bank(ref_rois)
    print(f"[REF] Bank ready: {ref_bank.shape}  (patches x refs x dim)\n")

    # ── Threshold slider ───────────────────────────────────────────────────────
    win_name = "DINOv2 Anomaly Detection"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    thresh_val = [int(THRESH * 100)]

    def on_thresh(val):
        thresh_val[0] = val

    cv2.createTrackbar("Threshold", win_name, thresh_val[0], 100, on_thresh)

    # ── Step 2: Process video ─────────────────────────────────────────────────
    count = 0
    prev_centroid_x = None
    debug_mode = False
    prev_mask = None
    frame_n = 0
    fps_display = 0.0
    fps_timer = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_n += 1

        # FPS calculation
        now = cv2.getTickCount()
        fps_display = cv2.getTickFrequency() / (now - fps_timer)
        fps_timer = now

        # Crop ROI — run DINO only on conveyor region
        roi = frame[:, roi_x0:roi_x1]

        # Extract live features and compare
        live_tokens = extract_patch_tokens(roi)
        sim_map, diff_map = compute_anomaly_map(live_tokens, ref_bank)

        # Upscale diff_map to ROI size smoothly, then threshold
        cur_thresh = thresh_val[0] / 100.0
        diff_roi = cv2.resize(diff_map, (roi_w, H), interpolation=cv2.INTER_LINEAR)
        mask_roi = (diff_roi > cur_thresh).astype(np.uint8) * 255

        # Light morphology cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel, iterations=1)

        # Temporal smoothing
        mask_float = mask_roi.astype(np.float32)
        if prev_mask is not None:
            mask_float = TEMPORAL_ALPHA * mask_float + (1.0 - TEMPORAL_ALPHA) * prev_mask
        prev_mask = mask_float
        mask_roi = (mask_float > 127).astype(np.uint8) * 255

        # Map mask back to full frame
        mask_full = np.zeros((H, W), dtype=np.uint8)
        mask_full[:, roi_x0:roi_x1] = mask_roi

        # Find contours on full-frame mask
        contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        display = frame.copy()

        # Draw ROI bounds
        cv2.rectangle(display, (roi_x0, 0), (roi_x1, H), (100, 100, 100), 1)

        # Draw counting line
        cv2.line(display, (line_x, 0), (line_x, H), (255, 0, 0), 2)

        # Draw bounding boxes around ALL valid contours
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA]

        for cnt in valid_contours:
            rect = cv2.minAreaRect(cnt)
            box_pts = cv2.boxPoints(rect).astype(int)
            cv2.drawContours(display, [box_pts], 0, (0, 255, 0), 2)

            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)

        # Line crossing uses the largest blob's centroid
        best = max(valid_contours, key=cv2.contourArea) if valid_contours else None

        if best is not None:
            M = cv2.moments(best)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])

                if prev_centroid_x is not None:
                    if prev_centroid_x < line_x <= cx:
                        count += 1
                        print(f"[COUNT {count}] Object crossed line at frame {frame_n}")

                prev_centroid_x = cx
            else:
                prev_centroid_x = None
        else:
            prev_centroid_x = None

        # Overlay count
        txt = f"COUNT: {count}"
        cv2.putText(display, txt, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)
        cv2.putText(display, txt, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 80), 3)

        # Overlay FPS
        fps_txt = f"FPS: {fps_display:.1f}"
        cv2.putText(display, fps_txt, (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(display, fps_txt, (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

        # Debug stats
        if debug_mode and frame_n % DEBUG_STATS_EVERY == 0:
            print(f"  [STATS] frame={frame_n}  sim  mean={sim_map.mean():.3f}  "
                  f"min={sim_map.min():.3f}  max={sim_map.max():.3f}  "
                  f"thresh={cur_thresh:.2f}")

        # Build display: detection | B&W threshold mask | heatmap overlay
        mask_bgr = cv2.cvtColor(mask_full, cv2.COLOR_GRAY2BGR)
        # Build full-frame diff for heatmap display
        diff_full = np.zeros((H, W), dtype=np.float32)
        diff_full[:, roi_x0:roi_x1] = diff_roi
        heatmap = make_heatmap(diff_full, H, W, cur_thresh)
        overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
        combined = np.hstack([display, mask_bgr, overlay])

        # Scale down if too wide
        max_w = 1400
        scale = min(1.0, max_w / combined.shape[1])
        if scale < 1.0:
            combined = cv2.resize(combined, (int(combined.shape[1] * scale),
                                             int(combined.shape[0] * scale)))

        # Rotate 90 degrees clockwise
        # combined = cv2.rotate(combined, cv2.ROTATE_90_CLOCKWISE)
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
                ref_rois.append(frame[:, roi_x0:roi_x1])
            if ref_rois:
                ref_bank = build_reference_bank(ref_rois)
                print(f"[REF] Bank updated: {ref_bank.shape}")
            count = 0
            prev_centroid_x = None
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
