#!/usr/bin/env python3
"""
patch_counter_v2.py — Patch-based DINOv2 conveyor belt counter.

Approach:
    1. Extract a narrow vertical strip at the counting line (right side of belt).
    2. Decompose into small patches stacked vertically.
    3. Dynamically calibrate DINOv2 patch embeddings from the first N empty-belt video frames.
    4. At runtime, compare each live patch embedding to its reference bank.
    5. If enough patches deviate from the empty belt → material detected → count.

No background subtraction, no contour detection, no centroid tracking.
Pure semantic patch comparison via DINOv2.

Usage:
    python patch_counter_v2.py --video path/to/input.mp4
    python patch_counter_v2.py --video path/to/input.mp4 --calibration-frames 30
    python patch_counter_v2.py --live
    python patch_counter_v2.py --live --gst-pipeline "v4l2src device=/dev/video0 ! videoconvert ! appsink"

Controls:  q=quit  r=reset  space=pause/resume  s=step  d=debug-sims
"""

import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
DEFAULT_CAMERA_DEVICE = "/dev/video0"
DEFAULT_CAMERA_WIDTH = 1280
DEFAULT_CAMERA_HEIGHT = 720
DEFAULT_CAMERA_FPS = 30

# ── DINOv2 ────────────────────────────────────────────────────────────────────
DINO_MODEL = "dinov2_vits14"

# ── Patch grid ────────────────────────────────────────────────────────────────
STRIP_X_FRACTION = 0.85       # counting line at 85% of frame width
STRIP_WIDTH      = 25         # pixel width of vertical column
PATCH_HEIGHT     = 25         # height of each patch
STRIP_SPACING    = 40         # px between strip A and strip B (B is STRIP_SPACING left of A)
NUM_STRIPS       = 2          # number of vertical strips used for consensus
CONSENSUS_MIN_STRIPS = 2      # patch is material only if at least this many strips trigger

# ── Detection thresholds ──────────────────────────────────────────────────────
SIM_THRESHOLD_MARGIN = 0.05   # margin below per-patch min self-similarity
MIN_CLUSTER_SIZE     = 2      # min contiguous triggered patches to form a cluster
MIN_TEMPORAL_HITS    = 2      # cluster must appear in N processed frames to count
SPLIT_ASSOC_DIST     = 5      # max center distance (patch units) to treat an extra cluster as a split of an existing object
SPLIT_CONFIRM_FRAMES = 4      # split must persist this many processed frames before counting as a new object

# ── Timing ────────────────────────────────────────────────────────────────────
PROCESS_EVERY_N  = 3          # process every Nth frame
DEBOUNCE_FRAMES  = 10         # cooldown per cluster slot after count
CALIBRATION_FRAMES = 20       # use first N frames to build empty-belt reference
CLUSTER_MAX_GAP  = 6          # max gap (in patch indices) to bridge within a cluster (handles hollow objects like wire coils)

# ── Display ───────────────────────────────────────────────────────────────────
DISPLAY_MAX_W = 1400


# ═══════════════════════════════════════════════════════════════════════════════
#  DINOv2 helpers (reused from dino_counter.py)
# ═══════════════════════════════════════════════════════════════════════════════

_DINO_INPUT_SIZE = (224, 224)
_DINO_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_DINO_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def _dino_tfm(rgb_patch):
    """Resize and normalize an RGB patch without depending on torchvision."""
    pil_img = Image.fromarray(rgb_patch)
    if hasattr(Image, "Resampling"):
        pil_img = pil_img.resize(_DINO_INPUT_SIZE, Image.Resampling.BILINEAR)
    else:
        pil_img = pil_img.resize(_DINO_INPUT_SIZE, Image.BILINEAR)

    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)
    return (tensor - _DINO_MEAN) / _DINO_STD


def load_dino(device):
    print("[DINO] Loading DINOv2...")
    m = torch.hub.load("facebookresearch/dinov2", DINO_MODEL, verbose=False)
    m.eval().to(device)
    print(f"[DINO] Ready on {device}")
    return m


@torch.no_grad()
def get_batch_embeddings(model, patches_bgr, device):
    """Batch-embed a list of BGR patches. Returns Tensor(N, embed_dim) L2-normalized."""
    tensors = [_dino_tfm(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)) for p in patches_bgr]
    batch = torch.stack(tensors).to(device)
    feats = model(batch)
    return F.normalize(feats, dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
#  PatchExtractor
# ═══════════════════════════════════════════════════════════════════════════════

class PatchExtractor:
    def __init__(self, strip_positions, strip_width, patch_height, frame_height):
        self.strip_positions = strip_positions
        self.strip_width = strip_width
        self.patch_height = patch_height
        self.frame_height = frame_height
        self._num_patches = frame_height // patch_height

    @property
    def num_patches(self):
        return self._num_patches

    def extract_strip(self, frame, strip_idx=0):
        """Crop a vertical strip by index."""
        x = self.strip_positions[strip_idx]
        return frame[:, x : x + self.strip_width].copy()

    def extract_strips(self, frame):
        """Return list of strips, one for each x position."""
        return [self.extract_strip(frame, i) for i in range(len(self.strip_positions))]

    def extract_patches(self, strip):
        """Split a strip into patches stacked vertically."""
        patches = []
        for i in range(self._num_patches):
            y0 = i * self.patch_height
            y1 = y0 + self.patch_height
            patches.append(strip[y0:y1, :])
        return patches


# ═══════════════════════════════════════════════════════════════════════════════
#  ReferenceBank
# ═══════════════════════════════════════════════════════════════════════════════

class ReferenceBank:
    def __init__(self):
        # Per-strip banks: each entry has shape (num_patches, num_images, embed_dim)
        self.embeddings = []
        self.thresholds = []  # each Tensor(num_patches,)
        self.num_patches = 0
        self.num_images = 0
        self.num_strips = 0

    def build(self, model, belt_images, extractor, device):
        """Build reference embeddings for all strips from calibration frames."""
        self.num_patches = extractor.num_patches
        self.num_images = len(belt_images)
        self.num_strips = len(extractor.strip_positions)
        all_embs = [[] for _ in range(self.num_strips)]
        for idx, img in enumerate(belt_images):
            strips = extractor.extract_strips(img)
            for si, strip in enumerate(strips):
                embs = get_batch_embeddings(model, extractor.extract_patches(strip), device)
                all_embs[si].append(embs)
            print(f"  [TRAIN] Image {idx+1}/{self.num_images} embedded ({self.num_strips} strips)")

        self.embeddings = [
            torch.stack(strip_embs, dim=0).permute(1, 0, 2) for strip_embs in all_embs
        ]
        self._compute_thresholds()
        shapes = ", ".join(f"S{si}={tuple(e.shape)}" for si, e in enumerate(self.embeddings))
        print(f"  [TRAIN] Bank shape: {shapes}")

    @torch.no_grad()
    def _compute_thresholds(self):
        """Compute per-patch adaptive thresholds for all strips."""
        self.thresholds = [self._thresholds_for(e) for e in self.embeddings]

    def _thresholds_for(self, embeddings):
        thresholds = []
        for i in range(self.num_patches):
            refs = embeddings[i]  # (I, D)
            sim_matrix = torch.mm(refs, refs.T)
            mask = ~torch.eye(sim_matrix.shape[0], dtype=torch.bool,
                              device=sim_matrix.device)
            off_diag = sim_matrix[mask]
            if off_diag.numel() > 0:
                thresholds.append(float(off_diag.min()) - SIM_THRESHOLD_MARGIN)
            else:
                thresholds.append(0.8)
        return torch.tensor(thresholds, device=embeddings.device)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "embeddings": [e.cpu() for e in self.embeddings],
            "thresholds": [t.cpu() for t in self.thresholds],
            "num_patches": self.num_patches,
            "num_images": self.num_images,
            "num_strips": self.num_strips,
            "strip_width": STRIP_WIDTH,
            "strip_spacing": STRIP_SPACING,
            "patch_height": PATCH_HEIGHT,
            "model": DINO_MODEL,
        }
        torch.save(data, path)
        print(f"  [TRAIN] Saved to {path}")

    def load(self, path, device):
        data = torch.load(path, map_location=device, weights_only=True)
        if (data["strip_width"] != STRIP_WIDTH or
            data["patch_height"] != PATCH_HEIGHT or
            data["model"] != DINO_MODEL or
            data.get("strip_spacing") != STRIP_SPACING):
            return False
        self.embeddings = [e.to(device) for e in data["embeddings"]]
        self.thresholds = [t.to(device) for t in data["thresholds"]]
        self.num_patches = data["num_patches"]
        self.num_images = data["num_images"]
        self.num_strips = data.get("num_strips", len(self.embeddings))
        print(f"  [TRAIN] Loaded cached strip bank with {self.num_strips} strips")
        return True

    @staticmethod
    def exists(path):
        return os.path.isfile(path)

    @torch.no_grad()
    def compare(self, embs_per_strip, consensus_min_strips):
        """
        Compare live patch embeddings from all strips against their banks.
        A patch is triggered when at least consensus_min_strips strips trigger.
        Returns:
            display_sims: Tensor(P,) min similarity across strips
            triggered_any: Tensor(P,) bool consensus trigger
            trig_list: list[Tensor(P,) bool], one per strip
            display_thresholds: Tensor(P,) min threshold across strips
        """
        sims_list = []
        trig_list = []
        for embs, refs, thr in zip(embs_per_strip, self.embeddings, self.thresholds):
            sims = torch.sum(embs.unsqueeze(1) * refs, dim=2).max(dim=1).values
            trig = sims < thr
            sims_list.append(sims)
            trig_list.append(trig)

        sims_stack = torch.stack(sims_list, dim=0)          # (S, P)
        trig_stack = torch.stack(trig_list, dim=0)          # (S, P)
        thr_stack = torch.stack(self.thresholds, dim=0)     # (S, P)

        trig_count = trig_stack.sum(dim=0)
        triggered = trig_count >= consensus_min_strips
        display_sims = sims_stack.min(dim=0).values
        display_thresholds = thr_stack.min(dim=0).values
        return display_sims, triggered, trig_list, display_thresholds

    def print_stats(self):
        """Print per-patch adaptive thresholds summary across strips."""
        print("\n[STATS] Per-patch adaptive thresholds (min/max over strips):")
        print(f"{'Patch':>6} {'ThrMin':>8} {'ThrMax':>8}")
        print("-" * 26)
        for i in range(self.num_patches):
            vals = [float(t[i]) for t in self.thresholds]
            print(f"{i:>6} {min(vals):>8.3f} {max(vals):>8.3f}")
        print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Cluster detection + multi-object counting
# ═══════════════════════════════════════════════════════════════════════════════

def find_clusters(triggered_list, min_size=MIN_CLUSTER_SIZE, max_gap=CLUSTER_MAX_GAP):
    """
    Find contiguous groups of triggered patches (with gap bridging).
    Returns list of (start_idx, end_idx) for each cluster.
    Filters out clusters smaller than min_size.
    """
    clusters = []
    n = len(triggered_list)
    i = 0
    while i < n:
        if triggered_list[i]:
            start = i
            end = i
            gap = 0
            j = i + 1
            while j < n:
                if triggered_list[j]:
                    end = j
                    gap = 0
                else:
                    gap += 1
                    if gap > max_gap:
                        break
                j += 1
            size = end - start + 1
            if size >= min_size:
                mid = (start + end) / 2.0
                clusters.append((start, end, mid))
            i = end + 1
        else:
            i += 1
    return clusters


class ClusterTracker:
    """
    Tracks clusters of triggered patches across frames.
    Each cluster slot has: center position, hit count, debounce timer, counted flag.
    Handles multiple objects crossing simultaneously.
    """
    def __init__(self, debounce_frames, min_temporal_hits, num_patches,
                 split_assoc_dist=SPLIT_ASSOC_DIST,
                 split_confirm_frames=SPLIT_CONFIRM_FRAMES):
        self.debounce_frames = debounce_frames
        self.min_temporal_hits = min_temporal_hits
        self.num_patches = num_patches
        self.split_assoc_dist = split_assoc_dist
        self.split_confirm_frames = split_confirm_frames
        self.count = 0
        self.slots = []  # list of dicts: {center, hits, debounce, counted, split_streak}
        self.state = "IDLE"

    def update(self, clusters):
        """
        clusters: list of (start, end, center) from find_clusters.
        Returns: (new_counts: int, active_clusters: list)
        """
        new_counts = 0
        matched_slots = set()
        matched_clusters = set()

        # Match clusters to existing slots by nearest center (one-to-one)
        for ci, (_, __, ccenter) in enumerate(clusters):
            best_slot = None
            best_dist = float('inf')
            for si, slot in enumerate(self.slots):
                if si in matched_slots:
                    continue
                dist = abs(slot["center"] - ccenter)
                if dist < best_dist and dist < self.num_patches * 0.3:
                    best_dist = dist
                    best_slot = si
            if best_slot is not None:
                matched_slots.add(best_slot)
                matched_clusters.add(ci)
                slot = self.slots[best_slot]
                slot["center"] = ccenter
                slot["hits"] += 1
                slot["debounce"] = 0
                # Count when hits reach threshold and not already counted
                if slot["hits"] >= self.min_temporal_hits and not slot["counted"]:
                    slot["counted"] = True
                    self.count += 1
                    new_counts += 1

        # For unmatched clusters near an already counted slot, apply split hysteresis.
        # This suppresses temporary hollow-object split artifacts (one object appearing as two).
        extras_by_slot = {}
        parent_slot_for_cluster = {}
        for ci, (_, __, ccenter) in enumerate(clusters):
            if ci in matched_clusters:
                continue
            best_slot = None
            best_dist = float("inf")
            for si, slot in enumerate(self.slots):
                if not slot["counted"] or slot["debounce"] > 0:
                    continue
                dist = abs(slot["center"] - ccenter)
                if dist < best_dist and dist <= self.split_assoc_dist:
                    best_dist = dist
                    best_slot = si
            if best_slot is not None:
                parent_slot_for_cluster[ci] = best_slot
                extras_by_slot[best_slot] = extras_by_slot.get(best_slot, 0) + 1

        for si, slot in enumerate(self.slots):
            if extras_by_slot.get(si, 0) > 0:
                slot["split_streak"] = slot.get("split_streak", 0) + 1
            else:
                slot["split_streak"] = max(0, slot.get("split_streak", 0) - 1)

        # Unmatched clusters -> new slots (or delayed if likely a temporary split)
        for ci, (_, __, ccenter) in enumerate(clusters):
            if ci not in matched_clusters:
                parent = parent_slot_for_cluster.get(ci)
                if parent is not None:
                    pslot = self.slots[parent]
                    if pslot.get("split_streak", 0) <= self.split_confirm_frames:
                        # Treat as temporary split of the same object; don't spawn a new count slot yet.
                        continue
                self.slots.append({
                    "center": ccenter,
                    "hits": 1,
                    "debounce": 0,
                    "counted": 1 >= self.min_temporal_hits,  # count immediately if min_hits=1
                    "split_streak": 0,
                })
                if 1 >= self.min_temporal_hits:
                    self.count += 1
                    new_counts += 1

        # Unmatched slots -> increment debounce, remove expired
        expired = []
        for si, slot in enumerate(self.slots):
            if si not in matched_slots:
                slot["debounce"] += 1
                if slot["debounce"] > self.debounce_frames:
                    expired.append(si)
        for si in reversed(expired):
            self.slots.pop(si)

        # Update state string
        if self.slots:
            active = sum(1 for s in self.slots if s["debounce"] == 0)
            if active > 0:
                self.state = f"DETECT({active})"
            else:
                self.state = "DEBOUNCE"
        else:
            self.state = "IDLE"

        return new_counts, [(s["center"], s["counted"]) for s in self.slots]

    def reset(self):
        self.count = 0
        self.slots.clear()
        self.state = "IDLE"


# ═══════════════════════════════════════════════════════════════════════════════
#  Display
# ═══════════════════════════════════════════════════════════════════════════════

def build_display(frame, strips_raw, patch_sims, triggered, thresholds,
                  triggered_per_strip,
                  count, state, paused, strip_positions, strip_width,
                  num_patches, patch_height, debug_sims):
    """
    Build a 4-panel display:
      Panel 1: Main video with strip overlays
      Panel 2+: Scaled-up strip previews (first 2 strips)
      Panel 4: Similarity heatmap per patch
    """
    H, W = frame.shape[:2]
    vis = frame.copy()

    # Draw strip zones as translucent vertical bands + outlines + labels.
    overlay_strips = vis.copy()
    strip_colors = [(0, 220, 220), (220, 220, 0), (220, 120, 0), (180, 80, 220), (0, 200, 120)]
    for si, sx in enumerate(strip_positions):
        c = strip_colors[si % len(strip_colors)]
        cv2.rectangle(overlay_strips, (sx, 0), (sx + strip_width, H - 1), c, -1)
    cv2.addWeighted(overlay_strips, 0.22, vis, 0.78, 0, vis)
    for si, sx in enumerate(strip_positions):
        c = strip_colors[si % len(strip_colors)]
        cv2.rectangle(vis, (sx, 0), (sx + strip_width, H - 1), c, 2)
        cv2.putText(vis, f"S{si+1}", (sx + 2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 2)

    # Highlight triggered patches on each strip independently (no fill in gaps).
    if triggered is not None:
        overlay = vis.copy()
        for i in range(num_patches):
            y0 = i * patch_height
            y1 = y0 + patch_height
            if triggered_per_strip is not None:
                for si, sx in enumerate(strip_positions):
                    if bool(triggered_per_strip[si][i]):
                        cv2.rectangle(overlay, (sx, y0), (sx + strip_width, y1), (0, 0, 255), -1)
            elif bool(triggered[i]):
                for sx in strip_positions:
                    cv2.rectangle(overlay, (sx, y0), (sx + strip_width, y1), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)

    # Count text (shadow + green)
    txt = f"COUNT: {count}"
    cv2.putText(vis, txt, (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)
    cv2.putText(vis, txt, (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 80), 3)

    # State
    if "DETECT" in state:
        state_color = (0, 0, 255)
    elif "DEBOUNCE" in state:
        state_color = (0, 180, 255)
    else:
        state_color = (200, 200, 200)
    cv2.putText(vis, state, (18, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv2.putText(vis, state, (18, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)

    if paused:
        cv2.putText(vis, "PAUSED", (W - 160, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 200), 3)

    # Strip preview panels (first 2 strips to keep UI compact).
    col_w = 60

    def _make_strip_panel(strip_raw, label):
        if strip_raw is not None:
            panel = cv2.resize(strip_raw, (col_w, H), interpolation=cv2.INTER_NEAREST)
            for i in range(1, num_patches):
                y = int(i * patch_height * (H / (num_patches * patch_height)))
                cv2.line(panel, (0, y), (col_w, y), (100, 100, 100), 1)
        else:
            panel = np.zeros((H, col_w, 3), dtype=np.uint8)
        cv2.putText(panel, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
        return panel

    strips_list = strips_raw if strips_raw is not None else []
    strip_up_panels = []
    for si in range(min(2, len(strips_list))):
        strip_up_panels.append(_make_strip_panel(strips_list[si], f"S{si+1}"))
    while len(strip_up_panels) < 2:
        strip_up_panels.append(_make_strip_panel(None, "N/A"))

    # Panel 4: Similarity heatmap
    heatmap_w = 120
    heatmap = np.zeros((H, heatmap_w, 3), dtype=np.uint8)
    if patch_sims is not None:
        bar_h = H / num_patches
        sims_np = patch_sims.cpu().numpy() if torch.is_tensor(patch_sims) else patch_sims
        trig_np = triggered if isinstance(triggered, list) else (
            triggered.cpu().numpy() if torch.is_tensor(triggered) else triggered)
        thr_np = thresholds.cpu().numpy() if torch.is_tensor(thresholds) else thresholds

        for i in range(num_patches):
            y0 = int(i * bar_h)
            y1 = int((i + 1) * bar_h)
            sim = float(sims_np[i])
            thr = float(thr_np[i]) if thr_np is not None else 0.8

            # Color relative to per-patch threshold
            if sim >= thr:
                # Above threshold = green (belt)
                margin = min(1.0, (sim - thr) / 0.1)  # how far above threshold
                g = int(100 + 155 * margin)
                color = (0, g, 0)
            else:
                # Below threshold = red (material)
                margin = min(1.0, (thr - sim) / 0.1)  # how far below threshold
                r = int(100 + 155 * margin)
                color = (0, 0, r)

            cv2.rectangle(heatmap, (0, y0), (heatmap_w - 1, y1), color, -1)

            # Border for triggered patches
            if trig_np is not None and trig_np[i]:
                cv2.rectangle(heatmap, (0, y0), (heatmap_w - 1, y1), (0, 0, 255), 2)

            # Similarity value text
            if debug_sims:
                cv2.putText(heatmap, f"{sim:.2f}/{thr:.2f}",
                            (2, y0 + int(bar_h / 2) + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)

    # Combine panels
    combined = np.hstack([vis, strip_up_panels[0], strip_up_panels[1], heatmap])
    scale = min(1.0, DISPLAY_MAX_W / combined.shape[1])
    if scale < 1.0:
        combined = cv2.resize(combined, (int(combined.shape[1] * scale),
                                         int(combined.shape[0] * scale)))
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Patch-based conveyor counter with dynamic empty-belt calibration."
    )
    parser.add_argument(
        "--video",
        default="",
        help="Path to input video file (required unless --live is used).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live webcam input via GStreamer instead of --video file input.",
    )
    parser.add_argument(
        "--gst-pipeline",
        default="",
        help="Custom GStreamer pipeline string for live mode.",
    )
    parser.add_argument(
        "--camera-device",
        default=DEFAULT_CAMERA_DEVICE,
        help="Camera device path used to build default GStreamer pipeline in live mode.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=DEFAULT_CAMERA_WIDTH,
        help="Camera width for default live GStreamer pipeline.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=DEFAULT_CAMERA_HEIGHT,
        help="Camera height for default live GStreamer pipeline.",
    )
    parser.add_argument(
        "--camera-fps",
        type=int,
        default=DEFAULT_CAMERA_FPS,
        help="Camera FPS for default live GStreamer pipeline.",
    )
    parser.add_argument(
        "--calibration-frames",
        type=int,
        default=CALIBRATION_FRAMES,
        help="Number of initial frames used for empty-belt calibration.",
    )
    parser.add_argument(
        "--strip-spacing",
        type=int,
        default=STRIP_SPACING,
        help="Horizontal distance in pixels between adjacent strips.",
    )
    parser.add_argument(
        "--strip-x-fraction",
        type=float,
        default=STRIP_X_FRACTION,
        help="Horizontal location of strip A as fraction of frame width (0.0 to 1.0).",
    )
    parser.add_argument(
        "--split-assoc-dist",
        type=int,
        default=SPLIT_ASSOC_DIST,
        help="Patch-center distance to treat extra cluster as split of an existing object.",
    )
    parser.add_argument(
        "--split-confirm-frames",
        type=int,
        default=SPLIT_CONFIRM_FRAMES,
        help="Frames a split must persist before it is treated as a new object.",
    )
    parser.add_argument(
        "--num-strips",
        type=int,
        default=NUM_STRIPS,
        help="Number of vertical strips to use (>=2 recommended for robustness).",
    )
    parser.add_argument(
        "--consensus-min-strips",
        type=int,
        default=CONSENSUS_MIN_STRIPS,
        help="Patch is triggered only when at least this many strips agree.",
    )
    return parser.parse_args()


def compute_strip_positions(frame_width, strip_width, strip_x_fraction, strip_spacing, num_strips):
    """
    Compute safe in-frame x positions for N strips.
    Strip 0 is anchor (right-most), subsequent strips are placed to the left.
    Returns: (positions, effective_spacing, requested_count)
    """
    max_x = max(0, frame_width - strip_width)
    anchor_x = int(frame_width * strip_x_fraction)
    anchor_x = min(max(anchor_x, 0), max_x)

    positions = []
    for i in range(num_strips):
        sx = anchor_x - i * strip_spacing
        sx = min(max(sx, 0), max_x)
        positions.append(sx)
    # Remove duplicates caused by clamping
    positions = sorted(set(positions), reverse=True)
    effective_spacing = positions[0] - positions[1] if len(positions) > 1 else 0
    return positions, effective_spacing, num_strips


def build_default_gst_pipeline(device, width, height, fps):
    """Build a default Linux webcam pipeline that outputs BGR frames for OpenCV."""
    return (
        f"v4l2src device={device} ! "
        "videoconvert ! "
        f"video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def open_input_capture(args):
    """
    Open input source based on args.
    Returns: (cap, source_label, source_detail)
    """
    if args.live:
        pipeline = args.gst_pipeline.strip() or build_default_gst_pipeline(
            args.camera_device, args.camera_width, args.camera_height, args.camera_fps
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        return cap, "live-gstreamer", pipeline

    cap = cv2.VideoCapture(args.video)
    return cap, "video-file", args.video


def main():
    args = parse_args()
    if args.calibration_frames < 1:
        print("ERROR: --calibration-frames must be >= 1")
        return
    if args.strip_spacing < 1:
        print("ERROR: --strip-spacing must be >= 1")
        return
    if not (0.0 <= args.strip_x_fraction <= 1.0):
        print("ERROR: --strip-x-fraction must be between 0.0 and 1.0")
        return
    if args.split_assoc_dist < 1:
        print("ERROR: --split-assoc-dist must be >= 1")
        return
    if args.split_confirm_frames < 1:
        print("ERROR: --split-confirm-frames must be >= 1")
        return
    if args.num_strips < 1:
        print("ERROR: --num-strips must be >= 1")
        return
    if args.consensus_min_strips < 1:
        print("ERROR: --consensus-min-strips must be >= 1")
        return
    if not args.live and not args.video.strip():
        print("ERROR: Provide --video <path> or use --live")
        return
    if args.live:
        if args.camera_width < 1 or args.camera_height < 1 or args.camera_fps < 1:
            print("ERROR: --camera-width, --camera-height and --camera-fps must be >= 1")
            return

    device = ("mps"  if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available()       else "cpu")
    print(f"[INFO] Device: {device}")

    model = load_dino(device)

    # Open source for dimensions
    cap, source_kind, source_detail = open_input_capture(args)
    if not cap.isOpened():
        if args.live:
            print("ERROR: Cannot open live GStreamer source.")
            print("Check OpenCV GStreamer support and pipeline validity:")
            print(source_detail)
        else:
            print(f"ERROR: Cannot open {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    strip_positions, eff_spacing, requested_strips = compute_strip_positions(
        W, STRIP_WIDTH, args.strip_x_fraction, args.strip_spacing, args.num_strips
    )
    if len(strip_positions) < 1:
        print("ERROR: Could not place any strip in frame.")
        return
    consensus_min = min(args.consensus_min_strips, len(strip_positions))
    if consensus_min != args.consensus_min_strips:
        print(f"[WARN] consensus-min-strips reduced to {consensus_min} (active strips={len(strip_positions)}).")
    extractor = PatchExtractor(strip_positions, STRIP_WIDTH, PATCH_HEIGHT, H)
    num_patches = extractor.num_patches

    print(f"\nInput kind  : {source_kind}")
    print(f"Input       : {source_detail}")
    print(f"Resolution  : {W}x{H} @ {fps:.1f} fps")
    pos_text = ", ".join(f"S{i+1}@x={x}" for i, x in enumerate(strip_positions))
    print(f"Strips      : {pos_text}")
    print(f"Spacing     : requested={args.strip_spacing}px, effective~={eff_spacing}px")
    print(f"Consensus   : {consensus_min} of {len(strip_positions)} strips")
    print(f"Patch grid  : {num_patches} patches of {PATCH_HEIGHT}x{STRIP_WIDTH}")
    print(f"Thresh margin: {SIM_THRESHOLD_MARGIN}  (adaptive per-patch)\n")
    print(f"Split guard : assoc_dist={args.split_assoc_dist}  confirm={args.split_confirm_frames}")
    if len(strip_positions) != requested_strips:
        print("[WARN] Some strips were clamped/merged at frame boundary. "
              "Use smaller --num-strips / --strip-spacing or larger --strip-x-fraction.")

    # ── Dynamic calibration phase ─────────────────────────────────────────────
    bank = ReferenceBank()
    print(f"[TRAIN] Calibrating from first {args.calibration_frames} frame(s)...")
    calibration_frames = []
    for i in range(args.calibration_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"ERROR: Video ended during calibration at frame {i}/{args.calibration_frames}")
            cap.release()
            return
        calibration_frames.append(frame)
    bank.build(model, calibration_frames, extractor, device)
    print("[TRAIN] Dynamic calibration complete.")

    bank.print_stats()

    # ── Cluster tracker ──────────────────────────────────────────────────────
    counter = ClusterTracker(
        DEBOUNCE_FRAMES,
        MIN_TEMPORAL_HITS,
        num_patches,
        split_assoc_dist=args.split_assoc_dist,
        split_confirm_frames=args.split_confirm_frames,
    )

    frame_n = len(calibration_frames)
    paused = False
    debug_sims = False
    last_sims = None
    last_triggered = None
    last_triggered_per_strip = None
    last_strips = None
    display_thresholds = None

    cv2.namedWindow("PatchCount", cv2.WINDOW_NORMAL)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print(f"\nVideo ended.  Final count: {counter.count}")
                break
            frame_n += 1
        else:
            # When paused, just handle keys
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                counter.reset()
                last_sims = None
                last_triggered = None
                last_triggered_per_strip = None
                last_strips = None
                print("[RESET]")
            elif key == ord(' '):
                paused = False
                print("[RESUMED]")
            elif key == ord('d'):
                debug_sims = not debug_sims
                print(f"[DEBUG SIMS {'ON' if debug_sims else 'OFF'}]")
            elif key == ord('s'):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_n += 1
                # Fall through to process this frame
            else:
                # Redisplay current state
                disp = build_display(frame, last_strips,
                                     last_sims, last_triggered, display_thresholds,
                                     last_triggered_per_strip,
                                     counter.count, counter.state, paused,
                                     strip_positions, STRIP_WIDTH, num_patches,
                                     PATCH_HEIGHT, debug_sims)
                cv2.imshow("PatchCount", disp)
                continue

        # ── Process every Nth frame ───────────────────────────────────────────
        if frame_n % PROCESS_EVERY_N == 0 or last_sims is None:
            strips = extractor.extract_strips(frame)
            embs_per_strip = [
                get_batch_embeddings(model, extractor.extract_patches(s), device)
                for s in strips
            ]
            max_sims, triggered_tensor, trig_tensors, display_thresholds = bank.compare(
                embs_per_strip, consensus_min
            )

            triggered = triggered_tensor.cpu().tolist()
            triggered_per_strip = [t.cpu().tolist() for t in trig_tensors]
            clusters = find_clusters(triggered)

            last_sims = max_sims
            last_triggered = triggered
            last_triggered_per_strip = triggered_per_strip
            last_strips = strips

            new_counts, active = counter.update(clusters)
            if new_counts > 0:
                n_triggered = sum(triggered)
                cluster_info = ", ".join(
                    f"center={c:.0f}" for c, _ in active if _)
                print(f"[COUNT {counter.count:>4}]  +{new_counts}  "
                      f"{n_triggered}/{num_patches} patches  "
                      f"clusters=[{cluster_info}]  frame={frame_n}")

            if debug_sims:
                sims_np = max_sims.cpu().numpy()
                trig_idx = [i for i, t in enumerate(triggered) if t]
                by_strip = []
                for si, ts in enumerate(triggered_per_strip):
                    idxs = [i for i, t in enumerate(ts) if t]
                    by_strip.append(f"S{si+1}={idxs}")
                print(f"  frame={frame_n}  triggered={trig_idx}  "
                      f"{' '.join(by_strip)}  "
                      f"clusters={[(s,e) for s,e,_ in clusters]}  "
                      f"min_sim={sims_np.min():.3f}  "
                      f"mean_sim={sims_np.mean():.3f}")

        # ── Display ───────────────────────────────────────────────────────────
        disp = build_display(frame, last_strips,
                             last_sims, last_triggered, display_thresholds,
                             last_triggered_per_strip,
                             counter.count, counter.state, paused,
                             strip_positions, STRIP_WIDTH, num_patches,
                             PATCH_HEIGHT, debug_sims)
        cv2.imshow("PatchCount", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            counter.reset()
            last_sims = None
            last_triggered = None
            last_triggered_per_strip = None
            last_strips = None
            print("[RESET]")
        elif key == ord(' '):
            paused = not paused
            print("[PAUSED]" if paused else "[RESUMED]")
        elif key == ord('d'):
            debug_sims = not debug_sims
            print(f"[DEBUG SIMS {'ON' if debug_sims else 'OFF'}]")
        elif key == ord('s') and paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_n += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n{'='*40}")
    print(f"Final count: {counter.count}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
