"""
SegmentConfirmer — Multi-technique segmentation confirmation for conveyor counting.
Clofus Innovations / ConveyorBot / CountBot pipeline.

Architecture:
    MOG2 blob → Watershed split → Contour geometry gate →
    GrabCut boundary refine → LBP texture gate → DINOv2 semantic confirm

Usage:
    confirmer = SegmentConfirmer(sku_refs, belt_sample, config)
    for detection in bytetrack_detections:
        result = confirmer.confirm(frame, detection.bbox, detection.track_id)
        if result.accepted:
            count += 1
        elif result.flagged:
            review_queue.append(result)


Simple flow:

Get a bounding box on the frame.
Check if there is motion inside that box.
Check the shape of the moving region.
Refine the object boundary with GrabCut.
Compare texture with the conveyor belt.
Optionally compare semantic features with DINOv2 references.
Combine all scores and mark it OK, REVIEW, or REJECT.
What each method does:

Motion in segment_confirmer.py (line 238)
Checks how much of the box is real foreground movement. If nothing is moving, it is probably not a product.
Geometry in segment_confirmer.py (line 251)
Looks at contour size, aspect ratio, and solidity. This helps reject strange shapes or tiny blobs.
GrabCut boundary refine in segment_confirmer.py (line 300)
Cleans the object mask so the boundary fits the object better, not just rough motion pixels.
LBP texture in segment_confirmer.py (line 346)
Compares the texture inside the box to the conveyor belt texture. If it looks too much like the belt, it is likely background.
DINOv2 semantic in segment_confirmer.py (line 364)
Compares the crop to reference product embeddings. This is the “does it look like the expected object class?” step.
Weighted decision in segment_confirmer.py (line 199)
Combines all scores into one final confidence.
How final labels work:

OK: score is high enough, accept as product
REVIEW: uncertain, maybe product
REJECT: low confidence, likely not product
In your current demo:

video proposals come from motion blobs in segment_confirmer.py (line 599)
boxes are drawn in segment_confirmer.py (line 617)
DINOv2 is not really active yet because demo uses empty sku_refs={} in segment_confirmer.py (line 655)
So right now the demo mainly uses:

motion
shape
boundary
texture
"""

import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional
from skimage.feature import local_binary_pattern


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ConfirmerConfig:
    # Watershed
    dist_threshold: float = 0.5        # fraction of max distance for sure-fg
    min_blob_area: int = 800           # px² — discard tiny noise blobs
    max_blob_area: int = 80000         # px² — discard oversized merges

    # Contour geometry gate (per-SKU overrides loaded separately)
    min_aspect_ratio: float = 0.2
    max_aspect_ratio: float = 5.0
    min_solidity: float = 0.4          # convex hull fill ratio

    # GrabCut
    enable_grabcut: bool = True
    grabcut_iters: int = 5
    grabcut_area_ratio_min: float = 0.3  # refined mask / bbox area
    grabcut_area_ratio_max: float = 0.95

    # LBP texture
    enable_texture: bool = True
    lbp_radius: int = 1
    lbp_points: int = 8
    lbp_chi_threshold: float = 0.3    # higher = more different from belt

    # DINOv2
    enable_semantic: bool = True
    dino_accept_threshold: float = 0.72
    dino_review_threshold: float = 0.55

    # Weighted voting
    weights: dict = field(default_factory=lambda: {
        "motion":   0.10,
        "geometry": 0.20,
        "boundary": 0.15,
        "texture":  0.15,
        "semantic": 0.40,
    })
    accept_threshold: float = 0.68
    review_threshold: float = 0.45


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ConfirmResult:
    track_id: int
    bbox: tuple                        # (x, y, w, h)
    accepted: bool
    flagged: bool                      # needs human review
    rejected: bool
    confidence: float
    scores: dict
    matched_sku: Optional[str] = None
    matched_sku_sim: float = 0.0
    refined_mask: Optional[np.ndarray] = None
    debug_info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SegmentConfirmer:
    """
    Confirms each ByteTrack detection through 5 layered techniques before
    accepting it as a valid count event.

    Args:
        sku_refs: dict[sku_name -> torch.Tensor] — DINOv2 embeddings per SKU.
                  Shape: (N_refs, embed_dim). Multiple refs per SKU supported.
        belt_sample: np.ndarray — a clean belt frame (no parts) for LBP baseline.
        dino_model: callable — takes (np.ndarray crop) → torch.Tensor embedding.
                    Should return normalized L2 embedding.
        config: ConfirmerConfig
    """

    def __init__(
        self,
        sku_refs: dict,
        belt_sample: np.ndarray,
        dino_model,
        config: Optional[ConfirmerConfig] = None,
    ):
        self.sku_refs = sku_refs          # {sku_name: Tensor(N, D)}
        self.dino_model = dino_model
        self.config = config or ConfirmerConfig()

        # Precompute belt LBP histogram for texture comparison
        self._belt_lbp_hist = self._compute_lbp_hist(belt_sample)

        # MOG2 background subtractor (shared with main pipeline ideally,
        # but can be standalone here)
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=40, detectShadows=True
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def confirm(
        self,
        frame: np.ndarray,
        bbox: tuple,
        track_id: int,
        fg_mask: Optional[np.ndarray] = None,
    ) -> ConfirmResult:
        """
        Run all 5 confirmation layers on a single detection.

        Args:
            frame:   Full BGR frame.
            bbox:    (x, y, w, h) from ByteTrack.
            track_id: ByteTrack ID.
            fg_mask: Optional pre-computed MOG2 mask (same size as frame).
                     If None, we run MOG2 internally (less accurate).
        """
        cfg = self.config
        scores = {}
        debug = {}
        x, y, w, h = [int(v) for v in bbox]

        # Clamp bbox to frame bounds
        H, W = frame.shape[:2]
        x, y = max(0, x), max(0, y)
        w = min(w, W - x)
        h = min(h, H - y)
        if w < 10 or h < 10:
            return self._reject(track_id, bbox, scores, "bbox_too_small")

        crop = frame[y:y+h, x:x+w]
        crop_mask = fg_mask[y:y+h, x:x+w] if fg_mask is not None else None

        # ── Layer 1: Motion strength ───────────────────────────────────
        scores["motion"], motion_mask = self._score_motion(
            crop, crop_mask, x, y, w, h
        )
        debug["motion_fill"] = scores["motion"]

        # ── Layer 2: Geometry gate ─────────────────────────────────────
        scores["geometry"], contours, debug_geo = self._score_geometry(
            motion_mask, w, h
        )
        debug.update(debug_geo)

        # Early reject: if geometry is zero, nothing to refine
        if scores["geometry"] == 0.0 and scores["motion"] < 0.3:
            return self._reject(track_id, bbox, scores, "geometry_fail")

        # ── Layer 3: GrabCut boundary refinement ──────────────────────
        scores["boundary"], refined_mask = self._score_grabcut(
            crop, motion_mask, w, h
        )
        debug["grabcut_area_ratio"] = scores["boundary"]

        # ── Layer 4: LBP texture ───────────────────────────────────────
        scores["texture"] = self._score_texture(crop, refined_mask)
        debug["lbp_chi"] = scores["texture"]

        # ── Layer 5: DINOv2 semantic ───────────────────────────────────
        best_sku, best_sim, semantic_score = self._score_semantic(
            crop, refined_mask
        )
        scores["semantic"] = semantic_score
        debug["dino_sim"] = best_sim
        debug["dino_sku"] = best_sku

        # ── Weighted vote ──────────────────────────────────────────────
        W_map = cfg.weights
        total = sum(scores[k] * W_map[k] for k in W_map)

        accepted = total >= cfg.accept_threshold
        flagged  = (not accepted) and (total >= cfg.review_threshold)
        rejected = not accepted and not flagged

        return ConfirmResult(
            track_id=track_id,
            bbox=(x, y, w, h),
            accepted=accepted,
            flagged=flagged,
            rejected=rejected,
            confidence=round(total, 4),
            scores={k: round(v, 4) for k, v in scores.items()},
            matched_sku=best_sku,
            matched_sku_sim=round(best_sim, 4),
            refined_mask=refined_mask,
            debug_info=debug,
        )

    def update_belt_reference(self, belt_frame: np.ndarray):
        """Call this when belt is empty to refresh LBP baseline."""
        self._belt_lbp_hist = self._compute_lbp_hist(belt_frame)

    def add_sku_ref(self, sku_name: str, embedding: torch.Tensor):
        """Add a new reference embedding for an SKU (online learning)."""
        if sku_name in self.sku_refs:
            self.sku_refs[sku_name] = torch.cat(
                [self.sku_refs[sku_name], embedding.unsqueeze(0)], dim=0
            )
        else:
            self.sku_refs[sku_name] = embedding.unsqueeze(0)

    # ------------------------------------------------------------------
    # Layer implementations
    # ------------------------------------------------------------------

    def _score_motion(self, crop, crop_mask, x, y, w, h):
        """Layer 1: Foreground fill ratio inside bbox."""
        if crop_mask is None:
            # Run MOG2 on crop — less accurate without full frame context
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop_mask = self.mog2.apply(crop)

        # MOG2 returns 255=foreground, 127=shadow, 0=background
        fg = (crop_mask == 255).astype(np.uint8) * 255
        fill_ratio = fg.sum() / 255 / (w * h + 1e-6)
        score = float(np.clip(fill_ratio / 0.4, 0.0, 1.0))
        return score, fg

    def _score_geometry(self, motion_mask, w, h):
        """Layer 2: Contour area, aspect ratio, solidity check."""
        cfg = self.config
        debug = {}

        if motion_mask is None or motion_mask.sum() == 0:
            return 0.0, [], debug

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0.0, [], debug

        # Score best contour (largest)
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        debug["contour_area"] = area

        if not (cfg.min_blob_area <= area <= cfg.max_blob_area):
            debug["geometry_fail"] = "area_out_of_range"
            return 0.0, contours, debug

        rx, ry, rw, rh = cv2.boundingRect(c)
        aspect = rw / (rh + 1e-6)
        debug["aspect_ratio"] = aspect

        if not (cfg.min_aspect_ratio <= aspect <= cfg.max_aspect_ratio):
            debug["geometry_fail"] = "aspect_out_of_range"
            return 0.3, contours, debug

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-6)
        debug["solidity"] = solidity

        if solidity < cfg.min_solidity:
            debug["geometry_fail"] = "low_solidity"
            return 0.4, contours, debug

        # All checks passed
        score = min(1.0, solidity * 0.5 + 0.5)
        return score, contours, debug

    def _score_grabcut(self, crop, motion_mask, w, h):
        """Layer 3: GrabCut refinement — area consistency check."""
        cfg = self.config

        if not cfg.enable_grabcut or cfg.grabcut_iters <= 0:
            return 0.5, motion_mask

        if crop.shape[0] < 10 or crop.shape[1] < 10:
            return 0.5, motion_mask  # pass-through, not enough pixels

        try:
            gc_mask = np.zeros(crop.shape[:2], np.uint8)
            bgd = np.zeros((1, 65), np.float64)
            fgd = np.zeros((1, 65), np.float64)

            # Use motion mask to seed GrabCut if available
            if motion_mask is not None and motion_mask.sum() > 0:
                gc_mask[motion_mask == 255] = cv2.GC_PR_FGD
                gc_mask[motion_mask == 0]   = cv2.GC_PR_BGD
                mode = cv2.GC_INIT_WITH_MASK
            else:
                rect = (2, 2, w - 4, h - 4)
                mode = cv2.GC_INIT_WITH_RECT

            if mode == cv2.GC_INIT_WITH_RECT:
                cv2.grabCut(crop, gc_mask, rect, bgd, fgd,
                            cfg.grabcut_iters, mode)
            else:
                cv2.grabCut(crop, gc_mask, None, bgd, fgd,
                            cfg.grabcut_iters, mode)

            refined = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                255, 0
            ).astype(np.uint8)

            area_ratio = refined.sum() / 255 / (w * h + 1e-6)
            if cfg.grabcut_area_ratio_min <= area_ratio <= cfg.grabcut_area_ratio_max:
                score = 1.0 - abs(area_ratio - 0.6) / 0.4  # peak at 60% fill
                score = float(np.clip(score, 0.0, 1.0))
            else:
                score = 0.3
                refined = motion_mask  # fallback to motion mask

        except cv2.error:
            return 0.5, motion_mask

        return score, refined

    def _score_texture(self, crop, mask):
        """Layer 4: LBP histogram distance from belt baseline."""
        cfg = self.config

        if not cfg.enable_texture:
            return 0.5

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        if mask is not None and mask.sum() > 100:
            # Compute LBP only inside segment mask
            seg_hist = self._compute_lbp_hist(gray, mask)
        else:
            seg_hist = self._compute_lbp_hist(gray)

        # Chi-squared distance from belt baseline
        chi = self._chi_squared(seg_hist, self._belt_lbp_hist)
        # Higher chi = more different from belt = more likely a real object
        score = float(np.clip(chi / cfg.lbp_chi_threshold, 0.0, 1.0))
        return score

    def _score_semantic(self, crop, mask):
        """Layer 5: DINOv2 cosine similarity against all SKU references."""
        cfg = self.config

        if not cfg.enable_semantic:
            return None, 0.0, 0.5

        if not self.sku_refs:
            return None, 0.0, 0.5  # neutral if no refs

        # Apply mask to crop before embedding
        if mask is not None and mask.sum() > 100:
            masked_crop = crop.copy()
            masked_crop[mask == 0] = 0  # zero out background
        else:
            masked_crop = crop

        try:
            embedding = self.dino_model(masked_crop)  # → Tensor(D,)
            embedding = F.normalize(embedding.unsqueeze(0), dim=1)  # (1, D)
        except Exception:
            return None, 0.0, 0.5

        best_sku = None
        best_sim = 0.0

        for sku_name, refs in self.sku_refs.items():
            # refs: (N, D) — take max similarity across references
            refs_norm = F.normalize(refs, dim=1)
            sims = F.cosine_similarity(embedding, refs_norm)
            sim = sims.max().item()
            if sim > best_sim:
                best_sim = sim
                best_sku = sku_name

        # Map similarity to score
        if best_sim >= cfg.dino_accept_threshold:
            score = 1.0
        elif best_sim >= cfg.dino_review_threshold:
            score = (best_sim - cfg.dino_review_threshold) / (
                cfg.dino_accept_threshold - cfg.dino_review_threshold
            )
        else:
            score = 0.0

        return best_sku, best_sim, float(score)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_lbp_hist(self, img, mask=None):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cfg = self.config
        lbp = local_binary_pattern(
            img, cfg.lbp_points, cfg.lbp_radius, method='uniform'
        )
        n_bins = cfg.lbp_points + 2
        if mask is not None:
            pixels = lbp[mask == 255]
        else:
            pixels = lbp.ravel()
        hist, _ = np.histogram(pixels, bins=n_bins, range=(0, n_bins), density=True)
        return hist.astype(np.float32)

    @staticmethod
    def _chi_squared(h1, h2):
        eps = 1e-10
        return float(np.sum((h1 - h2) ** 2 / (h1 + h2 + eps)) * 0.5)

    def _reject(self, track_id, bbox, scores, reason):
        return ConfirmResult(
            track_id=track_id,
            bbox=bbox,
            accepted=False,
            flagged=False,
            rejected=True,
            confidence=0.0,
            scores=scores,
            debug_info={"reject_reason": reason},
        )


# ---------------------------------------------------------------------------
# Integration shim — plug into your ByteTrack loop
# ---------------------------------------------------------------------------

class CountingPipeline:
    """
    Minimal example showing how SegmentConfirmer fits into ByteTrack loop.
    Replace with your actual ByteTrack + YOLOE calls.
    """

    def __init__(self, confirmer: SegmentConfirmer, count_line_y: int):
        self.confirmer = confirmer
        self.count_line_y = count_line_y
        self.count = 0
        self.reviewed = []
        self._crossed: set = set()       # track_ids that already crossed line
        self._last_bbox: dict = {}       # track_id -> last bbox centroid y

    def process_frame(self, frame: np.ndarray, tracks: list, fg_mask: np.ndarray):
        """
        Args:
            tracks: list of dicts with keys: track_id, bbox (x,y,w,h)
            fg_mask: MOG2 output mask, same HxW as frame
        """
        for track in tracks:
            tid = track["track_id"]
            bbox = track["bbox"]
            cx = bbox[0] + bbox[2] // 2
            cy = bbox[1] + bbox[3] // 2

            # Detect line crossing (centroid crosses count_line_y)
            prev_cy = self._last_bbox.get(tid, cy)
            crossing = (prev_cy < self.count_line_y <= cy) and tid not in self._crossed

            self._last_bbox[tid] = cy

            if not crossing:
                continue

            # Run confirmation only at crossing event
            result = self.confirmer.confirm(frame, bbox, tid, fg_mask)

            if result.accepted:
                self.count += 1
                self._crossed.add(tid)
                print(f"[COUNT] {self.count} | SKU={result.matched_sku} "
                      f"sim={result.matched_sku_sim:.2f} conf={result.confidence:.2f}")

            elif result.flagged:
                self.reviewed.append(result)
                print(f"[REVIEW] track={tid} conf={result.confidence:.2f} "
                      f"scores={result.scores}")

            else:
                print(f"[REJECT] track={tid} reason="
                      f"{result.debug_info.get('reject_reason','low_conf')} "
                      f"conf={result.confidence:.2f}")


# ---------------------------------------------------------------------------
# Simple demo runner
# ---------------------------------------------------------------------------

DEFAULT_VIDEO = "/home/vikbot/Documents/countbot/testing/video/bearing_wheel.mp4"
DEFAULT_OUTPUT = "/home/vikbot/Documents/countbot/testing/segment_confirmer_output.mp4"


def parse_args():
    parser = argparse.ArgumentParser(description="Run SegmentConfirmer on a video with live preview.")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Input video path.")
    parser.add_argument("--output", default="", help="Optional output video path.")
    parser.add_argument("--no-show", action="store_true", help="Disable live preview window.")
    parser.add_argument("--fast", action="store_true", help="Use a faster demo configuration with lighter confirmation.")
    parser.add_argument("--max-width", type=int, default=960, help="Resize preview width while keeping aspect ratio.")
    parser.add_argument("--min-area", type=int, default=1200, help="Minimum motion proposal area in pixels.")
    parser.add_argument("--box-pad", type=int, default=24, help="Extra padding added around each proposal box.")
    parser.add_argument("--merge-gap", type=int, default=28, help="Merge nearby proposal boxes within this many pixels.")
    parser.add_argument("--process-every", type=int, default=1, help="Run confirmation every Nth frame.")
    parser.add_argument("--grabcut-iters", type=int, default=5, help="GrabCut iterations per proposal. Use 0 to disable.")
    parser.add_argument("--disable-texture", action="store_true", help="Skip LBP texture scoring for more FPS.")
    return parser.parse_args()


def resize_keep_aspect(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


def boxes_touch_or_close(box_a, box_b, gap: int) -> bool:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    return not (
        ax2 + gap < bx or bx2 + gap < ax or ay2 + gap < by or by2 + gap < ay
    )


def merge_boxes(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    x1 = min(ax, bx)
    y1 = min(ay, by)
    x2 = max(ax + aw, bx + bw)
    y2 = max(ay + ah, by + bh)
    return (x1, y1, x2 - x1, y2 - y1)


def merge_nearby_boxes(boxes: list, gap: int) -> list:
    if len(boxes) <= 1:
        return boxes

    merged = boxes[:]
    changed = True
    while changed:
        changed = False
        next_boxes = []
        used = [False] * len(merged)

        for i, box in enumerate(merged):
            if used[i]:
                continue
            used[i] = True
            current = box
            group_changed = True

            while group_changed:
                group_changed = False
                for j, other in enumerate(merged):
                    if used[j]:
                        continue
                    if not boxes_touch_or_close(current, other, gap):
                        continue
                    current = merge_boxes(current, other)
                    used[j] = True
                    changed = True
                    group_changed = True

            next_boxes.append(current)

        merged = next_boxes

    return merged


def expand_box(box, frame_shape, pad: int):
    x, y, w, h = box
    frame_h, frame_w = frame_shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame_w, x + w + pad)
    y2 = min(frame_h, y + h + pad)
    return (x1, y1, x2 - x1, y2 - y1)


def build_proposals(frame: np.ndarray, fg_mask: np.ndarray, min_area: int, box_pad: int, merge_gap: int) -> list:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    grow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.dilate(cleaned, grow_kernel, iterations=2)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    proposals = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        proposals.append(expand_box((x, y, w, h), frame.shape, box_pad))
    return merge_nearby_boxes(proposals, merge_gap)


def draw_result(frame: np.ndarray, result: ConfirmResult):
    x, y, w, h = result.bbox
    if result.accepted:
        color = (0, 255, 0)
        label = f"OK {result.confidence:.2f}"
    elif result.flagged:
        color = (0, 200, 255)
        label = f"REVIEW {result.confidence:.2f}"
    else:
        color = (0, 0, 255)
        label = f"REJECT {result.confidence:.2f}"

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        frame,
        label,
        (x, max(y - 8, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )


def main():
    args = parse_args()
    process_every = max(1, args.process_every)
    preview_width = args.max_width

    config = ConfirmerConfig()
    config.grabcut_iters = max(0, args.grabcut_iters)
    if args.disable_texture:
        config.enable_texture = False
    if args.fast:
        preview_width = min(preview_width, 640)
        process_every = max(process_every, 2)
        config.enable_grabcut = False
        config.grabcut_iters = 0
        config.enable_texture = False

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"[error] cannot open video: {args.video}")

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise SystemExit(f"[error] cannot read first frame: {args.video}")

    first_frame = resize_keep_aspect(first_frame, preview_width)
    belt_sample = first_frame.copy()
    confirmer = SegmentConfirmer(
        sku_refs={},
        belt_sample=belt_sample,
        dino_model=lambda crop: torch.zeros(384),
        config=config,
    )
    mog2 = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=True)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    if args.output:
        h, w = first_frame.shape[:2]
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
        print(f"[init] saving -> {args.output}")

    frame_count = 0
    tick_start = cv2.getTickCount()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last_results = []
    last_proposals = []

    try:
        while True:
            if frame_count == 0:
                frame = first_frame.copy()
                ok = True
            else:
                ok, frame = cap.read()
                if ok:
                    frame = resize_keep_aspect(frame, preview_width)

            if not ok:
                break

            frame_count += 1
            fg_mask = mog2.apply(frame)
            fg_mask = np.where(fg_mask == 255, 255, 0).astype(np.uint8)

            if frame_count == 1 or frame_count % process_every == 0:
                proposals = build_proposals(frame, fg_mask, args.min_area, args.box_pad, args.merge_gap)
                results = []
                for track_id, bbox in enumerate(proposals, start=1):
                    result = confirmer.confirm(frame, bbox, track_id, fg_mask)
                    results.append(result)
                last_proposals = proposals
                last_results = results
            else:
                proposals = last_proposals
                results = last_results

            for result in results:
                draw_result(frame, result)

            elapsed = (cv2.getTickCount() - tick_start) / cv2.getTickFrequency()
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            accepted = sum(1 for result in results if result.accepted)
            flagged = sum(1 for result in results if result.flagged)

            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(frame, f"Accepted: {accepted}  Review: {flagged}  Props: {len(proposals)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

            print(
                f"Frame {frame_count}/{total_frames}  FPS={fps:.1f}  proposals={len(proposals)}  accepted={accepted}  review={flagged}",
                end="\r",
            )

            if writer:
                writer.write(frame)

            if not args.no_show:
                cv2.imshow("SegmentConfirmer Demo", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n[run] stopped by user")
                    break

    finally:
        cap.release()
        if writer:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()
        print(f"\n[done] processed {frame_count} frames")


if __name__ == "__main__":
    main()
