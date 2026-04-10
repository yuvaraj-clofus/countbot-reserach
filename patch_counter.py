#!/usr/bin/env python3
"""
patch_counter.py — Patch-based DINOv2 conveyor belt counter.

Approach:
    1. Extract a narrow vertical strip at the counting line (right side of belt).
    2. Decompose into small patches stacked vertically.
    3. Pre-compute DINOv2 embeddings for each patch position from empty belt images.
    4. At runtime, compare each live patch embedding to its reference bank.
    5. If enough patches deviate from the empty belt → material detected → count.

No background subtraction, no contour detection, no centroid tracking.
Pure semantic patch comparison via DINOv2.

Controls:  q=quit  r=reset  space=pause/resume  s=step  d=debug-sims
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import glob
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
VIDEO_PATH   = "test.mp4"
BELT_IMG_DIR = "images/empty_belt"
TRAINING_DIR = "training/patch_embeddings"

# ── DINOv2 ────────────────────────────────────────────────────────────────────
DINO_MODEL = "dinov2_vits14"

# ── Patch grid ────────────────────────────────────────────────────────────────
STRIP_X_FRACTION = 0.85       # counting line at 85% of frame width
STRIP_WIDTH      = 25         # pixel width of vertical column
PATCH_HEIGHT     = 25         # height of each patch
STRIP_SPACING    = 40         # px between strip A and strip B (B is STRIP_SPACING left of A)
NUM_STRIPS_PER_LINE = 3       # strips per line group (start and end)
STRIP_GROUP_STEP    = 12      # px spacing between adjacent strips within a line group
CONSENSUS_MIN_STRIPS = 2      # patch triggers on a line only if this many strips agree

# ── Detection thresholds ──────────────────────────────────────────────────────
SIM_THRESHOLD_MARGIN = 0.05   # margin below per-patch min self-similarity
MIN_CLUSTER_SIZE     = 2      # min contiguous triggered patches to form a cluster
MIN_TEMPORAL_HITS    = 3      # start line must be seen in N processed frames before arming

# ── Timing ────────────────────────────────────────────────────────────────────
PROCESS_EVERY_N  = 3          # process every Nth frame
DEBOUNCE_FRAMES  = 10         # cooldown per cluster slot after count
WARMUP_FRAMES    = 5          # skip initial frames
CLUSTER_MAX_GAP  = 6          # max gap (in patch indices) to bridge within a cluster (handles hollow objects like wire coils)
MATCH_DIST_PATCH = 6          # max vertical center distance to match start/end clusters
POST_COUNT_BLOCK_MULT = 2     # keep counted centers blocked for DEBOUNCE_FRAMES * this factor
LINE_CLUSTER_MAX_GAP = 3      # stricter gap for start/end line clusters (helps split parallel objects)
SPLIT_ASSOC_DIST = 4          # start-cluster distance to treat as possible split of same object
SPLIT_CONFIRM_FRAMES = 3      # split must persist this many processed frames before spawning new track

# ── Display ───────────────────────────────────────────────────────────────────
DISPLAY_MAX_W = 1400


# ═══════════════════════════════════════════════════════════════════════════════
#  DINOv2 helpers (reused from dino_counter.py)
# ═══════════════════════════════════════════════════════════════════════════════

_dino_tfm = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_dino(device):
    print("[DINO] Loading DINOv2...")
    m = torch.hub.load("facebookresearch/dinov2", DINO_MODEL, verbose=False)
    m.eval().to(device)
    print(f"[DINO] Ready on {device}")
    return m


@torch.no_grad()
def _forward_patch_tokens(model, batch):
    """
    Return DINO patch tokens for a batch of 224x224 inputs.
    Output shape: (B, num_tokens, embed_dim), L2-normalized along embed_dim.
    """
    if hasattr(model, "forward_features"):
        feats = model.forward_features(batch)
        if isinstance(feats, dict):
            tokens = feats["x_norm_patchtokens"]
        else:
            tokens = feats
    else:
        tokens = model.get_intermediate_layers(batch, n=1, reshape=False)[0]
    return F.normalize(tokens, dim=2)


def _tokens_to_vertical_patch_embeddings(tokens, num_patches):
    """
    Convert DINO token grid to vertical patch embeddings.
    Tokens from 224x224 (typically 16x16) are upsampled along height to num_patches,
    then averaged over width to produce one embedding per vertical patch.
    """
    bsz, tok_n, dim = tokens.shape
    side = int(tok_n ** 0.5)
    if side * side != tok_n:
        raise RuntimeError(f"Unexpected token count {tok_n}; cannot reshape to square grid")
    grid = tokens.view(bsz, side, side, dim).permute(0, 3, 1, 2)  # (B, D, Ht, Wt)
    up = F.interpolate(grid, size=(num_patches, side), mode="bilinear", align_corners=False)
    patch_embs = up.mean(dim=3).permute(0, 2, 1)  # (B, P, D)
    return F.normalize(patch_embs, dim=2)


@torch.no_grad()
def get_multi_strip_embeddings(model, strips_bgr, num_patches, device):
    """Embed N strips in one DINO forward pass at 224x224 each."""
    tensors = [_dino_tfm(cv2.cvtColor(s, cv2.COLOR_BGR2RGB)) for s in strips_bgr]
    batch = torch.stack(tensors).to(device)
    tokens = _forward_patch_tokens(model, batch)
    patch_embs = _tokens_to_vertical_patch_embeddings(tokens, num_patches)  # (N, P, D)
    return [patch_embs[i] for i in range(patch_embs.shape[0])]


# ═══════════════════════════════════════════════════════════════════════════════
#  PatchExtractor
# ═══════════════════════════════════════════════════════════════════════════════

class PatchExtractor:
    def __init__(self, end_positions, start_positions, strip_width, patch_height, frame_height):
        self.end_positions = end_positions      # end line group (right side)
        self.start_positions = start_positions  # start line group (left side)
        self.strip_width = strip_width
        self.patch_height = patch_height
        self.frame_height = frame_height
        self._num_patches = frame_height // patch_height

    @property
    def num_patches(self):
        return self._num_patches

    def extract_strip(self, frame, x):
        return frame[:, x : x + self.strip_width].copy()

    def extract_end_strips(self, frame):
        return [self.extract_strip(frame, x) for x in self.end_positions]

    def extract_start_strips(self, frame):
        return [self.extract_strip(frame, x) for x in self.start_positions]

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
        # Per-line strip banks:
        # end_embeddings/start_embeddings are lists with one Tensor(P, I, D) per strip.
        self.end_embeddings = []
        self.start_embeddings = []
        self.end_thresholds = []   # list[Tensor(P,)]
        self.start_thresholds = [] # list[Tensor(P,)]
        self.num_patches = 0
        self.num_images = 0
        self.num_end_strips = 0
        self.num_start_strips = 0

    def build(self, model, belt_images, extractor, device):
        """Build reference embeddings for all start/end strips from empty belt images."""
        self.num_patches = extractor.num_patches
        self.num_images = len(belt_images)
        self.num_end_strips = len(extractor.end_positions)
        self.num_start_strips = len(extractor.start_positions)
        all_end = [[] for _ in range(self.num_end_strips)]
        all_start = [[] for _ in range(self.num_start_strips)]
        for idx, img in enumerate(belt_images):
            end_strips = extractor.extract_end_strips(img)
            start_strips = extractor.extract_start_strips(img)
            all_strips = end_strips + start_strips
            all_embs = get_multi_strip_embeddings(model, all_strips, self.num_patches, device)
            end_embs = all_embs[:self.num_end_strips]
            start_embs = all_embs[self.num_end_strips:]
            for si, e in enumerate(end_embs):
                all_end[si].append(e)
            for si, e in enumerate(start_embs):
                all_start[si].append(e)
            print(f"  [TRAIN] Image {idx+1}/{self.num_images} embedded ({self.num_end_strips + self.num_start_strips} strips)")

        self.end_embeddings = [torch.stack(x, dim=0).permute(1, 0, 2) for x in all_end]
        self.start_embeddings = [torch.stack(x, dim=0).permute(1, 0, 2) for x in all_start]
        self._compute_thresholds()
        print(f"  [TRAIN] Bank shape: end={len(self.end_embeddings)} strips, start={len(self.start_embeddings)} strips")

    @torch.no_grad()
    def _compute_thresholds(self):
        """Compute per-patch adaptive thresholds for all strips."""
        self.end_thresholds = [self._thresholds_for(e) for e in self.end_embeddings]
        self.start_thresholds = [self._thresholds_for(e) for e in self.start_embeddings]

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
            "end_embeddings": [e.cpu() for e in self.end_embeddings],
            "start_embeddings": [e.cpu() for e in self.start_embeddings],
            "end_thresholds": [t.cpu() for t in self.end_thresholds],
            "start_thresholds": [t.cpu() for t in self.start_thresholds],
            "num_patches": self.num_patches,
            "num_images": self.num_images,
            "num_end_strips": self.num_end_strips,
            "num_start_strips": self.num_start_strips,
            "strip_width": STRIP_WIDTH,
            "strip_spacing": STRIP_SPACING,
            "patch_height": PATCH_HEIGHT,
            "model": DINO_MODEL,
            "embedding_mode": "dino224_tokens_multistrip_v2",
        }
        torch.save(data, path)
        print(f"  [TRAIN] Saved to {path}")

    def load(self, path, device):
        data = torch.load(path, map_location=device, weights_only=True)
        if (data["strip_width"] != STRIP_WIDTH or
            data["patch_height"] != PATCH_HEIGHT or
            data["model"] != DINO_MODEL or
            data.get("strip_spacing") != STRIP_SPACING or
            data.get("embedding_mode") != "dino224_tokens_multistrip_v2"):
            return False
        self.end_embeddings = [e.to(device) for e in data["end_embeddings"]]
        self.start_embeddings = [e.to(device) for e in data["start_embeddings"]]
        self.end_thresholds = [t.to(device) for t in data["end_thresholds"]]
        self.start_thresholds = [t.to(device) for t in data["start_thresholds"]]
        self.num_patches = data["num_patches"]
        self.num_images = data["num_images"]
        self.num_end_strips = data.get("num_end_strips", len(self.end_embeddings))
        self.num_start_strips = data.get("num_start_strips", len(self.start_embeddings))
        print(f"  [TRAIN] Loaded cached multi-strip bank: end={self.num_end_strips}, start={self.num_start_strips}")
        return True

    @staticmethod
    def exists(path):
        return os.path.isfile(path)

    @torch.no_grad()
    def _compare_line(self, live_embs_list, ref_list, thr_list):
        sims_list = []
        trig_list = []
        for live, refs, thr in zip(live_embs_list, ref_list, thr_list):
            sims = torch.sum(live.unsqueeze(1) * refs, dim=2).max(dim=1).values
            trig = sims < thr
            sims_list.append(sims)
            trig_list.append(trig)
        sims_stack = torch.stack(sims_list, dim=0)
        trig_stack = torch.stack(trig_list, dim=0)
        thr_stack = torch.stack(thr_list, dim=0)
        return sims_stack, trig_stack, thr_stack

    @torch.no_grad()
    def compare(self, end_live_embs, start_live_embs, consensus_min_strips):
        """
        Compare live patch embeddings from all line strips against their banks.
        Returns:
          display_sims: Tensor(P,), min similarity across all strips
          triggered_or: Tensor(P,) bool, OR of line consensus
          end_consensus: Tensor(P,) bool
          start_consensus: Tensor(P,) bool
          display_thresholds: Tensor(P,) min threshold across all strips
        """
        end_sims, end_trig, end_thr = self._compare_line(
            end_live_embs, self.end_embeddings, self.end_thresholds
        )
        start_sims, start_trig, start_thr = self._compare_line(
            start_live_embs, self.start_embeddings, self.start_thresholds
        )

        end_consensus_n = min(consensus_min_strips, end_trig.shape[0])
        start_consensus_n = min(consensus_min_strips, start_trig.shape[0])
        end_consensus = end_trig.sum(dim=0) >= end_consensus_n
        start_consensus = start_trig.sum(dim=0) >= start_consensus_n
        triggered_or = end_consensus | start_consensus

        all_sims = torch.cat([end_sims, start_sims], dim=0)
        all_thr = torch.cat([end_thr, start_thr], dim=0)
        display_sims = all_sims.min(dim=0).values
        display_thresholds = all_thr.min(dim=0).values
        return display_sims, triggered_or, end_consensus, start_consensus, display_thresholds

    def print_stats(self):
        """Print per-patch adaptive thresholds summary across all strips."""
        print("\n[STATS] Per-patch adaptive thresholds (min/max across strips):")
        print(f"{'Patch':>6} {'ThrMin':>8} {'ThrMax':>8}")
        print("-" * 26)
        for i in range(self.num_patches):
            vals = [float(t[i]) for t in (self.end_thresholds + self.start_thresholds)]
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


def compute_line_strip_positions(base_x, num_strips, step, frame_width, strip_width):
    """Create clamped x-positions around base_x for one line group."""
    max_x = max(0, frame_width - strip_width)
    base_x = min(max(base_x, 0), max_x)
    center_i = (num_strips - 1) / 2.0
    pos = []
    for i in range(num_strips):
        x = int(round(base_x + (i - center_i) * step))
        x = min(max(x, 0), max_x)
        pos.append(x)
    # Keep order stable and remove duplicates from clamping.
    out = []
    for x in pos:
        if x not in out:
            out.append(x)
    return out


class LineCrossingTracker:
    """
    Two-line crossing tracker:
      Start line = strip B (left), End line = strip A (right).
    A count is produced when a cluster is observed on start, then matched on end.
    """
    def __init__(self, min_start_hits, track_max_age, match_dist,
                 split_assoc_dist=SPLIT_ASSOC_DIST,
                 split_confirm_frames=SPLIT_CONFIRM_FRAMES):
        self.min_start_hits = min_start_hits
        self.track_max_age = track_max_age
        self.match_dist = match_dist
        self.split_assoc_dist = split_assoc_dist
        self.split_confirm_frames = split_confirm_frames
        self.count = 0
        self.state = "IDLE"
        self.tracks = []  # dict(center, start_hits, age, split_streak)
        self.recent_counts = []  # dict(center, ttl)

    def _tick_recent_counts(self):
        keep = []
        for rc in self.recent_counts:
            rc["ttl"] -= 1
            if rc["ttl"] > 0:
                keep.append(rc)
        self.recent_counts = keep

    def _is_center_blocked(self, center):
        for rc in self.recent_counts:
            if abs(rc["center"] - center) <= self.match_dist * 2:
                return True
        return False

    def _match_track(self, center, require_armed=False, used=None):
        best_i = None
        best_dist = float("inf")
        used = used or set()
        for i, t in enumerate(self.tracks):
            if i in used:
                continue
            if require_armed and t["start_hits"] < self.min_start_hits:
                continue
            dist = abs(t["center"] - center)
            if dist <= self.match_dist and dist < best_dist:
                best_dist = dist
                best_i = i
        return best_i

    def update(self, start_clusters, end_clusters):
        """
        start_clusters: list[(start_idx, end_idx, center)] from strip B
        end_clusters:   list[(start_idx, end_idx, center)] from strip A
        """
        self._tick_recent_counts()

        for t in self.tracks:
            t["age"] += 1

        # Pass 1: one-to-one match start clusters to existing tracks.
        matched_tracks = set()
        matched_cluster_idx = set()
        for ci, (_, __, center) in enumerate(start_clusters):
            if self._is_center_blocked(center):
                continue
            ti = self._match_track(center, require_armed=False, used=matched_tracks)
            if ti is None:
                continue
            tr = self.tracks[ti]
            tr["center"] = 0.7 * tr["center"] + 0.3 * center
            tr["start_hits"] += 1
            tr["age"] = 0
            matched_tracks.add(ti)
            matched_cluster_idx.add(ci)

        # Pass 2: unmatched start clusters either spawn a new track or are suppressed
        # as temporary split artifacts near an already-armed track.
        extras_by_track = {}
        parent_for_cluster = {}
        for ci, (_, __, center) in enumerate(start_clusters):
            if ci in matched_cluster_idx or self._is_center_blocked(center):
                continue
            ti = self._match_track(center, require_armed=True, used=set())
            if ti is None:
                continue
            if abs(self.tracks[ti]["center"] - center) <= self.split_assoc_dist:
                parent_for_cluster[ci] = ti
                extras_by_track[ti] = extras_by_track.get(ti, 0) + 1

        for ti, t in enumerate(self.tracks):
            if extras_by_track.get(ti, 0) > 0:
                t["split_streak"] = t.get("split_streak", 0) + 1
            else:
                t["split_streak"] = max(0, t.get("split_streak", 0) - 1)

        for ci, (_, __, center) in enumerate(start_clusters):
            if ci in matched_cluster_idx or self._is_center_blocked(center):
                continue
            parent = parent_for_cluster.get(ci)
            if parent is not None and self.tracks[parent].get("split_streak", 0) <= self.split_confirm_frames:
                continue
            self.tracks.append({"center": center, "start_hits": 1, "age": 0, "split_streak": 0})

        new_counts = 0
        consumed_tracks = set()
        for _, __, center in end_clusters:
            ti = self._match_track(center, require_armed=True, used=consumed_tracks)
            if ti is not None:
                new_counts += 1
                consumed_tracks.add(ti)
                self.recent_counts.append({
                    "center": center,
                    "ttl": self.track_max_age * POST_COUNT_BLOCK_MULT,
                })

        if consumed_tracks:
            self.count += new_counts

        keep = []
        for i, t in enumerate(self.tracks):
            if i in consumed_tracks:
                continue
            if t["age"] <= self.track_max_age:
                keep.append(t)
        self.tracks = keep

        armed = [t["center"] for t in self.tracks if t["start_hits"] >= self.min_start_hits]
        if armed and end_clusters:
            self.state = f"CROSS({len(armed)})"
        elif armed:
            self.state = f"ARMED({len(armed)})"
        elif self.tracks:
            self.state = f"START({len(self.tracks)})"
        else:
            self.state = "IDLE"

        return new_counts, armed

    def reset(self):
        self.count = 0
        self.tracks = []
        self.recent_counts = []
        self.state = "IDLE"


# ═══════════════════════════════════════════════════════════════════════════════
#  Display
# ═══════════════════════════════════════════════════════════════════════════════

def build_display(frame, strip_a, strip_b, patch_sims, triggered, thresholds,
                  count, state, paused, end_positions, start_positions,
                  num_patches, patch_height, debug_sims):
    """
    Build a 4-panel display:
      Panel 1: Main video with both counting lines + overlays
      Panel 2: Scaled-up strip A
      Panel 3: Scaled-up strip B
      Panel 4: Similarity heatmap per patch
    """
    H, W = frame.shape[:2]
    vis = frame.copy()

    # Draw end/start strip groups.
    for x in end_positions:
        cv2.line(vis, (x, 0), (x, H), (0, 220, 220), 2)
    for x in start_positions:
        cv2.line(vis, (x, 0), (x, H), (220, 220, 0), 2)

    # Highlight triggered patches on the frame (span between both strips)
    if triggered is not None:
        overlay = vis.copy()
        all_x = end_positions + start_positions
        x_lo = min(all_x)
        x_hi = max(all_x) + STRIP_WIDTH
        for i in range(num_patches):
            if triggered[i]:
                y0 = i * patch_height
                y1 = y0 + patch_height
                cv2.rectangle(overlay, (x_lo, y0), (x_hi, y1), (0, 0, 255), -1)
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

    # Panel 2 & 3: Patch columns A and B (scaled up for visibility)
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

    strip_up_a = _make_strip_panel(strip_a, "A")
    strip_up_b = _make_strip_panel(strip_b, "B")

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
    combined = np.hstack([vis, strip_up_a, strip_up_b, heatmap])
    scale = min(1.0, DISPLAY_MAX_W / combined.shape[1])
    if scale < 1.0:
        combined = cv2.resize(combined, (int(combined.shape[1] * scale),
                                         int(combined.shape[0] * scale)))
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    device = ("mps"  if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available()       else "cpu")
    print(f"[INFO] Device: {device}")

    model = load_dino(device)

    # Open video for dimensions
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    strip_x_end_base = int(W * STRIP_X_FRACTION)
    strip_x_start_base = strip_x_end_base - STRIP_SPACING
    end_positions = compute_line_strip_positions(
        strip_x_end_base, NUM_STRIPS_PER_LINE, STRIP_GROUP_STEP, W, STRIP_WIDTH
    )
    start_positions = compute_line_strip_positions(
        strip_x_start_base, NUM_STRIPS_PER_LINE, STRIP_GROUP_STEP, W, STRIP_WIDTH
    )
    extractor = PatchExtractor(end_positions, start_positions, STRIP_WIDTH, PATCH_HEIGHT, H)
    num_patches = extractor.num_patches

    print(f"\nVideo       : {W}x{H} @ {fps:.1f} fps")
    print(f"End strips  : {end_positions}")
    print(f"Start strips: {start_positions}  (base spacing={STRIP_SPACING}px)")
    print(f"Consensus   : {CONSENSUS_MIN_STRIPS}/{NUM_STRIPS_PER_LINE} per line")
    print(f"Patch grid  : {num_patches} patches of {PATCH_HEIGHT}x{STRIP_WIDTH}")
    print(f"Thresh margin: {SIM_THRESHOLD_MARGIN}  (adaptive per-patch)\n")

    # ── Training phase ────────────────────────────────────────────────────────
    bank_path = os.path.join(TRAINING_DIR, "patch_bank.pt")
    bank = ReferenceBank()

    if bank.exists(bank_path):
        if not bank.load(bank_path, device):
            print("  [TRAIN] Config mismatch, rebuilding...")
            bank = ReferenceBank()

    if not bank.end_embeddings:
        print("[TRAIN] Building reference bank from empty belt images...")
        paths = sorted(glob.glob(os.path.join(BELT_IMG_DIR, "*.png")) +
                       glob.glob(os.path.join(BELT_IMG_DIR, "*.jpg")))
        if not paths:
            print(f"ERROR: No images in {BELT_IMG_DIR}")
            return
        belt_images = []
        for p in paths:
            img = cv2.imread(p)
            if img is not None:
                belt_images.append(img)
        if not belt_images:
            print("ERROR: Could not load any belt images")
            return
        bank.build(model, belt_images, extractor, device)
        bank.save(bank_path)

    bank.print_stats()

    # ── Line-crossing tracker (start=B -> end=A) ────────────────────────────
    counter = LineCrossingTracker(
        min_start_hits=MIN_TEMPORAL_HITS,
        track_max_age=DEBOUNCE_FRAMES,
        match_dist=MATCH_DIST_PATCH,
        split_assoc_dist=SPLIT_ASSOC_DIST,
        split_confirm_frames=SPLIT_CONFIRM_FRAMES,
    )

    frame_n = 0
    paused = False
    debug_sims = False
    last_sims = None
    last_triggered = None
    last_end_strip = None
    last_start_strip = None
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
                last_end_strip = last_start_strip = None
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
                disp = build_display(frame, last_end_strip, last_start_strip,
                                     last_sims, last_triggered, display_thresholds,
                                     counter.count, counter.state, paused,
                                     end_positions, start_positions, num_patches,
                                     PATCH_HEIGHT, debug_sims)
                cv2.imshow("PatchCount", disp)
                continue

        # ── Warmup ────────────────────────────────────────────────────────────
        if frame_n <= WARMUP_FRAMES:
            vis = frame.copy()
            bar_w = int((frame_n / WARMUP_FRAMES) * W)
            cv2.rectangle(vis, (0, H - 20), (bar_w, H), (0, 200, 255), -1)
            cv2.putText(vis, f"Warming up... {frame_n}/{WARMUP_FRAMES}",
                        (10, H - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            cv2.imshow("PatchCount", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # ── Process every Nth frame ───────────────────────────────────────────
        if frame_n % PROCESS_EVERY_N == 0 or last_sims is None:
            end_strips = extractor.extract_end_strips(frame)
            start_strips = extractor.extract_start_strips(frame)
            all_embs = get_multi_strip_embeddings(model, end_strips + start_strips, num_patches, device)
            end_embs = all_embs[:len(end_positions)]
            start_embs = all_embs[len(end_positions):]
            max_sims, triggered_tensor, trig_end_tensor, trig_start_tensor, display_thresholds = bank.compare(
                end_embs, start_embs, CONSENSUS_MIN_STRIPS
            )

            triggered = triggered_tensor.cpu().tolist()   # OR trigger for display
            triggered_end = trig_end_tensor.cpu().tolist()
            triggered_start = trig_start_tensor.cpu().tolist()
            start_clusters = find_clusters(triggered_start, max_gap=LINE_CLUSTER_MAX_GAP)
            end_clusters = find_clusters(triggered_end, max_gap=LINE_CLUSTER_MAX_GAP)

            last_sims = max_sims
            last_triggered = triggered
            last_end_strip = end_strips[0] if end_strips else None
            last_start_strip = start_strips[0] if start_strips else None

            new_counts, armed_centers = counter.update(start_clusters, end_clusters)
            if new_counts > 0:
                n_triggered = sum(triggered)
                cluster_info = ", ".join(f"{c:.0f}" for c in armed_centers)
                print(f"[COUNT {counter.count:>4}]  +{new_counts}  "
                      f"{n_triggered}/{num_patches} patches  "
                      f"armed=[{cluster_info}] frame={frame_n}")

            if debug_sims:
                sims_np = max_sims.cpu().numpy()
                trig_end_idx = [i for i, t in enumerate(triggered_end) if t]
                trig_start_idx = [i for i, t in enumerate(triggered_start) if t]
                print(f"  frame={frame_n}  trigEnd={trig_end_idx} trigStart={trig_start_idx}  "
                      f"start={[(s,e) for s,e,_ in start_clusters]}  "
                      f"end={[(s,e) for s,e,_ in end_clusters]}  "
                      f"min_sim={sims_np.min():.3f}  "
                      f"mean_sim={sims_np.mean():.3f}")

        # ── Display ───────────────────────────────────────────────────────────
        disp = build_display(frame, last_end_strip, last_start_strip,
                             last_sims, last_triggered, display_thresholds,
                             counter.count, counter.state, paused,
                             end_positions, start_positions, num_patches,
                             PATCH_HEIGHT, debug_sims)
        cv2.imshow("PatchCount", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            counter.reset()
            last_sims = None
            last_triggered = None
            last_end_strip = last_start_strip = None
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
