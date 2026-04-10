#!/usr/bin/env python3
"""
FastSAM + SegmentConfirmer counting pipeline.

Pipeline:
  1. FastSAM detects product boxes
  2. Nearby product fragments are already merged by FastSAM detect.py
  3. A simple centroid tracker keeps stable IDs
  4. When a track crosses the count line, SegmentConfirmer validates it
  5. Only accepted tracks increase the final count

  python3 /home/vikbot/Documents/countbot/testing/fastsam_confirm_count.py \
  --source /home/vikbot/Documents/countbot/testing/video/bearing_wheel.mp4 \
  --engine /home/vikbot/Documents/countbot/testing/FastSAM/weights/FastSAM-s.engine \
  --no-save

python3 /home/vikbot/Documents/countbot/testing/fastsam_confirm_count.py \
  --source 0 \
  --engine /home/vikbot/Documents/countbot/testing/FastSAM/weights/FastSAM-s.engine \
  --no-save

  python3 /home/vikbot/Documents/countbot/testing/fastsam_confirm_count.py \
  --source 0 \
  --engine /home/vikbot/Documents/countbot/testing/FastSAM/weights/FastSAM-s.engine \
  --direction left \
  --line-x 50 \
  --no-save


"""

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


DEFAULT_SOURCE = "/home/vikbot/Documents/countbot/testing/video/bearing_wheel.mp4"
DEFAULT_ENGINE = "/home/vikbot/Documents/countbot/testing/FastSAM/weights/FastSAM-s.engine"
DEFAULT_OUTPUT = "/home/vikbot/Documents/countbot/testing/fastsam_confirm_count_output.mp4"
DEFAULT_REFS = "/home/vikbot/Documents/countbot/testing/models/2wheel.pt"


@dataclass
class TrackState:
    track_id: int
    bbox: Tuple[int, int, int, int, float]
    centroid: Tuple[int, int]
    prev_centroid: Optional[Tuple[int, int]] = None
    missed: int = 0
    counted: bool = False
    decision_made: bool = False
    status: str = "TRACK"
    status_conf: float = 0.0


class SimpleTracker:
    def __init__(self, max_distance: int = 90, max_missed: int = 12):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: Dict[int, TrackState] = {}

    @staticmethod
    def _centroid(bbox: Tuple[int, int, int, int, float]) -> Tuple[int, int]:
        x1, y1, x2, y2, _ = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _register(self, bbox: Tuple[int, int, int, int, float]) -> None:
        self.tracks[self.next_id] = TrackState(
            track_id=self.next_id,
            bbox=bbox,
            centroid=self._centroid(bbox),
        )
        self.next_id += 1

    def update(self, detections: List[Tuple[int, int, int, int, float]]) -> Dict[int, TrackState]:
        if not detections:
            stale_ids = []
            for track_id, track in self.tracks.items():
                track.missed += 1
                if track.missed > self.max_missed:
                    stale_ids.append(track_id)
            for track_id in stale_ids:
                del self.tracks[track_id]
            return self.tracks

        if not self.tracks:
            for bbox in detections:
                self._register(bbox)
            return self.tracks

        detection_centroids = [self._centroid(bbox) for bbox in detections]
        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[track_id].centroid for track_id in track_ids]

        distances = []
        for row, track_centroid in enumerate(track_centroids):
            for col, det_centroid in enumerate(detection_centroids):
                dist = ((track_centroid[0] - det_centroid[0]) ** 2 + (track_centroid[1] - det_centroid[1]) ** 2) ** 0.5
                distances.append((dist, row, col))

        matched_tracks = set()
        matched_detections = set()
        for dist, row, col in sorted(distances, key=lambda item: item[0]):
            if row in matched_tracks or col in matched_detections:
                continue
            if dist > self.max_distance:
                continue

            track_id = track_ids[row]
            track = self.tracks[track_id]
            track.prev_centroid = track.centroid
            track.centroid = detection_centroids[col]
            track.bbox = detections[col]
            track.missed = 0
            matched_tracks.add(row)
            matched_detections.add(col)

        stale_ids = []
        for row, track_id in enumerate(track_ids):
            if row in matched_tracks:
                continue
            track = self.tracks[track_id]
            track.missed += 1
            if track.missed > self.max_missed:
                stale_ids.append(track_id)
        for track_id in stale_ids:
            del self.tracks[track_id]

        for col, bbox in enumerate(detections):
            if col not in matched_detections:
                self._register(bbox)

        return self.tracks


def parse_args():
    parser = argparse.ArgumentParser(description="FastSAM detect -> confirm -> count pipeline.")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Video file or camera source. Use 0 for webcam.")
    parser.add_argument("--engine", default=DEFAULT_ENGINE, help="FastSAM TensorRT engine path.")
    parser.add_argument("--refs", default=DEFAULT_REFS, help="Reference embeddings .pt file for DINOv2 matching.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Annotated output video path.")
    parser.add_argument("--no-save", action="store_true", help="Disable saving output video.")
    parser.add_argument("--no-show", action="store_true", help="Disable live preview window.")
    parser.add_argument("--width", type=int, default=640, help="Resize width for processing.")
    parser.add_argument("--height", type=int, default=480, help="Resize height for processing.")
    parser.add_argument("--line-y", type=float, default=58.0, help="Horizontal line position as percent of frame height.")
    parser.add_argument("--line-x", type=float, default=50.0, help="Vertical line position as percent of frame width.")
    parser.add_argument(
        "--direction",
        choices=("down", "up", "left", "right"),
        default="down",
        help="Crossing direction that triggers confirmation.",
    )
    parser.add_argument("--max-track-distance", type=int, default=90, help="Maximum centroid distance for track matching.")
    parser.add_argument("--max-track-missed", type=int, default=12, help="Drop a track after this many missing frames.")
    parser.add_argument("--match-threshold", type=float, default=0.75, help="Cosine similarity needed for OK match.")
    parser.add_argument("--review-threshold", type=float, default=0.60, help="Cosine similarity needed for REVIEW.")
    return parser.parse_args()


def load_fastsam_module():
    module_path = Path(__file__).resolve().parent / "FastSAM" / "detect.py"
    spec = importlib.util.spec_from_file_location("fastsam_detect_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load FastSAM helper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def crossed_line(prev_centroid: Tuple[int, int], centroid: Tuple[int, int], line_pos: int, direction: str) -> bool:
    if direction == "down":
        return prev_centroid[1] < line_pos <= centroid[1]
    if direction == "up":
        return prev_centroid[1] > line_pos >= centroid[1]
    if direction == "right":
        return prev_centroid[0] < line_pos <= centroid[0]
    return prev_centroid[0] > line_pos >= centroid[0]


def bbox_xywh(bbox: Tuple[int, int, int, int, float]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2, _ = bbox
    return (x1, y1, x2 - x1, y2 - y1)


def load_reference_embeddings(pt_path: str) -> torch.Tensor:
    print(f"[init] loading reference embeddings -> {pt_path}")
    data = torch.load(pt_path, map_location="cpu")
    embeddings = []

    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, torch.Tensor):
                embeddings.append(F.normalize(value.cpu().float(), dim=-1))
    elif isinstance(data, torch.Tensor):
        if data.ndim == 1:
            embeddings = [F.normalize(data.cpu().float(), dim=-1)]
        else:
            embeddings = [F.normalize(row.cpu().float(), dim=-1) for row in data]

    if not embeddings:
        raise ValueError(f"No embeddings found in {pt_path}")

    refs = torch.stack(embeddings, dim=0)
    print(f"[init] {len(refs)} reference embeddings ready")
    return refs


def load_dino():
    print("[init] loading DINOv2 model...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("[init] DINOv2 ready")
    return model, tf


def make_segment_crop(frame: np.ndarray, bbox: Tuple[int, int, int, int, float], fg_mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    x1, y1, x2, y2, _ = bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    if fg_mask is None:
        return crop

    crop_mask = fg_mask[y1:y2, x1:x2]
    if crop_mask.size == 0 or np.count_nonzero(crop_mask) < 100:
        return crop

    masked = crop.copy()
    masked[crop_mask == 0] = 0
    return masked


def get_embedding(roi_bgr: np.ndarray, model, tf) -> Optional[torch.Tensor]:
    if roi_bgr is None or roi_bgr.size == 0:
        return None

    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    tensor = tf(Image.fromarray(rgb)).unsqueeze(0)
    with torch.no_grad():
        embedding = model(tensor)
    return F.normalize(embedding, dim=-1).squeeze(0)


def dino_decision(
    roi_bgr: np.ndarray,
    ref_embeddings: torch.Tensor,
    model,
    tf,
    match_threshold: float,
    review_threshold: float,
) -> Tuple[str, float]:
    embedding = get_embedding(roi_bgr, model, tf)
    if embedding is None:
        return "REJECT", 0.0

    cosine = F.cosine_similarity(embedding.unsqueeze(0), ref_embeddings, dim=-1)
    best = float(cosine.max().item())
    if best >= match_threshold:
        return "OK", round(best, 4)
    if best >= review_threshold:
        return "REVIEW", round(best, 4)
    return "REJECT", round(best, 4)


def build_motion_mask(prev_gray: Optional[np.ndarray], gray: np.ndarray, kernel: np.ndarray, threshold: int = 30):
    if prev_gray is None:
        return None
    diff = cv2.absdiff(prev_gray, gray)
    _, motion_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    return motion_mask


def draw_line(frame: np.ndarray, line_pos: int, direction: str):
    h, w = frame.shape[:2]
    if direction in ("down", "up"):
        cv2.line(frame, (0, line_pos), (w, line_pos), (255, 255, 0), 2)
    else:
        cv2.line(frame, (line_pos, 0), (line_pos, h), (255, 255, 0), 2)


def draw_tracks(frame: np.ndarray, tracks: Dict[int, TrackState]):
    colors = {
        "TRACK": (0, 165, 255),
        "OK": (0, 255, 0),
        "REVIEW": (0, 200, 255),
        "REJECT": (0, 0, 255),
    }

    for track in tracks.values():
        if track.missed > 0:
            continue
        x1, y1, x2, y2, conf = track.bbox
        color = colors.get(track.status, (255, 255, 255))
        label = f"ID {track.track_id} {track.status}"
        if track.status == "TRACK":
            label = f"ID {track.track_id} {conf:.2f}"
        elif track.status_conf:
            label = f"ID {track.track_id} {track.status} {track.status_conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, track.centroid, 4, color, -1)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )


def main():
    args = parse_args()
    fastsam = load_fastsam_module()
    model_bundle, device = fastsam.load_model(args.engine)
    ref_embeddings = load_reference_embeddings(args.refs)
    dino_model, dino_tf = load_dino()

    source = 0 if str(args.source) == "0" else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"[error] cannot open source: {args.source}")

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        raise SystemExit(f"[error] cannot read from source: {args.source}")

    first_frame = cv2.resize(first_frame, (args.width, args.height))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.direction in ("down", "up"):
        line_pos = int((args.line_y / 100.0) * args.height)
        line_desc = f"horizontal@{args.line_y:.1f}%"
    else:
        line_pos = int((args.line_x / 100.0) * args.width)
        line_desc = f"vertical@{args.line_x:.1f}%"

    tracker = SimpleTracker(args.max_track_distance, args.max_track_missed)
    motion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    writer = None
    if args.output and not args.no_save:
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (args.width, args.height))
        print(f"[init] saving -> {args.output}")

    print(f"[init] source={args.source} size={args.width}x{args.height} fps={src_fps:.1f}")
    print(
        f"[init] line={args.direction} {line_desc}  refs={Path(args.refs).name}  "
        f"ok>={args.match_threshold:.2f} review>={args.review_threshold:.2f}"
    )
    print("[run] press Q to quit")

    frame_count = 0
    total_count = 0
    review_count = 0
    reject_count = 0
    prev_gray = None
    tick_start = cv2.getTickCount()

    try:
        while True:
            if frame_count == 0:
                frame = first_frame.copy()
                ok = True
            else:
                ok, frame = cap.read()
                if ok:
                    frame = cv2.resize(frame, (args.width, args.height))

            if not ok:
                break

            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_mask = build_motion_mask(prev_gray, gray, motion_kernel, fastsam.MOTION_DIFF_THRESH)
            if prev_gray is None:
                fg_mask = None
            else:
                fg_mask = cv2.absdiff(prev_gray, gray)
                _, fg_mask = cv2.threshold(fg_mask, fastsam.MOTION_DIFF_THRESH, 255, cv2.THRESH_BINARY)
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, motion_kernel)
            prev_gray = gray.copy()

            detections = fastsam.detect(model_bundle, device, frame, motion_mask)
            tracks = tracker.update(detections)

            for track in tracks.values():
                if track.missed > 0 or track.prev_centroid is None or track.decision_made:
                    continue
                if not crossed_line(track.prev_centroid, track.centroid, line_pos, args.direction):
                    continue

                track.decision_made = True
                segment_crop = make_segment_crop(frame, track.bbox, fg_mask)
                status, score = dino_decision(
                    segment_crop,
                    ref_embeddings,
                    dino_model,
                    dino_tf,
                    args.match_threshold,
                    args.review_threshold,
                )
                track.status = status
                track.status_conf = score
                if status == "OK":
                    track.counted = True
                    total_count += 1
                elif status == "REVIEW":
                    review_count += 1
                else:
                    reject_count += 1

            out = frame.copy()
            draw_line(out, line_pos, args.direction)
            draw_tracks(out, tracks)

            elapsed = (cv2.getTickCount() - tick_start) / cv2.getTickFrequency()
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            active_tracks = sum(1 for track in tracks.values() if track.missed == 0)

            cv2.putText(out, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            if total_frames > 0:
                cv2.putText(out, f"Frame: {frame_count}/{total_frames}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            else:
                cv2.putText(out, f"Frame: {frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(out, f"Tracks: {active_tracks}  Det: {len(detections)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(out, f"Count: {total_count}  Review: {review_count}  Reject: {reject_count}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 120), 2)

            print(
                f"Frame {frame_count}/{total_frames if total_frames > 0 else '?'}  FPS={fps:.1f}  Det={len(detections)}  Count={total_count}",
                end="\r",
            )

            if writer:
                writer.write(out)

            if not args.no_show:
                cv2.imshow("FastSAM Confirm Count", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n[run] stopped by user")
                    break

    finally:
        cap.release()
        if writer:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()
        print(f"\n[done] processed {frame_count} frames | count={total_count} review={review_count} reject={reject_count}")


if __name__ == "__main__":
    main()
