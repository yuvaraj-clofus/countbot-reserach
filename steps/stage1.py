#!/usr/bin/env python3
"""Simple stage 1 selector."""

from __future__ import annotations

import cv2
import numpy as np

PREVIOUS_OBJECTS = []
FASTSAM_BACKEND = None


def color_threshold(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 50, 50], dtype=np.uint8)
    upper = np.array([10, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    return build_output(img, mask, "color")


def mog2(
    img,
    bg_subtractor,
    binary_threshold=200,
    min_area=1200,
    min_width=20,
    min_height=20,
    bbox_padding=0,
    inside_margin=4,
    merge_gap=35,
    line_fraction=0.5,
    center_match_distance=120,
):
    if bg_subtractor is None:
        bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    fg_mask = bg_subtractor.apply(img)
    _, fg_mask = cv2.threshold(fg_mask, int(binary_threshold), 255, cv2.THRESH_BINARY)
    return build_output(
        img,
        fg_mask,
        "mog2",
        min_area=min_area,
        min_width=min_width,
        min_height=min_height,
        bbox_padding=bbox_padding,
        inside_margin=inside_margin,
        merge_gap=merge_gap,
        line_fraction=line_fraction,
        center_match_distance=center_match_distance,
    )


def nanosam(img):
    mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
    return build_output(img, mask, "nanosam")


def fastsam(
    img,
    fastsam_model_path,
    fastsam_conf_threshold=0.40,
    fastsam_nms_iou=0.45,
    fastsam_imgsz=640,
    fastsam_motion_diff_thresh=30,
    fastsam_min_box_area=500,
    fastsam_max_box_area_ratio=0.50,
    fastsam_max_box_width_ratio=0.70,
    fastsam_max_box_height_ratio=0.70,
    fastsam_dark_roi_thresh=40,
    fastsam_motion_overlap_thresh=0.20,
    fastsam_merge_box_iou_thresh=0.10,
    fastsam_merge_box_gap=20,
    fastsam_merge_axis_overlap=0.35,
    line_fraction=0.5,
    center_match_distance=120,
):
    backend = get_fastsam_backend()
    boxes, motion_mask = backend.detect(
        img,
        engine_path=fastsam_model_path,
        conf_threshold=fastsam_conf_threshold,
        nms_iou=fastsam_nms_iou,
        imgsz=fastsam_imgsz,
        motion_diff_thresh=fastsam_motion_diff_thresh,
        min_box_area=fastsam_min_box_area,
        max_box_area_ratio=fastsam_max_box_area_ratio,
        max_box_width_ratio=fastsam_max_box_width_ratio,
        max_box_height_ratio=fastsam_max_box_height_ratio,
        dark_roi_thresh=fastsam_dark_roi_thresh,
        motion_overlap_thresh=fastsam_motion_overlap_thresh,
        merge_box_iou_thresh=fastsam_merge_box_iou_thresh,
        merge_box_gap=fastsam_merge_box_gap,
        merge_axis_overlap=fastsam_merge_axis_overlap,
    )
    mask = motion_mask if motion_mask is not None else boxes_to_mask(img.shape[:2], boxes)
    return build_output_from_bboxes(
        img,
        mask,
        "fastsam",
        boxes,
        line_fraction=line_fraction,
        center_match_distance=center_match_distance,
    )


handlers = {
    "color": color_threshold,
    "mog2": mog2,
    "nanosam": nanosam,
    "fastsam": fastsam,
}


def select_method(img):
    mean_value = float(np.mean(img))
    if mean_value < 50.0:
        return "mog2"
    if mean_value > 150.0:
        return "color"
    return "nanosam"


def run(
    img,
    method="mog2",
    bg_subtractor=None,
    binary_threshold=200,
    min_area=1200,
    min_width=20,
    min_height=20,
    bbox_padding=0,
    inside_margin=4,
    merge_gap=35,
    line_fraction=0.5,
    center_match_distance=120,
    fastsam_model_path="",
    fastsam_conf_threshold=0.40,
    fastsam_nms_iou=0.45,
    fastsam_imgsz=640,
    fastsam_motion_diff_thresh=30,
    fastsam_min_box_area=500,
    fastsam_max_box_area_ratio=0.50,
    fastsam_max_box_width_ratio=0.70,
    fastsam_max_box_height_ratio=0.70,
    fastsam_dark_roi_thresh=40,
    fastsam_motion_overlap_thresh=0.20,
    fastsam_merge_box_iou_thresh=0.10,
    fastsam_merge_box_gap=20,
    fastsam_merge_axis_overlap=0.35,
):
    selected = method or select_method(img)
    if selected == "auto":
        selected = select_method(img)

    if selected not in handlers:
        raise ValueError(f"unknown stage1 method: {selected}")

    if selected == "mog2":
        return handlers[selected](
            img,
            bg_subtractor,
            binary_threshold=binary_threshold,
            min_area=min_area,
            min_width=min_width,
            min_height=min_height,
            bbox_padding=bbox_padding,
            inside_margin=inside_margin,
            merge_gap=merge_gap,
            line_fraction=line_fraction,
            center_match_distance=center_match_distance,
        )
    if selected == "fastsam":
        return handlers[selected](
            img,
            fastsam_model_path=fastsam_model_path,
            fastsam_conf_threshold=fastsam_conf_threshold,
            fastsam_nms_iou=fastsam_nms_iou,
            fastsam_imgsz=fastsam_imgsz,
            fastsam_motion_diff_thresh=fastsam_motion_diff_thresh,
            fastsam_min_box_area=fastsam_min_box_area,
            fastsam_max_box_area_ratio=fastsam_max_box_area_ratio,
            fastsam_max_box_width_ratio=fastsam_max_box_width_ratio,
            fastsam_max_box_height_ratio=fastsam_max_box_height_ratio,
            fastsam_dark_roi_thresh=fastsam_dark_roi_thresh,
            fastsam_motion_overlap_thresh=fastsam_motion_overlap_thresh,
            fastsam_merge_box_iou_thresh=fastsam_merge_box_iou_thresh,
            fastsam_merge_box_gap=fastsam_merge_box_gap,
            fastsam_merge_axis_overlap=fastsam_merge_axis_overlap,
            line_fraction=line_fraction,
            center_match_distance=center_match_distance,
        )
    return handlers[selected](img)


def build_output(
    img,
    mask,
    method,
    min_area=1200,
    min_width=20,
    min_height=20,
    bbox_padding=0,
    inside_margin=4,
    merge_gap=35,
    line_fraction=0.5,
    center_match_distance=120,
):
    cleaned_mask = clean_mask(mask)
    bboxes = all_bboxes(
        cleaned_mask,
        min_area=min_area,
        min_width=min_width,
        min_height=min_height,
        bbox_padding=bbox_padding,
        inside_margin=inside_margin,
        merge_gap=merge_gap,
    )
    return build_output_from_bboxes(
        img,
        cleaned_mask,
        method,
        bboxes,
        line_fraction=line_fraction,
        center_match_distance=center_match_distance,
    )


def build_output_from_bboxes(img, mask, method, bboxes, line_fraction=0.5, center_match_distance=120):
    height, width = img.shape[:2]
    line_x = int(width * float(line_fraction))
    objects = [{"bbox": bbox, "center": bbox_center(bbox)} for bbox in bboxes]
    triggered_bbox, triggered_center = detect_line_trigger(
        objects,
        line_x=line_x,
        center_match_distance=center_match_distance,
    )
    first_bbox = bboxes[0] if bboxes else None
    center = bbox_center(first_bbox)
    return {
        "type": "mask",
        "method": method,
        "data": mask,
        "bboxes": bboxes,
        "bbox": first_bbox,
        "center": center,
        "frame": img,
        "line_x": line_x,
        "triggered": triggered_bbox is not None,
        "triggered_bbox": triggered_bbox,
        "triggered_center": triggered_center,
    }


def clean_mask(mask):
    if mask is None:
        return None

    cleaned = mask.astype(np.uint8)
    kernel = np.ones((5, 5), dtype=np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.GaussianBlur(cleaned, (5, 5), 0)
    _, cleaned = cv2.threshold(cleaned, 127, 255, cv2.THRESH_BINARY)
    return cleaned


def get_fastsam_backend():
    global FASTSAM_BACKEND
    if FASTSAM_BACKEND is not None:
        return FASTSAM_BACKEND

    try:
        import fastsam_backend
    except Exception as exc:
        raise RuntimeError(f"FastSAM backend is not available: {exc}") from exc

    FASTSAM_BACKEND = fastsam_backend
    return FASTSAM_BACKEND


def boxes_to_mask(shape, boxes):
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for bbox in boxes:
        x1, y1, x2, y2 = [int(value) for value in bbox]
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    return mask


def box_area(bbox):
    x1, y1, x2, y2 = [int(value) for value in bbox]
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = [int(value) for value in box_a]
    bx1, by1, bx2, by2 = [int(value) for value in box_b]
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = box_area(box_a)
    area_b = box_area(box_b)
    return inter / max(1, area_a + area_b - inter)


def center_distance(center_a, center_b):
    ax, ay = [int(value) for value in center_a]
    bx, by = [int(value) for value in center_b]
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


def crossed_line(prev_x, current_x, line_x):
    if prev_x == current_x:
        return False
    return (prev_x < line_x <= current_x) or (prev_x > line_x >= current_x)


def find_previous_match(obj, previous_objects, used_previous, center_match_distance):
    best_index = -1
    best_iou = 0.0
    best_distance = float("inf")

    for index, previous in enumerate(previous_objects):
        if index in used_previous:
            continue

        iou = box_iou(obj["bbox"], previous["bbox"])
        distance = center_distance(obj["center"], previous["center"])
        if iou <= 0.0 and distance > float(center_match_distance):
            continue

        if iou > best_iou or (abs(iou - best_iou) < 1e-6 and distance < best_distance):
            best_index = index
            best_iou = iou
            best_distance = distance

    return best_index


def detect_line_trigger(objects, line_x, center_match_distance):
    global PREVIOUS_OBJECTS

    triggered_bbox = None
    triggered_center = None
    best_trigger_distance = float("inf")
    used_previous = set()
    current_objects = []

    for obj in objects:
        current_objects.append({"bbox": obj["bbox"], "center": obj["center"]})
        match_index = find_previous_match(obj, PREVIOUS_OBJECTS, used_previous, center_match_distance)
        if match_index < 0:
            continue

        used_previous.add(match_index)
        previous_center = PREVIOUS_OBJECTS[match_index]["center"]
        current_center = obj["center"]

        if not crossed_line(int(previous_center[0]), int(current_center[0]), int(line_x)):
            continue

        trigger_distance = abs(int(current_center[0]) - int(line_x))
        if trigger_distance < best_trigger_distance:
            best_trigger_distance = trigger_distance
            triggered_bbox = obj["bbox"]
            triggered_center = obj["center"]

    PREVIOUS_OBJECTS = current_objects
    return triggered_bbox, triggered_center


def box_contains(outer, inner, margin=0):
    ox1, oy1, ox2, oy2 = [int(value) for value in outer]
    ix1, iy1, ix2, iy2 = [int(value) for value in inner]
    pad = max(0, int(margin))
    return (
        ix1 >= (ox1 - pad)
        and iy1 >= (oy1 - pad)
        and ix2 <= (ox2 + pad)
        and iy2 <= (oy2 + pad)
    )


def remove_inner_boxes(boxes, inside_margin=4):
    if len(boxes) <= 1:
        return boxes

    kept = []
    ordered = sorted(boxes, key=box_area, reverse=True)
    for idx, box in enumerate(ordered):
        is_inner = False
        for other_idx, other in enumerate(ordered):
            if idx == other_idx:
                continue
            if box_contains(other, box, inside_margin):
                is_inner = True
                break
        if not is_inner:
            kept.append(box)
    return kept


def boxes_should_merge(box_a, box_b, gap):
    ax1, ay1, ax2, ay2 = [int(value) for value in box_a]
    bx1, by1, bx2, by2 = [int(value) for value in box_b]

    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    if inter_w > 0 and inter_h > 0:
        return True

    gap_x = max(0, max(ax1, bx1) - min(ax2, bx2))
    gap_y = max(0, max(ay1, by1) - min(ay2, by2))
    overlap_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    overlap_h = max(0, min(ay2, by2) - max(ay1, by1))
    min_w = max(1, min(ax2 - ax1, bx2 - bx1))
    min_h = max(1, min(ay2 - ay1, by2 - by1))

    if gap_x <= int(gap) and (overlap_h / min_h) >= 0.25:
        return True
    if gap_y <= int(gap) and (overlap_w / min_w) >= 0.25:
        return True
    return False


def merge_boxes(boxes, gap=35):
    if len(boxes) <= 1:
        return boxes

    merged = [list(box) for box in boxes]
    changed = True
    while changed:
        changed = False
        used = [False] * len(merged)
        next_boxes = []

        for idx, box in enumerate(merged):
            if used[idx]:
                continue

            used[idx] = True
            mx1, my1, mx2, my2 = [int(value) for value in box]

            for other_idx in range(idx + 1, len(merged)):
                if used[other_idx]:
                    continue
                if not boxes_should_merge((mx1, my1, mx2, my2), merged[other_idx], gap):
                    continue

                ox1, oy1, ox2, oy2 = [int(value) for value in merged[other_idx]]
                mx1 = min(mx1, ox1)
                my1 = min(my1, oy1)
                mx2 = max(mx2, ox2)
                my2 = max(my2, oy2)
                used[other_idx] = True
                changed = True

            next_boxes.append([mx1, my1, mx2, my2])

        merged = next_boxes

    return merged


def all_bboxes(
    mask,
    min_area=1200,
    max_area_ratio=0.70,
    min_width=20,
    min_height=20,
    bbox_padding=0,
    inside_margin=4,
    merge_gap=35,
):
    if mask is None:
        return []

    height, width = mask.shape[:2]
    max_area = float(height * width) * float(max_area_ratio)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < float(min_area) or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < int(min_width) or h < int(min_height):
            continue

        pad = max(0, int(bbox_padding))
        result.append([
            max(0, int(x - pad)),
            max(0, int(y - pad)),
            min(width, int(x + w + pad)),
            min(height, int(y + h + pad)),
        ])

    result = remove_inner_boxes(result, inside_margin=inside_margin)
    result = merge_boxes(result, gap=merge_gap)
    result = remove_inner_boxes(result, inside_margin=inside_margin)
    result = sorted(result, key=box_area, reverse=True)
    return result


def bbox_center(bbox):
    if bbox is None:
        return None

    x1, y1, x2, y2 = [int(value) for value in bbox]
    return [int((x1 + x2) / 2), int((y1 + y2) / 2)]


def reset_state():
    global PREVIOUS_OBJECTS
    PREVIOUS_OBJECTS = []
    if FASTSAM_BACKEND is not None:
        FASTSAM_BACKEND.reset_state()


def main():
    print("stage1.py is a selector module. Use it from main.py.")


if __name__ == "__main__":
    main()
