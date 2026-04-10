#!/usr/bin/env python3
"""Local FastSAM TensorRT backend for steps/stage1.py."""

from __future__ import annotations

import ctypes
import ctypes.util
import inspect
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch
from ultralytics.utils import ops

try:
    from ultralytics.utils import nms as ultralytics_nms
except ImportError:
    ultralytics_nms = ops


NMS_SUPPORTS_AGNOSTIC_NMS = "agnostic_nms" in inspect.signature(
    ultralytics_nms.non_max_suppression
).parameters

ENGINE_CACHE: dict[str, "TRTEngine"] = {}
PREVIOUS_GRAY = None


def _as_void_p(value):
    if isinstance(value, np.ndarray):
        return ctypes.c_void_p(value.ctypes.data)
    if isinstance(value, int):
        return ctypes.c_void_p(value)
    if value is None:
        return ctypes.c_void_p()
    return value


class _CudaMemcpyKind:
    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2


class _CudaRuntimeShim:
    cudaMemcpyKind = _CudaMemcpyKind

    def __init__(self):
        libname = ctypes.util.find_library("cudart")
        if not libname:
            raise RuntimeError("libcudart not found")
        self.lib = ctypes.CDLL(libname)

    def cudaMalloc(self, size):
        ptr = ctypes.c_void_p()
        err = self.lib.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(size))
        return err, ptr.value

    def cudaFree(self, ptr):
        err = self.lib.cudaFree(_as_void_p(ptr))
        return (err,)

    def cudaMemcpy(self, dst, src, nbytes, kind):
        err = self.lib.cudaMemcpy(_as_void_p(dst), _as_void_p(src), ctypes.c_size_t(nbytes), ctypes.c_int(kind))
        return (err,)


cudart = _CudaRuntimeShim()


def cuda_call(call):
    err, res = call[0], call[1:]
    if int(err) != 0:
        raise RuntimeError(f"CUDA error: {err}")
    if len(res) == 1:
        return res[0]
    return res


def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))


def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))


class TRTEngine:
    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, namespace="")
        with open(engine_path, "rb") as handle, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(handle.read())
        self.context = self.engine.create_execution_context()

        self.inputs = []
        self.outputs = []
        self.allocations = []

        for binding in self._iter_bindings():
            is_input = binding["is_input"]
            dtype = binding["dtype"]
            shape = binding["shape"]
            size = dtype.itemsize
            for dim in shape:
                size *= dim
            allocation = cuda_call(cudart.cudaMalloc(size))
            host_allocation = None if is_input else np.zeros(shape, dtype)
            item = {
                "name": binding["name"],
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
                "host_allocation": host_allocation,
            }
            self.allocations.append(allocation)
            if is_input:
                self.inputs.append(item)
            else:
                self.outputs.append(item)

    def _iter_bindings(self):
        if hasattr(self.engine, "num_bindings"):
            for index in range(self.engine.num_bindings):
                is_input = self.engine.binding_is_input(index)
                name = self.engine.get_binding_name(index)
                dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(index)))
                shape = tuple(self.context.get_binding_shape(index))
                if is_input and len(shape) and shape[0] < 0:
                    profile_shape = self.engine.get_profile_shape(0, name)
                    self.context.set_binding_shape(index, profile_shape[2])
                    shape = tuple(self.context.get_binding_shape(index))
                yield {"name": name, "dtype": dtype, "shape": shape, "is_input": is_input}
            return

        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            shape = tuple(self.context.get_tensor_shape(name))
            yield {"name": name, "dtype": dtype, "shape": shape, "is_input": is_input}

    def infer(self, batch: np.ndarray):
        memcpy_host_to_device(self.inputs[0]["allocation"], batch)
        if hasattr(self.context, "set_tensor_address"):
            for binding in self.inputs + self.outputs:
                self.context.set_tensor_address(binding["name"], int(binding["allocation"]))
        self.context.execute_v2(self.allocations)
        for output in self.outputs:
            memcpy_device_to_host(output["host_allocation"], output["allocation"])
        return [output["host_allocation"] for output in self.outputs]


def reset_state():
    global PREVIOUS_GRAY
    PREVIOUS_GRAY = None


def load_engine(engine_path: str):
    engine_path = str(Path(engine_path).resolve())
    if engine_path not in ENGINE_CACHE:
        if not Path(engine_path).exists():
            raise FileNotFoundError(f"FastSAM engine not found: {engine_path}")
        ENGINE_CACHE[engine_path] = TRTEngine(engine_path)
    return ENGINE_CACHE[engine_path]


def preprocess(frame_bgr: np.ndarray, imgsz: int):
    height, width = frame_bgr.shape[:2]
    scale = min(imgsz / max(1, height), imgsz / max(1, width))
    new_width = int(width * scale)
    new_height = int(height * scale)
    padded = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    resized = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), (new_width, new_height))
    pad_y = (imgsz - new_height) // 2
    pad_x = (imgsz - new_width) // 2
    padded[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized
    blob = np.array([padded], dtype=np.float32) / 255.0
    blob = np.ascontiguousarray(np.transpose(blob, (0, 3, 1, 2)))
    return blob


def postprocess(preds, blob_shape, frame, conf_threshold, nms_iou):
    if len(preds) == 2:
        prediction = torch.from_numpy(preds[0])
    elif len(preds) >= 6:
        prediction = torch.from_numpy(preds[5])
    else:
        raise ValueError(f"Unexpected TensorRT output count: {len(preds)}")

    nms_kwargs = {"max_det": 100, "nc": 1}
    if NMS_SUPPORTS_AGNOSTIC_NMS:
        nms_kwargs["agnostic_nms"] = True
    else:
        nms_kwargs["agnostic"] = True
    predictions = ultralytics_nms.non_max_suppression(prediction, conf_threshold, nms_iou, **nms_kwargs)

    for pred in predictions:
        if len(pred):
            pred[:, :4] = ops.scale_boxes(blob_shape[2:], pred[:, :4], frame.shape)
    return predictions


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2, _ = box_a
    bx1, by1, bx2, by2, _ = box_b
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / max(1, area_a + area_b - inter)


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
    return [box for box, flag in zip(bboxes, keep) if flag]


def should_merge_boxes(box_a, box_b, merge_box_iou_thresh, merge_box_gap, merge_axis_overlap):
    ax1, ay1, ax2, ay2, _ = box_a
    bx1, by1, bx2, by2, _ = box_b

    if box_iou(box_a, box_b) >= merge_box_iou_thresh:
        return True

    gap_x = max(0, max(ax1, bx1) - min(ax2, bx2))
    gap_y = max(0, max(ay1, by1) - min(ay2, by2))
    overlap_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    overlap_h = max(0, min(ay2, by2) - max(ay1, by1))
    min_w = max(1, min(ax2 - ax1, bx2 - bx1))
    min_h = max(1, min(ay2 - ay1, by2 - by1))
    horizontal_overlap = overlap_w / min_w
    vertical_overlap = overlap_h / min_h

    if gap_x <= merge_box_gap and vertical_overlap >= merge_axis_overlap:
        return True
    if gap_y <= merge_box_gap and horizontal_overlap >= merge_axis_overlap:
        return True
    return False


def merge_nearby_boxes(bboxes, merge_box_iou_thresh, merge_box_gap, merge_axis_overlap):
    if len(bboxes) <= 1:
        return bboxes

    merged = bboxes[:]
    changed = True
    while changed:
        changed = False
        used = [False] * len(merged)
        next_boxes = []

        for index, box in enumerate(merged):
            if used[index]:
                continue

            used[index] = True
            mx1, my1, mx2, my2, mconf = box
            group_changed = True

            while group_changed:
                group_changed = False
                current = (mx1, my1, mx2, my2, mconf)
                for other_index, other in enumerate(merged):
                    if used[other_index]:
                        continue
                    if not should_merge_boxes(current, other, merge_box_iou_thresh, merge_box_gap, merge_axis_overlap):
                        continue
                    ox1, oy1, ox2, oy2, oconf = other
                    mx1 = min(mx1, ox1)
                    my1 = min(my1, oy1)
                    mx2 = max(mx2, ox2)
                    my2 = max(my2, oy2)
                    mconf = max(mconf, oconf)
                    used[other_index] = True
                    changed = True
                    group_changed = True

            next_boxes.append((mx1, my1, mx2, my2, mconf))

        merged = next_boxes

    return merged


def build_motion_mask(frame, motion_diff_thresh):
    global PREVIOUS_GRAY

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if PREVIOUS_GRAY is None:
        PREVIOUS_GRAY = gray.copy()
        return None

    diff = cv2.absdiff(PREVIOUS_GRAY, gray)
    _, motion_mask = cv2.threshold(diff, int(motion_diff_thresh), 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
    PREVIOUS_GRAY = gray.copy()
    return motion_mask


def detect(
    frame,
    engine_path,
    conf_threshold=0.40,
    nms_iou=0.45,
    imgsz=640,
    motion_diff_thresh=30,
    min_box_area=500,
    max_box_area_ratio=0.50,
    max_box_width_ratio=0.70,
    max_box_height_ratio=0.70,
    dark_roi_thresh=40,
    motion_overlap_thresh=0.20,
    merge_box_iou_thresh=0.10,
    merge_box_gap=20,
    merge_axis_overlap=0.35,
):
    engine = load_engine(engine_path)
    motion_mask = build_motion_mask(frame, motion_diff_thresh)

    height, width = frame.shape[:2]
    frame_area = height * width
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blob = preprocess(frame, int(imgsz))
    preds = engine.infer(blob)
    predictions = postprocess(preds, blob.shape, frame, float(conf_threshold), float(nms_iou))

    raw_boxes = []
    for pred in predictions:
        if len(pred) == 0:
            continue
        for det in pred[:, :6].tolist():
            x1, y1, x2, y2, conf, _ = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            bw = x2 - x1
            bh = y2 - y1
            area = bw * bh

            if area < min_box_area or area > frame_area * max_box_area_ratio:
                continue
            if bw > width * max_box_width_ratio or bh > height * max_box_height_ratio:
                continue

            if motion_mask is not None:
                roi_motion = motion_mask[y1:y2, x1:x2]
                if roi_motion.size > 0 and np.count_nonzero(roi_motion) / roi_motion.size < motion_overlap_thresh:
                    continue

            roi_gray = frame_gray[y1:y2, x1:x2]
            if roi_gray.size > 0 and roi_gray.mean() < dark_roi_thresh:
                continue

            raw_boxes.append((x1, y1, x2, y2, float(conf)))

    merged = merge_nearby_boxes(
        raw_boxes,
        float(merge_box_iou_thresh),
        int(merge_box_gap),
        float(merge_axis_overlap),
    )
    filtered = remove_child_boxes(merged)
    boxes = [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2, _ in filtered]
    return boxes, motion_mask
