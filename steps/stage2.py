#!/usr/bin/env python3
"""Simple stage 2 selector."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REFS = str(SCRIPT_DIR / "master-models" / "2wheel.pt")
DEFAULT_DEVICE = "cuda"
DEFAULT_MODEL_NAME = "dinov2_vits14"
DEFAULT_THRESHOLD = 0.78

REFS_PATH = DEFAULT_REFS
DEVICE = DEFAULT_DEVICE
MODEL_NAME = DEFAULT_MODEL_NAME
THRESHOLD = DEFAULT_THRESHOLD

TORCH = None
F = None
IMAGE = None
TRANSFORMS = None
MODEL = None
PREPROCESS = None
RESOLVED_DEVICE = None
REFERENCE_EMBEDDINGS = None


def dino(stage1_output):
    mask = stage1_output.get("data")
    bbox = stage1_output.get("bbox")
    if mask is None or bbox is None:
        return empty_result("dino", stage1_output)

    score = round(float(np.mean(mask)) / 255.0, 4)
    label = "MATCH" if score >= 0.50 else "NOT_MATCH"
    return build_result("dino", stage1_output, label, score)


def dinov2(stage1_output):
    frame = stage1_output.get("frame")
    bbox = stage1_output.get("bbox")
    mask = stage1_output.get("data")

    if frame is None or bbox is None:
        return empty_result("dinov2", stage1_output)

    ensure_ready()

    roi = crop_roi(frame, bbox)
    if roi is None:
        return empty_result("dinov2", stage1_output)

    roi = apply_mask_to_roi(roi, mask, bbox)
    embedding = get_embedding(roi, MODEL, PREPROCESS, RESOLVED_DEVICE, TORCH, F, IMAGE)
    if embedding is None:
        return empty_result("dinov2", stage1_output)

    score = compare_embedding(embedding, REFERENCE_EMBEDDINGS, F)
    label = "MATCH" if score >= THRESHOLD else "NOT_MATCH"
    return build_result("dinov2", stage1_output, label, score)


handlers = {
    "dino": dino,
    "dinov2": dinov2,
}


def select_method(stage1_output):
    mask = stage1_output.get("data")
    if mask is None:
        return "dino"
    if float(np.mean(mask)) > 100.0:
        return "dinov2"
    return "dino"


def run(stage1_output, method="dinov2"):
    selected = method or select_method(stage1_output)
    if selected == "auto":
        selected = select_method(stage1_output)

    if selected not in handlers:
        raise ValueError(f"unknown stage2 method: {selected}")

    return handlers[selected](stage1_output)


def configure(refs_path=DEFAULT_REFS, device=DEFAULT_DEVICE, model_name=DEFAULT_MODEL_NAME, threshold=DEFAULT_THRESHOLD):
    global REFS_PATH, DEVICE, MODEL_NAME, THRESHOLD
    global MODEL, PREPROCESS, RESOLVED_DEVICE, REFERENCE_EMBEDDINGS

    REFS_PATH = str(refs_path)
    DEVICE = str(device)
    MODEL_NAME = str(model_name)
    THRESHOLD = float(threshold)

    MODEL = None
    PREPROCESS = None
    RESOLVED_DEVICE = None
    REFERENCE_EMBEDDINGS = None


def empty_result(method, stage1_output):
    return {
        "type": "result",
        "method": method,
        "label": "NO_OBJECT",
        "score": 0.0,
        "bbox": stage1_output.get("bbox"),
    }


def build_result(method, stage1_output, label, score):
    return {
        "type": "result",
        "method": method,
        "label": label,
        "score": float(score),
        "bbox": stage1_output.get("bbox"),
    }


def require_stage2_deps():
    missing = []

    try:
        import torch
        import torch.nn.functional as functional
    except ModuleNotFoundError:
        missing.append("torch")
        torch = None
        functional = None

    try:
        from PIL import Image
    except ModuleNotFoundError:
        missing.append("Pillow")
        Image = None

    try:
        from torchvision import transforms
    except Exception:
        transforms = None

    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"stage2 needs these packages: {joined}")

    return torch, functional, Image, transforms


def load_reference_embeddings(pt_path, torch, functional):
    pt_path = Path(pt_path)
    if not pt_path.exists():
        raise FileNotFoundError(f"reference file not found: {pt_path}")

    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=True)
    except TypeError:
        data = torch.load(pt_path, map_location="cpu")

    embeddings = []
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, torch.Tensor):
                embeddings.append(functional.normalize(value.float().cpu(), dim=-1))
    elif isinstance(data, torch.Tensor):
        if data.ndim == 1:
            embeddings.append(functional.normalize(data.float().cpu(), dim=-1))
        else:
            for row in data:
                embeddings.append(functional.normalize(row.float().cpu(), dim=-1))

    if not embeddings:
        raise ValueError(f"no embeddings found inside: {pt_path}")

    return torch.stack(embeddings, dim=0)


def build_transform(torch, transforms):
    if transforms is not None:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def transform(image):
        width, height = image.size
        scale = 256.0 / min(width, height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        image = image.resize((new_width, new_height))
        left = max(0, (new_width - 224) // 2)
        top = max(0, (new_height - 224) // 2)
        image = image.crop((left, top, left + 224, top + 224))

        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
        return (tensor - mean) / std

    return transform


def load_dino(model_name, device, torch, transforms):
    resolved_device = str(device)
    if resolved_device.startswith("cuda") and not torch.cuda.is_available():
        print("Stage2: cuda not available, using cpu")
        resolved_device = "cpu"

    print(f"Stage2: loading {model_name} on {resolved_device}")
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.to(resolved_device)
    model.eval()
    preprocess = build_transform(torch, transforms)
    return model, preprocess, resolved_device


def ensure_ready():
    global TORCH, F, IMAGE, TRANSFORMS
    global MODEL, PREPROCESS, RESOLVED_DEVICE, REFERENCE_EMBEDDINGS

    if MODEL is not None and PREPROCESS is not None and REFERENCE_EMBEDDINGS is not None:
        return

    TORCH, F, IMAGE, TRANSFORMS = require_stage2_deps()
    REFERENCE_EMBEDDINGS = load_reference_embeddings(REFS_PATH, TORCH, F)
    MODEL, PREPROCESS, RESOLVED_DEVICE = load_dino(MODEL_NAME, DEVICE, TORCH, TRANSFORMS)
    REFERENCE_EMBEDDINGS = REFERENCE_EMBEDDINGS.to(RESOLVED_DEVICE)


def crop_roi(frame, bbox):
    if bbox is None:
        return None

    x1, y1, x2, y2 = [int(value) for value in bbox]
    if x2 <= x1 or y2 <= y1:
        return None

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return roi


def apply_mask_to_roi(roi, mask, bbox):
    if roi is None or roi.size == 0 or mask is None or bbox is None:
        return roi

    x1, y1, x2, y2 = [int(value) for value in bbox]
    mask_roi = mask[y1:y2, x1:x2]
    if mask_roi.size == 0:
        return roi

    if mask_roi.shape[:2] != roi.shape[:2]:
        mask_roi = cv2.resize(mask_roi, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)

    masked = roi.copy()
    masked[mask_roi == 0] = 0
    return masked


def get_embedding(roi_bgr, model, preprocess, device, torch, functional, Image):
    if roi_bgr is None or roi_bgr.size == 0:
        return None

    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    tensor = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor)
    return functional.normalize(embedding, dim=-1).squeeze(0)


def compare_embedding(embedding, reference_embeddings, functional):
    embedding = embedding.to(reference_embeddings.device)
    similarity = functional.cosine_similarity(embedding.unsqueeze(0), reference_embeddings, dim=-1)
    return round(float(similarity.max().item()), 4)


def main():
    print("stage2.py is a selector module. Use it from main.py.")


if __name__ == "__main__":
    main()
