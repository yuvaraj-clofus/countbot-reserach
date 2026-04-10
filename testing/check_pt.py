import io
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


def _make_dino_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _ensure_dino_model():
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()
    return model


def image_to_embedding(image_path, model=None, transform=None, device='cpu'):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"{image_path} not found")

    if model is None:
        model = _ensure_dino_model()
    if transform is None:
        transform = _make_dino_transform()

    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(tensor)
    emb = torch.nn.functional.normalize(emb, dim=-1).squeeze(0)
    return emb.cpu()


def check_pt_file(pt_path, image_path=None, threshold=0.6):
    pt_path = Path(pt_path)
    if not pt_path.exists():
        raise FileNotFoundError(f"{pt_path} not found")

    data = torch.load(pt_path)
    if not data:
        raise ValueError("No embeddings in pt file")

    # Roundtrip check
    buf = io.BytesIO()
    torch.save(data, buf)
    buf.seek(0)
    data2 = torch.load(buf)

    if set(data.keys()) != set(data2.keys()):
        raise ValueError("PT roundtrip key mismatch")

    for key in data:
        if not torch.allclose(data[key], data2[key], atol=1e-5, rtol=1e-3):
            raise ValueError(f"Embedding {key} mismatch after roundtrip")

    print(f"PT check OK: {pt_path.name}, samples: {len(data)}")

    if image_path:
        model = _ensure_dino_model()
        transform = _make_dino_transform()
        image_emb = image_to_embedding(image_path, model=model, transform=transform)

        best_key = None
        best_score = -1.0

        for key, emb in data.items():
            emb_norm = torch.nn.functional.normalize(emb, dim=-1)
            sim = torch.nn.functional.cosine_similarity(image_emb, emb_norm, dim=0).item()
            if sim > best_score:
                best_score = sim
                best_key = key

        print(f"Best match: {best_key} (cosine={best_score:.4f})")
        if best_score >= threshold:
            print(f"IMAGE MATCHED pt sample ({best_score:.4f} >= {threshold})")
        else:
            print(f"IMAGE DID NOT MATCH (best {best_score:.4f} < {threshold})")


if __name__ == "__main__":
    import sys
    pt = sys.argv[1] if len(sys.argv) > 1 else "slave/pt_file/BCWIN20BL00009_suspension.pt"
    img = sys.argv[2] if len(sys.argv) > 2 else None
    check_pt_file(pt, image_path=img)
