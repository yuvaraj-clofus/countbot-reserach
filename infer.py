import torch
import cv2
import numpy as np

# -------- CONFIG --------
model_name = "dinov2_vits14"
ref_feature_path = "/home/vikbot/Documents/countbot/1_dinov2_6_patch_strip.pt"
image_path = "test.png"

# -------- LOAD MODEL --------
model = torch.hub.load('facebookresearch/dinov2', model_name)
model.eval()

# -------- LOAD REFERENCE FEATURE --------
ref_feature = torch.load(ref_feature_path)  # shape: (1, D)

# -------- IMAGE PREPROCESS --------
def preprocess(img):
    img = cv2.resize(img, (224, 224))  # DINO expects 224
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0

    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img).float().unsqueeze(0)
    return img

# -------- LOAD IMAGE --------
img = cv2.imread(image_path)
img_tensor = preprocess(img)

# -------- EXTRACT FEATURE --------
with torch.no_grad():
    feature = model(img_tensor)

# -------- COSINE SIMILARITY --------
similarity = torch.nn.functional.cosine_similarity(feature, ref_feature)

print("Similarity:", similarity.item())