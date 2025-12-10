import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path

from hybrid_kvasir_train import (
    HybridResNet18,
    parse_feature_file,
    FEATURE_GROUP_ORDER,
    IMAGE_SIZE,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------
def load_model(ckpt_path="kvasir_hybrid_resnet18.pth"):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    classes = ckpt["classes"]
    meta_dim = ckpt["meta_dim"]

    model = HybridResNet18(num_classes=len(classes), meta_dim=meta_dim)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    return model, classes, meta_dim


# -----------------------------
# IMAGE TRANSFORM (same as val)
# -----------------------------
val_tfms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# -----------------------------
# PREDICT SINGLE IMAGE
# -----------------------------
def predict_one(image_path, feature_path, ckpt_path="kvasir_hybrid_resnet18.pth"):
    model, classes, meta_dim = load_model(ckpt_path)

    # Load image
    img = Image.open(image_path).convert("RGB")
    img = val_tfms(img).unsqueeze(0).to(DEVICE)

    # Load metadata (.features)
    feat_vec = parse_feature_file(feature_path)
    if len(feat_vec) != meta_dim:
        if len(feat_vec) < meta_dim:
            pad = np.zeros(meta_dim - len(feat_vec), dtype=np.float32)
            feat_vec = np.concatenate([feat_vec, pad])
        else:
            feat_vec = feat_vec[:meta_dim]

    meta = torch.from_numpy(feat_vec).float().unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        logits = model(img, meta)
        probs = torch.softmax(logits, dim=1)
        pred_idx = logits.argmax(dim=1).item()

    print(f"Predicted Class: {classes[pred_idx]}")
    print(f"Probabilities: {probs.cpu().numpy()}")

    return classes[pred_idx], probs.cpu().numpy()


# -----------------------------
# EVALUATE ON A CSV (FULL SET)
# -----------------------------
def evaluate_dataset(csv_path="kvasir_pairs.csv", ckpt_path="kvasir_hybrid_resnet18.pth"):
    df = pd.read_csv(csv_path)

    model, classes, meta_dim = load_model(ckpt_path)

    correct = 0
    total = 0

    for idx, row in df.iterrows():
        img_path = row["image_path"]
        feat_path = row["feature_path"]
        true_label = row["class"]

        pred, _ = predict_one(img_path, feat_path, ckpt_path)

        if pred == true_label:
            correct += 1
        total += 1

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(df)} images...")

    acc = correct / total
    print(f"\nFinal Accuracy: {acc:.4f}")
    return acc


# -----------------------------
# MAIN for quick testing
# -----------------------------
if __name__ == "__main__":
    # Example single prediction
    img = r"C:\Users\User\Desktop\lums\3rd semester\Data Science Visualize\Project\kvasir-dataset-v2\dyed-lifted-polyps\0a7bdce4-ac0d-44ef-93ee-92dfc8fe0b81.jpg"
    feat = r"C:\Users\User\Desktop\lums\3rd semester\Data Science Visualize\Project\kvasir-dataset-v2-features\kvasir-dataset-v2-features\dyed-lifted-polyps\0a7bdce4-ac0d-44ef-93ee-92dfc8fe0b81.features"

    # Uncomment to test:
    predict_one(img, feat)

    # Uncomment to evaluate full dataset:
    # evaluate_dataset("kvasir_pairs.csv")
    pass
