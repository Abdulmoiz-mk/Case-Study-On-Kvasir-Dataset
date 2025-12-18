import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

import dask.bag as db

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns


CSV_PATH = "kvasir_pairs.csv"
PARQUET_PATH = "kvasir_meta.parquet"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 6
LR = 1e-4
DEVICE = "cpu"
NUM_WORKERS = 0

NUM_DASK_PARTITIONS = 8

FEATURE_GROUP_ORDER = [
    "JCD",
    "Tamura",
    "ColorLayout",
    "EdgeHistogram",
    "AutoColorCorrelogram",
    "PHOG",
]

GROUP_DIMS = {
    "JCD": 168,
    "Tamura": 18,
    "ColorLayout": 33,
    "EdgeHistogram": 80,
    "AutoColorCorrelogram": 256,
    "PHOG": 630,
}

META_DIM = sum(GROUP_DIMS.values())


def parse_feature_file_to_groups(path: str) -> Dict[str, np.ndarray]:
    groups = {}
    with open(path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            name, values = line.strip().split(":", 1)
            groups[name.strip()] = np.fromstring(values, sep=",", dtype=np.float32)
    return groups


def groups_to_flat(groups: Dict[str, np.ndarray]) -> np.ndarray:
    vecs = []
    for g in FEATURE_GROUP_ORDER:
        vecs.append(groups.get(g, np.zeros(GROUP_DIMS[g], dtype=np.float32)))
    return np.concatenate(vecs, axis=0)


def build_parquet():
    print("Building parquet using Dask...")
    df = pd.read_csv(CSV_PATH)
    records = df.to_dict("records")

    def process(rec):
        try:
            g = parse_feature_file_to_groups(rec["feature_path"])
            meta = groups_to_flat(g).tolist()
        except:
            meta = np.zeros(META_DIM).tolist()

        return {
            "image_path": rec["image_path"],
            "feature_path": rec["feature_path"],
            "class": rec["class"],
            "meta": meta,
        }

    bag = db.from_sequence(records, npartitions=NUM_DASK_PARTITIONS)
    parsed = bag.map(process).compute()

    df_parsed = pd.DataFrame(parsed)
    le = LabelEncoder()
    df_parsed["label_id"] = le.fit_transform(df_parsed["class"])

    df_parsed.to_parquet(PARQUET_PATH, index=False)
    print("Saved:", PARQUET_PATH)
    return df_parsed, le


class KvasirHybridDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        meta = torch.tensor(row["meta"], dtype=torch.float32)
        label = int(row["label_id"])
        return img, meta, label


class HybridResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        cnn_dim = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

        self.meta_bn = nn.BatchNorm1d(META_DIM)
        self.fc = nn.Sequential(
            nn.Linear(cnn_dim + META_DIM, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x_img, x_meta):
        img_feat = self.cnn(x_img)
        meta_feat = self.meta_bn(x_meta)
        x = torch.cat([img_feat, meta_feat], dim=1)
        return self.fc(x)


if __name__ == "__main__":

    if Path(PARQUET_PATH).exists():
        df_meta = pd.read_parquet(PARQUET_PATH)
    else:
        df_meta, _ = build_parquet()

    train_df, val_df = train_test_split(
        df_meta,
        test_size=0.2,
        stratify=df_meta["label_id"],
        random_state=42
    )

    train_tfms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    train_ds = KvasirHybridDataset(train_df, train_tfms)
    val_ds = KvasirHybridDataset(val_df, val_tfms)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE)

    model = HybridResNet18(num_classes=df_meta["label_id"].nunique()).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_acc_hist, val_acc_hist = [], []

    for epoch in range(NUM_EPOCHS):
        model.train()
        correct, total = 0, 0

        for i, (imgs, metas, labels) in enumerate(train_loader):
            imgs, metas, labels = imgs.to(DEVICE), metas.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs, metas)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            if i % 20 == 0:
                print(f"Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss {loss.item():.4f}")

        train_acc = correct / total
        train_acc_hist.append(train_acc)

        model.eval()
        correct, total = 0, 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for i, (imgs, metas, labels) in enumerate(train_loader):
                imgs, metas = imgs.to(DEVICE), metas.to(DEVICE)
                outputs = model(imgs, metas)
                preds = outputs.argmax(1)

                y_true.extend(labels.numpy())
                y_pred.extend(preds.cpu().numpy())

                correct += (preds.cpu() == labels).sum().item()
                total += labels.size(0)
                if i % 20 == 0:
                    print(f"Epoch {epoch+1} | Batch {i}/{len(train_loader)} | Loss {loss.item():.4f}")

        val_acc = correct / total
        val_acc_hist.append(val_acc)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")


    print("\nF1 Score:", f1_score(y_true, y_pred, average="weighted"))
    print("Recall:", recall_score(y_true, y_pred, average="weighted"))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    plt.figure(figsize=(7,4))
    plt.plot(train_acc_hist, label="Train Acc")
    plt.plot(val_acc_hist, label="Val Acc")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

    torch.save(model.state_dict(), "kvasir_hybrid_final.pth")
    print("Model saved.")
