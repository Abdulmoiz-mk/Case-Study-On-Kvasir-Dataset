import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# ================== CONFIG ==================

CSV_PATH = "kvasir_pairs.csv"     # your pairs CSV: class, image_path, feature_path
IMAGE_SIZE = 224
BATCH_SIZE = 32       # smaller for CPU
NUM_EPOCHS = 6       # start small to test; increase later
LR = 1e-4
DEVICE = "cpu"        # force CPU
NUM_WORKERS = 0       # Windows-safe (no multiprocessing)

# feature groups in consistent order
FEATURE_GROUP_ORDER = [
    "JCD",
    "Tamura",
    "ColorLayout",
    "EdgeHistogram",
    "AutoColorCorrelogram",
    "PHOG",
]

# ================== PARSE .FEATURE FILES ==================

def parse_feature_file(path: str) -> np.ndarray:
    """
    Parse one .features file into a flat np.array.
    Each line is: Name: v1,v2,v3,...
    We concatenate feature groups in FEATURE_GROUP_ORDER.
    """
    path = Path(path)
    groups = {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ":" not in line:
                continue

            name, values_str = line.split(":", 1)
            name = name.strip()
            # convert "1.0,2.0,3.5,..." to array
            vals = np.fromstring(values_str, sep=",", dtype=np.float32)
            groups[name] = vals

    vecs = []
    for g in FEATURE_GROUP_ORDER:
        if g in groups:
            vecs.append(groups[g])

    if not vecs:
        return np.array([], dtype=np.float32)

    return np.concatenate(vecs, axis=0)


# ================== DATASET & MODEL DEFINITIONS ==================

class KvasirHybridDataset(Dataset):
    def __init__(self, df_subset: pd.DataFrame, meta_dim: int, transform=None):
        self.df = df_subset.reset_index(drop=True)
        self.meta_dim = meta_dim
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ---- image ----
        img_path = row["image_path"]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # ---- metadata ----
        feat_vec = parse_feature_file(row["feature_path"])
        # enforce fixed dim
        if len(feat_vec) != self.meta_dim:
            if len(feat_vec) < self.meta_dim:
                pad = np.zeros(self.meta_dim - len(feat_vec), dtype=np.float32)
                feat_vec = np.concatenate([feat_vec, pad])
            else:
                feat_vec = feat_vec[:self.meta_dim]
        meta = torch.from_numpy(feat_vec).float()

        # ---- label ----
        label = int(row["label_id"])

        return img, meta, label


class HybridResNet18(nn.Module):
    def __init__(self, num_classes: int, meta_dim: int):
        super().__init__()

        # load pretrained resnet18 (compatible with different torchvision versions)
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.cnn = models.resnet18(weights=weights)
        except Exception:
            self.cnn = models.resnet18(pretrained=True)

        cnn_out_dim = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()

        self.meta_bn = nn.BatchNorm1d(meta_dim)

        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim + meta_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x_img, x_meta):
        cnn_feats = self.cnn(x_img)            # [B, cnn_out_dim]
        x_meta = self.meta_bn(x_meta)          # [B, meta_dim]
        x = torch.cat([cnn_feats, x_meta], 1)  # [B, cnn_out_dim + meta_dim]
        logits = self.fc(x)
        return logits


def batch_accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)


# ================== MAIN ==================

if __name__ == "__main__":
    # ----- 1. LOAD CSV -----
    df = pd.read_csv(CSV_PATH)

    assert {"class", "image_path", "feature_path"}.issubset(df.columns), \
        "CSV must contain 'class', 'image_path', 'feature_path' columns."

    label_enc = LabelEncoder()
    df["label_id"] = label_enc.fit_transform(df["class"])
    num_classes = df["label_id"].nunique()

    print("Classes:", label_enc.classes_)
    print("Num samples:", len(df))

    # ----- 2. DETERMINE META DIM -----
    sample_feat_vec = parse_feature_file(df["feature_path"].iloc[0])
    META_DIM = len(sample_feat_vec)
    print("Metadata feature dimension:", META_DIM)

    # ----- 3. TRAIN / VAL SPLIT -----
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label_id"],
    )

    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

    # ----- 4. TRANSFORMS -----
    train_tfms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # ----- 5. DATASETS & DATALOADERS -----
    train_ds = KvasirHybridDataset(train_df, meta_dim=META_DIM, transform=train_tfms)
    val_ds   = KvasirHybridDataset(val_df,   meta_dim=META_DIM, transform=val_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    # ----- 6. MODEL, LOSS, OPTIMIZER -----
    model = HybridResNet18(num_classes=num_classes, meta_dim=META_DIM).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # ----- 7. TRAINING LOOP -----
    for epoch in range(1, NUM_EPOCHS + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        total = 0

        print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")

        for batch_idx, (imgs, metas, labels) in enumerate(train_loader, start=1):
            imgs = imgs.to(DEVICE)
            metas = metas.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs, metas)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            train_loss += loss.item() * bs
            train_acc += batch_accuracy(outputs, labels) * bs
            total += bs

            if batch_idx % 20 == 0 or batch_idx == 1:
                print(
                    f"  [Train] Batch {batch_idx}/{len(train_loader)} "
                    f"- running loss: {train_loss/total:.4f}"
                )

        train_loss /= total
        train_acc /= total

        # ---- validation ----
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        total_val = 0

        with torch.no_grad():
            for imgs, metas, labels in val_loader:
                imgs = imgs.to(DEVICE)
                metas = metas.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(imgs, metas)
                loss = criterion(outputs, labels)

                bs = labels.size(0)
                val_loss += loss.item() * bs
                val_acc += batch_accuracy(outputs, labels) * bs
                total_val += bs

        val_loss /= total_val
        val_acc /= total_val

        print(
            f"Epoch {epoch}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    # ----- 8. SAVE MODEL -----
    ckpt_path = "kvasir_hybrid_resnet18.pth"
    torch.save(
        {
            "model_state": model.state_dict(),
            "classes": list(label_enc.classes_),
            "meta_dim": META_DIM,
        },
        ckpt_path,
    )
    print(f"\nTraining complete. Model saved to {ckpt_path}")
