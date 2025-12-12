# hybrid_kvasir_train_with_dask.py
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

# Dask imports
import dask.bag as db
import dask.dataframe as dd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# ================== CONFIG ==================

CSV_PATH = "kvasir_pairs.csv"     # pairs CSV: class, image_path, feature_path
PARQUET_PATH = "kvasir_meta.parquet"
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 6
LR = 1e-4
DEVICE = "cpu"
NUM_WORKERS = 0

# Dask partitions (set to CPU cores; 0/1 -> single partition)
NUM_DASK_PARTITIONS = 8

# feature groups in consistent order
FEATURE_GROUP_ORDER = [
    "JCD",
    "Tamura",
    "ColorLayout",
    "EdgeHistogram",
    "AutoColorCorrelogram",
    "PHOG",
]

# Known dims from your inspection (used to pad if missing)
GROUP_DIMS = {
    "JCD": 168,
    "Tamura": 18,
    "ColorLayout": 33,
    "EdgeHistogram": 80,
    "AutoColorCorrelogram": 256,
    "PHOG": 630,
}

# ================== FEATURE PARSING (same as before) ==================

def parse_feature_file_to_groups(path: str) -> Dict[str, np.ndarray]:
    """
    Read a .features file and return dict {group_name: ndarray}.
    """
    path = Path(path)
    groups = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            name, values_str = line.split(":", 1)
            name = name.strip()
            vals = np.fromstring(values_str, sep=",", dtype=np.float32)
            groups[name] = vals
    return groups


def groups_to_flat(groups: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Concatenate groups in FEATURE_GROUP_ORDER and pad missing groups by zeros.
    """
    vecs = []
    for g in FEATURE_GROUP_ORDER:
        if g in groups:
            vecs.append(groups[g])
        else:
            vecs.append(np.zeros(GROUP_DIMS[g], dtype=np.float32))
    return np.concatenate(vecs, axis=0)


# ================== DASK PREPROCESSING ==================

def build_parquet_from_features(csv_path=CSV_PATH, parquet_path=PARQUET_PATH, npartitions=NUM_DASK_PARTITIONS):
    """
    Parse all .features files using dask.bag and save a parquet with columns:
      image_path (str), feature_path (str), class (str), label_id (int), meta (list of floats)
    """
    print("Building parquet with Dask... this may take a few minutes (parallel parsing).")
    # read small csv with pandas (it's small; only paths & classes)
    pairs_df = pd.read_csv(csv_path)
    records = pairs_df.to_dict("records")

    # function to parse one record (runs in worker processes)
    def parse_record(rec):
        feat_path = rec["feature_path"]
        try:
            groups = parse_feature_file_to_groups(feat_path)
            flat = groups_to_flat(groups)
            # convert to python list for safe serialization in dask/dataframe
            rec_out = {
                "image_path": rec["image_path"],
                "feature_path": rec["feature_path"],
                "class": rec["class"],
                "meta": flat.tolist()
            }
        except Exception as e:
            # if parsing fails, fill zeros (but print for diagnosis)
            print(f"[WARN] failed parse {feat_path}: {e}")
            rec_out = {
                "image_path": rec["image_path"],
                "feature_path": rec["feature_path"],
                "class": rec["class"],
                "meta": np.zeros(sum(GROUP_DIMS.values()), dtype=np.float32).tolist()
            }
        return rec_out

    bag = db.from_sequence(records, npartitions=npartitions)
    parsed = bag.map(parse_record)

    # convert to dask dataframe - meta describes column dtypes
    meta = {
        "image_path": str,
        "feature_path": str,
        "class": str,
        "meta": object,
    }
    ddf = parsed.to_dataframe(meta=meta)

    # compute to pandas (dataset is ~8k x 1185 -> fits memory)
    df_parsed = ddf.compute()

    # Optionally: create label_id column here (useful)
    label_enc = LabelEncoder()
    df_parsed["label_id"] = label_enc.fit_transform(df_parsed["class"])

    # save parquet (use pyarrow/fastparquet)
    df_parsed.to_parquet(parquet_path, index=False)
    print(f"Saved parquet to: {parquet_path}")
    return df_parsed, label_enc


# ================== DATASET (uses precomputed meta if present) ==================

class KvasirHybridDataset(Dataset):
    def __init__(self, df_subset: pd.DataFrame, meta_dim: int, transform=None, use_precomputed_meta=True):
        self.df = df_subset.reset_index(drop=True)
        self.meta_dim = meta_dim
        self.transform = transform
        self.use_precomputed_meta = use_precomputed_meta

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # image
        img_path = row["image_path"]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # metadata:
        if self.use_precomputed_meta and "meta" in row:
            feat_list = row["meta"]
            feat_vec = np.array(feat_list, dtype=np.float32)
        else:
            feat_vec = groups_to_flat(parse_feature_file_to_groups(row["feature_path"]))

        # enforce dimension
        if len(feat_vec) != self.meta_dim:
            if len(feat_vec) < self.meta_dim:
                pad = np.zeros(self.meta_dim - len(feat_vec), dtype=np.float32)
                feat_vec = np.concatenate([feat_vec, pad])
            else:
                feat_vec = feat_vec[:self.meta_dim]

        meta = torch.from_numpy(feat_vec).float()

        label = int(row["label_id"])
        return img, meta, label


# ================== MODEL (same as before) ==================

class HybridResNet18(nn.Module):
    def __init__(self, num_classes: int, meta_dim: int):
        super().__init__()
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
        cnn_feats = self.cnn(x_img)
        x_meta = self.meta_bn(x_meta)
        x = torch.cat([cnn_feats, x_meta], 1)
        logits = self.fc(x)
        return logits


def batch_accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)


# ================== MAIN ==================

if __name__ == "__main__":
    # 1) If parquet exists, load it; else build using dask
    if Path(PARQUET_PATH).exists():
        print("Loading precomputed parquet:", PARQUET_PATH)
        df_meta = pd.read_parquet(PARQUET_PATH)
        # ensure label_id exists (in case parquet was created earlier w/o it)
        if "label_id" not in df_meta.columns:
            le = LabelEncoder()
            df_meta["label_id"] = le.fit_transform(df_meta["class"])
    else:
        df_meta, label_enc = build_parquet_from_features(CSV_PATH, PARQUET_PATH, npartitions=NUM_DASK_PARTITIONS)

    # now df_meta has columns: image_path, feature_path, class, meta (list), label_id
    print("Parsed meta shape:", df_meta.shape)

    # determine meta dim
    sample_meta = np.array(df_meta["meta"].iloc[0], dtype=np.float32)
    META_DIM = len(sample_meta)
    print("Metadata feature dimension:", META_DIM)

    # training split (we can use pandas since the data is reasonably sized)
    train_df, val_df = train_test_split(df_meta, test_size=0.2, stratify=df_meta["label_id"], random_state=42)
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

    # transforms
    train_tfms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # datasets & dataloaders (use_precomputed_meta=True avoids re-parsing files)
    train_ds = KvasirHybridDataset(train_df, meta_dim=META_DIM, transform=train_tfms, use_precomputed_meta=True)
    val_ds = KvasirHybridDataset(val_df, meta_dim=META_DIM, transform=val_tfms, use_precomputed_meta=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)

    num_classes = train_df["label_id"].nunique()
    print("Num classes:", num_classes)

    # model, loss, optimizer
    model = HybridResNet18(num_classes=num_classes, meta_dim=META_DIM).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # training loop (same as before)
    best_val = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
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
                print(f"  [Train] Batch {batch_idx}/{len(train_loader)} - running loss: {train_loss/total:.4f}")

        train_loss /= total
        train_acc /= total

        # validation
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

        print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # save best model automatically
        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": list(sorted(df_meta["class"].unique(), key=lambda x: int(np.where(train_df['class'].unique()==x)[0][0]) if False else 0)),
                "meta_dim": META_DIM,
            }, "kvasir_hybrid_resnet18_best.pth")
            print(f"  [INFO] Saved best model (val_acc={best_val:.4f}) -> kvasir_hybrid_resnet18_best.pth")

    # final save
    torch.save({
        "model_state": model.state_dict(),
        "classes": list(sorted(df_meta["class"].unique())),
        "meta_dim": META_DIM,
    }, "kvasir_hybrid_resnet18_final.pth")
    print("\nTraining complete. Final model saved to kvasir_hybrid_resnet18_final.pth")
