import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

import dask.bag as db
import dask.dataframe as dd


CSV_PATH = "kvasir_pairs.csv"       # Index file
PARQUET_PATH = "kvasir_meta.parquet"  # Cached parsed dataset

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

N_PARTITIONS = max(4, os.cpu_count() or 4)


def parse_feature_file(path: str):
    """Parse .features into group -> np.array."""
    groups = {}
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                name, vals = line.split(":", 1)
                groups[name.strip()] = np.fromstring(vals, sep=",", dtype=np.float32)
    except:
        for g in FEATURE_GROUP_ORDER:
            groups[g] = np.zeros(GROUP_DIMS[g], dtype=np.float32)
    return groups


def groups_to_flat(groups: dict):
    vecs = []
    for g in FEATURE_GROUP_ORDER:
        if g in groups:
            vecs.append(groups[g])
        else:
            vecs.append(np.zeros(GROUP_DIMS[g], dtype=np.float32))
    return np.concatenate(vecs, axis=0)


def compute_img_stats(path: str):
    """Return width, height, brightness."""
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        bright = float(np.array(img).mean())
        return w, h, bright
    except:
        return 0, 0, 0.0


def parse_record(rec):
    """Dask worker: read image stats + metadata for one file."""
    img_path = rec["image_path"]
    feat_path = rec["feature_path"]

    w, h, bright = compute_img_stats(img_path)
    groups = parse_feature_file(feat_path)
    meta_flat = groups_to_flat(groups)

    return {
        "image_path": img_path,
        "feature_path": feat_path,
        "class": rec["class"],
        "width": w,
        "height": h,
        "brightness": bright,
        "meta": meta_flat.tolist(),
    }


def main():
    index_df = pd.read_csv(CSV_PATH)
    print("Loaded index rows:", len(index_df))

    
    if Path(PARQUET_PATH).exists():
        df_meta = pd.read_parquet(PARQUET_PATH)
        print("Loaded cached parquet:", PARQUET_PATH)

        required_cols = {"width", "height", "brightness", "meta"}

        if required_cols.issubset(df_meta.columns):
            print("Parquet already complete → skipping Dask parsing.")
        else:
            print("Parquet missing columns → recomputing missing fields with Dask...")

            recs = [{"image_path": p} for p in df_meta["image_path"]]

            bag = db.from_sequence(recs, npartitions=N_PARTITIONS)
            stats = bag.map(lambda r: {
                "image_path": r["image_path"],
                **dict(zip(
                    ["width", "height", "brightness"],
                    compute_img_stats(r["image_path"])
                ))
            })

            stats_df = stats.to_dataframe().compute()
            df_meta = df_meta.merge(stats_df, on="image_path", how="left")

            df_meta.to_parquet(PARQUET_PATH, index=False)
            print("Updated parquet saved.")

    else:
        print("No parquet found → performing full Dask parsing...")

        recs = index_df.to_dict("records")
        bag = db.from_sequence(recs, npartitions=N_PARTITIONS)
        parsed = bag.map(parse_record)

        meta_schema = {
            "image_path": str,
            "feature_path": str,
            "class": str,
            "width": int,
            "height": int,
            "brightness": float,
            "meta": object,
        }

        ddf = parsed.to_dataframe(meta=meta_schema)
        df_meta = ddf.compute()

        df_meta.to_parquet(PARQUET_PATH, index=False)
        print("Saved new parquet:", PARQUET_PATH)

    print("Parsed dataframe:", df_meta.shape)

    df_meta["label_id"] = pd.factorize(df_meta["class"])[0]

    
    plt.figure(figsize=(8, 4))
    df_meta["class"].value_counts().plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.hist(df_meta["width"], bins=20, alpha=0.7, label="width")
    plt.hist(df_meta["height"], bins=20, alpha=0.7, label="height")
    plt.title("Image Width/Height Distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.hist(df_meta["brightness"], bins=30, alpha=0.8)
    plt.title("Brightness Distribution")
    plt.tight_layout()
    plt.show()

    print("Building metadata matrix...")
    meta = np.vstack(df_meta["meta"].values)
    META_DIM = meta.shape[1]
    print("meta:", meta.shape)

    meta_df = pd.DataFrame(meta, columns=[f"f_{i}" for i in range(META_DIM)])
    meta_df["class"] = df_meta["class"]

    print("Running PCA...")
    pca = PCA(n_components=2).fit_transform(meta)
    pca_df = pd.DataFrame(pca, columns=["PC1", "PC2"])
    pca_df["class"] = df_meta["class"]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="class", alpha=0.7)
    plt.title("PCA of Feature Metadata")
    plt.tight_layout()
    plt.show()


    norms = np.linalg.norm(meta, axis=1)
    df_meta["norm"] = norms
    print(df_meta.nlargest(3, "norm")[["class", "image_path", "norm"]])


if __name__ == "__main__":
    main()
