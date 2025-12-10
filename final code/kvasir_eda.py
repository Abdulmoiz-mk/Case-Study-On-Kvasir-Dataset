import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# ------------- CONFIG -------------

CSV_PATH = "kvasir_pairs.csv"   # pairs file (class, image_path, feature_path)

FEATURE_GROUP_ORDER = [
    "JCD",
    "Tamura",
    "ColorLayout",
    "EdgeHistogram",
    "AutoColorCorrelogram",
    "PHOG",
]

# Known dims from your inspection
GROUP_DIMS = {
    "JCD": 168,
    "Tamura": 18,
    "ColorLayout": 33,
    "EdgeHistogram": 80,
    "AutoColorCorrelogram": 256,
    "PHOG": 630,
}

# ------------- UTILS -------------

def parse_feature_file(path: str) -> dict:
    """
    Return dict {group_name: np.array([...])} for one .features file.
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


def features_to_flat_vector(groups: dict) -> np.ndarray:
    """
    Concatenate feature groups in FEATURE_GROUP_ORDER into one long vector.
    """
    vecs = []
    for g in FEATURE_GROUP_ORDER:
        if g in groups:
            vecs.append(groups[g])
        else:
            # if missing, pad zeros of appropriate length
            dim = GROUP_DIMS[g]
            vecs.append(np.zeros(dim, dtype=np.float32))
    return np.concatenate(vecs, axis=0)


def get_group_slices():
    """
    Return dict {group_name: (start_idx, end_idx)} for flat vector index ranges.
    """
    slices = {}
    start = 0
    for g in FEATURE_GROUP_ORDER:
        dim = GROUP_DIMS[g]
        slices[g] = (start, start + dim)
        start += dim
    return slices


# ------------- MAIN EDA -------------

def main():
    df = pd.read_csv(CSV_PATH)
    print("Loaded pairs:", df.shape)

    # ==============================
    # 1. CLASS DISTRIBUTION
    # ==============================
    class_counts = df["class"].value_counts().sort_index()
    plt.figure(figsize=(8, 4))
    class_counts.plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # ==============================
    # 2. IMAGE-LEVEL EDA
    # ==============================
    widths = []
    heights = []
    brightness = []

    print("Collecting image stats (sizes, brightness)...")
    for p in df["image_path"]:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        widths.append(w)
        heights.append(h)
        # brightness as mean pixel value
        brightness.append(np.array(img).mean())

    img_stats = pd.DataFrame({
        "width": widths,
        "height": heights,
        "brightness": brightness,
        "class": df["class"],
    })

    plt.figure(figsize=(8, 4))
    plt.hist(img_stats["width"], bins=20, alpha=0.7, label="width")
    plt.hist(img_stats["height"], bins=20, alpha=0.7, label="height")
    plt.title("Image Width/Height Distribution")
    plt.xlabel("Pixels")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.hist(img_stats["brightness"], bins=30, alpha=0.8)
    plt.title("Image Brightness Distribution")
    plt.xlabel("Mean Pixel Value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Sample images per class
    print("Showing random samples per class...")
    classes = sorted(df["class"].unique())
    n_show_per_class = 3
    for c in classes:
        subset = df[df["class"] == c].sample(n=min(n_show_per_class, len(df[df["class"] == c])), random_state=42)
        plt.figure(figsize=(3 * n_show_per_class, 3))
        plt.suptitle(f"Samples from class: {c}")
        for i, (_, row) in enumerate(subset.iterrows()):
            img = Image.open(row["image_path"]).convert("RGB")
            plt.subplot(1, n_show_per_class, i + 1)
            plt.imshow(img)
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    # ==============================
    # 3. METADATA MATRIX
    # ==============================
    print("Parsing .features into flat vectors...")
    all_vecs = []
    for p in df["feature_path"]:
        groups = parse_feature_file(p)
        flat = features_to_flat_vector(groups)
        all_vecs.append(flat)

    meta = np.vstack(all_vecs)  # shape [N, 1185]
    META_DIM = meta.shape[1]
    print("Metadata matrix shape:", meta.shape)

    # quick overall stats
    meta_df = pd.DataFrame(meta, columns=[f"f_{i}" for i in range(META_DIM)])
    meta_df["class"] = df["class"]

    # ==============================
    # 4. FEATURE GROUP SUMMARY
    # ==============================
    slices = get_group_slices()

    # per-sample mean of each group
    group_means = {}
    for g, (s, e) in slices.items():
        group_means[g] = meta[:, s:e].mean(axis=1)

    group_means_df = pd.DataFrame(group_means)
    group_means_df["class"] = df["class"]

    # Boxplot across all samples
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=group_means_df[FEATURE_GROUP_ORDER])
    plt.title("Distribution of Mean Value per Feature Group (All Samples)")
    plt.ylabel("Mean Feature Value")
    plt.tight_layout()
    plt.show()

    # Per-class bar plot of average group means
    per_class_group_means = group_means_df.groupby("class")[FEATURE_GROUP_ORDER].mean()
    plt.figure(figsize=(10, 5))
    per_class_group_means.plot(kind="bar")
    plt.title("Average Feature Group Means per Class")
    plt.ylabel("Mean Value")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # ==============================
    # 5. PER-CLASS DETAIL: Example with Tamura
    # ==============================
    tam_start, tam_end = slices["Tamura"]
    tamura_feats = meta[:, tam_start:tam_end]  # [N, 18]
    tamura_df = pd.DataFrame(tamura_feats, columns=[f"Tamura_{i}" for i in range(tam_end - tam_start)])
    tamura_df["class"] = df["class"]

    tamura_class_mean = tamura_df.groupby("class").mean()

    plt.figure(figsize=(12, 6))
    for c in classes:
        plt.plot(
            range(tamura_class_mean.shape[1]),
            tamura_class_mean.loc[c].values,
            marker="o",
            label=c,
        )
    plt.title("Tamura Feature Means per Class")
    plt.xlabel("Tamura Feature Index")
    plt.ylabel("Mean Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ==============================
    # 6. CORRELATION (on subset)
    # ==============================
    # Use first 100 metadata features to keep plot manageable
    corr_subset = meta_df[[f"f_{i}" for i in range(100)]].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_subset, cmap="viridis", center=0)
    plt.title("Correlation Heatmap (First 100 Metadata Features)")
    plt.tight_layout()
    plt.show()

    # ==============================
    # 7. PCA VISUALIZATION
    # ==============================
    print("Running PCA on metadata...")
    pca = PCA(n_components=2, random_state=42)
    meta_2d = pca.fit_transform(meta)

    pca_df = pd.DataFrame(meta_2d, columns=["PC1", "PC2"])
    pca_df["class"] = df["class"]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="class",
        palette="tab10",
        alpha=0.7,
        s=30,
    )
    plt.title("PCA of Handcrafted Metadata Features")
    plt.tight_layout()
    plt.show()

    # ==============================
    # 8. OUTLIER-LIKE SAMPLES (by metadata norm)
    # ==============================
    norms = np.linalg.norm(meta, axis=1)
    df["meta_norm"] = norms

    # show top 3 highest norm images (potential outliers / strong patterns)
    top_k = 3
    outliers = df.sort_values("meta_norm", ascending=False).head(top_k)
    print("Top metadata-norm samples (potential outliers / strong patterns):")
    print(outliers[["class", "image_path", "meta_norm"]])

    for i, (_, row) in enumerate(outliers.iterrows(), start=1):
        img = Image.open(row["image_path"]).convert("RGB")
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Outlier {i} | Class: {row['class']} | Norm: {row['meta_norm']:.2f}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
