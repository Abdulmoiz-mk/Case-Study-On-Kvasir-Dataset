#Feature Pairing code

# from pathlib import Path
# import pandas as pd

# IMG_ROOT = Path(r"C:\Users\User\Desktop\lums\3rd semester\Data Science Visualize\Project\kvasir-dataset-v2")
# FEAT_ROOT = Path(r"C:\Users\User\Desktop\lums\3rd semester\Data Science Visualize\Project\kvasir-dataset-v2-features\kvasir-dataset-v2-features")

# rows = []

# # Loop through each class folder inside the image folder
# for class_dir in sorted(d for d in IMG_ROOT.iterdir() if d.is_dir()):
#     class_name = class_dir.name
#     feat_class_dir = FEAT_ROOT / class_name

#     for img_path in class_dir.glob("*.jpg"):
#         stem = img_path.stem
#         feat_path = feat_class_dir / f"{stem}.features"

#         if feat_path.exists():
#             rows.append({
#                 "class": class_name,
#                 "image_path": str(img_path),
#                 "feature_path": str(feat_path),
#             })
#         else:
#             print(f"[WARN] Missing feature: {feat_path}")

# df_pairs = pd.DataFrame(rows)
# print("Total pairs:", len(df_pairs))
# df_pairs.to_csv("kvasir_pairs.csv", index=False)


#Feature detail extraction code from features file

from pathlib import Path
import numpy as np

FEATURE_GROUP_ORDER = [
    "JCD",
    "Tamura",
    "ColorLayout",
    "EdgeHistogram",
    "AutoColorCorrelogram",
    "PHOG",
]

def read_feature_file(path, as_flat_vector=False):
    """
    Read a .features file and return either:
      - dict {feature_group_name: np.array([...])}
      - OR a single concatenated 1D vector (if as_flat_vector=True)
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

            vals = np.fromstring(values_str, sep=",", dtype=np.float32)
            groups[name] = vals

    if not as_flat_vector:
        return groups

    vecs = []
    for g in FEATURE_GROUP_ORDER:
        if g in groups:
            vecs.append(groups[g])
        else:
            pass

    if not vecs:
        return np.array([], dtype=np.float32)

    return np.concatenate(vecs, axis=0)


if __name__ == "__main__":
    feature_path = r"C:\Users\User\Desktop\lums\3rd semester\Data Science Visualize\Project\kvasir-dataset-v2-features\kvasir-dataset-v2-features\normal-z-line\0b55fa7f-8271-4c50-8df4-6d228336a21b.features"

    groups = read_feature_file(feature_path, as_flat_vector=False)
    print("=== GROUPS FOUND ===")
    for name, arr in groups.items():
        print(f"{name}: shape={arr.shape}, first 5={arr[:5]}")

    flat_vec = read_feature_file(feature_path, as_flat_vector=True)
    print("\n=== FLATTENED VECTOR ===")
    print("Total dim:", flat_vec.shape[0])
    print("First 10 values:", flat_vec[:10])
