import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Kvasir EDA Dashboard", layout="wide")

PARQUET_PATH = "kvasir_meta.parquet"

FEATURE_GROUPS = {
    "JCD": (0, 168),
    "Tamura": (168, 186),
    "ColorLayout": (186, 219),
    "EdgeHistogram": (219, 299),
    "AutoColorCorrelogram": (299, 555),
    "PHOG": (555, 1185),
}

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_parquet(PARQUET_PATH)

df = load_data()
meta = np.vstack(df["meta"].values)

# ---------------- SIDEBAR ----------------
st.sidebar.title("Controls")
selected_class = st.sidebar.selectbox(
    "Select Class",
    ["All"] + sorted(df["class"].unique().tolist())
)

if selected_class != "All":
    mask = df["class"] == selected_class
    df = df[mask]
    meta = meta[mask.values]

# ---------------- HEADER ----------------
st.title("📊 Kvasir Dataset – EDA Dashboard")

st.markdown("""
Interactive dashboard for **endoscopic image analysis**  
Dataset: **Kvasir-v2**  
Includes **image statistics**, **handcrafted features**, and **dimensionality reduction**
""")

# ---------------- METRICS ----------------
c1, c2, c3 = st.columns(3)
c1.metric("Total Images", len(df))
c2.metric("Classes", df["class"].nunique())
c3.metric("Feature Dimension", meta.shape[1])

# ---------------- CLASS DISTRIBUTION ----------------
st.subheader("Class Distribution")
fig, ax = plt.subplots()
df["class"].value_counts().plot(kind="bar", ax=ax)
ax.set_ylabel("Count")
st.pyplot(fig)

# ---------------- IMAGE STATS ----------------
st.subheader("Image Statistics")

c1, c2 = st.columns(2)

with c1:
    fig, ax = plt.subplots()
    ax.hist(df["width"], bins=20)
    ax.set_title("Image Width")
    st.pyplot(fig)

with c2:
    fig, ax = plt.subplots()
    ax.hist(df["height"], bins=20)
    ax.set_title("Image Height")
    st.pyplot(fig)

fig, ax = plt.subplots()
ax.hist(df["brightness"], bins=30)
ax.set_title("Brightness Distribution")
st.pyplot(fig)

# ---------------- SAMPLE IMAGES ----------------
st.subheader("Sample Images")
cols = st.columns(4)
sample_df = df.sample(min(4, len(df)), random_state=42)

for col, (_, row) in zip(cols, sample_df.iterrows()):
    img = Image.open(row["image_path"])
    col.image(img, caption=row["class"], use_column_width=True)

# ---------------- FEATURE GROUP ANALYSIS ----------------
st.subheader("Feature Group Analysis")

group_means = {}
for g, (s, e) in FEATURE_GROUPS.items():
    group_means[g] = meta[:, s:e].mean(axis=1)

group_df = pd.DataFrame(group_means)
group_df["class"] = df["class"].values

fig, ax = plt.subplots(figsize=(10, 4))
sns.boxplot(data=group_df.drop(columns="class"), ax=ax)
ax.set_title("Feature Group Mean Distributions")
st.pyplot(fig)

# ---------------- PCA ----------------
st.subheader("PCA Projection (Metadata)")

pca = PCA(n_components=2)
meta_2d = pca.fit_transform(meta)

pca_df = pd.DataFrame(meta_2d, columns=["PC1", "PC2"])
pca_df["class"] = df["class"].values

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="class",
    alpha=0.7,
    ax=ax,
)
ax.set_title("PCA of Handcrafted Features")
st.pyplot(fig)

# ---------------- OUTLIERS ----------------
st.subheader("Outlier Images (High Metadata Norm)")

df["meta_norm"] = np.linalg.norm(meta, axis=1)
outliers = df.nlargest(3, "meta_norm")

cols = st.columns(3)
for col, (_, row) in zip(cols, outliers.iterrows()):
    img = Image.open(row["image_path"])
    col.image(img, caption=f"{row['class']}\nNorm={row['meta_norm']:.2f}", use_column_width=True)
