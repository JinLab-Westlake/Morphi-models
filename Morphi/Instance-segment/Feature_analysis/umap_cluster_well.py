import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import hdbscan


# =========================
# Parameters
# =========================

input_csv = "/storage/jinxinLab/wangyimin/Cell_seg/Datasets/20240928/20240926_a2780_post_drug__2024-09-26T22_59_12-Measurement/channel/flat_plate/feature_without_none.csv"        # 改成你的输入文件
output_dir = "/storage/jinxinLab/wangyimin/Cell_seg/Datasets/20240928/20240926_a2780_post_drug__2024-09-26T22_59_12-Measurement/channel/flat_plate/umap_well_level"
os.makedirs(output_dir, exist_ok=True)

output_excel = os.path.join(output_dir, "well_level_umap_cluster_mean_std.xlsx")

USE_STD_FOR_CLUSTERING = False      # True = mean + std both for clustering; False = Only use mean


# =========================
# Reading data
# =========================

df = pd.read_csv(input_csv)
df.columns = df.columns.str.strip()

print("Original data:", df.shape)



# from mask_path extract fov_id / well_id
df["fov_id"] = df["mask_path"].astype(str).str.extract(r"(r\d{2}c\d{2}f\d{2})")
df["well_id"] = df["fov_id"].str.extract(r"(r\d{2}c\d{2})")

df = df.dropna(subset=["well_id"]).copy()

print("Data with well_id:", df.shape)



# Feature selection
# Note: Mortality related columns are not involved in UMAP/clustering and are only used for final coloring

exclude_cols = [
    # IDs / metadata
    "label_id",
    "instance_id",
    "object_id",
    "image_name",
    "fov_key",
    "row",
    "col",
    "brightfield_path",
    "mask_path",
    "fov_id",
    "well_id",
    "drugname",
    "plate",
    "folder_name",
    "token",

    # live/dead labels，not attend UMAP / clustering
    "died/all",
    "death_rate",
    "dead_count",
    "all_count",
    "live_pixels",
    "dead_pixels",
    "overlap_pixels",
    "instance_area",
    "components_in_label"
]

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in exclude_cols]

print("Original feature columns:")
print(feature_cols)



# calculate mean and std by well_id
well_feature_summary = (
    df.groupby("well_id")[feature_cols]
    .agg(["mean", "std"])
)

# flatten column names
well_feature_summary.columns = [
    f"{col}_{stat}" for col, stat in well_feature_summary.columns
]

well_feature_summary = well_feature_summary.reset_index()

# std If there is only one object that is NaN, fill in 0 here
std_cols = [c for c in well_feature_summary.columns if c.endswith("_std")]
well_feature_summary[std_cols] = well_feature_summary[std_cols].fillna(0)


# Compute the death rate for each well_id

if {"dead_count", "all_count"}.issubset(df.columns):

    death_summary = (
        df.groupby("well_id", as_index=False)
        .agg(
            dead_count_sum=("dead_count", "sum"),
            all_count_sum=("all_count", "sum"),
            n_objects=("well_id", "count"),
            n_fovs=("fov_id", "nunique")
        )
    )

    death_summary["death_ratio"] = (
        death_summary["dead_count_sum"] / death_summary["all_count_sum"]
    )

    if "died/all" in df.columns:
        death_extra = (
            df.groupby("well_id", as_index=False)
            .agg(
                death_ratio_object_mean=("died/all", "mean"),
                death_ratio_object_std=("died/all", "std")
            )
        )

        death_extra["death_ratio_object_std"] = death_extra["death_ratio_object_std"].fillna(0)

        death_summary = death_summary.merge(
            death_extra,
            on="well_id",
            how="left"
        )

else:

    death_summary = (
        df.groupby("well_id", as_index=False)
        .agg(
            death_ratio=("died/all", "mean"),
            death_ratio_object_std=("died/all", "std"),
            n_objects=("well_id", "count"),
            n_fovs=("fov_id", "nunique")
        )
    )

    death_summary["death_ratio_object_std"] = death_summary["death_ratio_object_std"].fillna(0)



# well-level feature and mortality were combined

well_df = well_feature_summary.merge(
    death_summary,
    on="well_id",
    how="left"
)

print("Well-level summary:", well_df.shape)
print(well_df.head())


# well-level features for UMAP/clustering are selected

mean_feature_cols = [f"{c}_mean" for c in feature_cols]
std_feature_cols = [f"{c}_std" for c in feature_cols]

if USE_STD_FOR_CLUSTERING:
    cluster_feature_cols = mean_feature_cols + std_feature_cols
else:
    cluster_feature_cols = mean_feature_cols

print("Features used for UMAP / clustering:")
print(cluster_feature_cols)


# clear NaN

X_df = well_df[cluster_feature_cols].copy()
X_df = X_df.replace([np.inf, -np.inf], np.nan)

valid_idx = X_df.dropna().index

X_df = X_df.loc[valid_idx].copy()
well_df = well_df.loc[valid_idx].copy()

print("Clean well-level data:", X_df.shape)


# standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)


# UMAP

# 24drugs-14conc
reducer = umap.UMAP(
    n_neighbors=30,
    min_dist=0.15,
    spread=1.0,
    n_components=2,
    metric="euclidean",
    random_state=42
)



embedding = reducer.fit_transform(X_scaled)

well_df["UMAP1"] = embedding[:, 0]
well_df["UMAP2"] = embedding[:, 1]


# 24drugs-14conc
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=6,
    min_samples=4,
    metric="euclidean"
)

well_df["cluster"] = clusterer.fit_predict(well_df[["UMAP1", "UMAP2"]])

n_clusters = len(set(well_df["cluster"]) - {-1})
n_noise = (well_df["cluster"] == -1).sum()

print("Auto detected clusters:", n_clusters)
print("Noise points:", n_noise)

cluster_death_summary = (
    well_df
    .groupby("cluster", as_index=False)
    .agg(
        n_wells=("well_id", "count"),
        death_ratio_mean=("death_ratio", "mean"),
        death_ratio_median=("death_ratio", "median"),
        death_ratio_std=("death_ratio", "std"),
        death_ratio_q25=("death_ratio", lambda x: x.quantile(0.25)),
        death_ratio_q75=("death_ratio", lambda x: x.quantile(0.75)),
    )
)

cluster_death_summary["death_ratio_std"] = cluster_death_summary["death_ratio_std"].fillna(0)
cluster_death_summary = cluster_death_summary.sort_values("death_ratio_mean")

print(cluster_death_summary)




# Saving umap

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    well_df["UMAP1"],
    well_df["UMAP2"],
    c=well_df["death_ratio"],
    s=100,
    alpha=0.85,
    edgecolors="black",
    linewidths=0.5
)

plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("Well-level UMAP colored by death ratio")

cbar = plt.colorbar(scatter)
cbar.set_label("Death ratio")


plt.tight_layout()

death_fig_path = os.path.join(output_dir, "well_level_umap_colored_by_death_ratio.png")
plt.savefig(death_fig_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved figure:", death_fig_path)

plt.figure(figsize=(8, 6))

unique_clusters = sorted(well_df["cluster"].unique())
non_noise_clusters = [c for c in unique_clusters if c != -1]

# Separate colormap
cmap = plt.cm.get_cmap("tab20", max(len(non_noise_clusters), 1))

# noise cluster
for i, cluster_id in enumerate(non_noise_clusters):
    sub_df = well_df[well_df["cluster"] == cluster_id]

    plt.scatter(
        sub_df["UMAP1"],
        sub_df["UMAP2"],
        s=100,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.5,
        color=cmap(i),
        label=f"Cluster {cluster_id}"
    )

# Noise -> black
noise_df = well_df[well_df["cluster"] == -1]

if len(noise_df) > 0:
    plt.scatter(
        noise_df["UMAP1"],
        noise_df["UMAP2"],
        s=100,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.5,
        color="black",
        label="Noise"
    )

plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.title("Well-level UMAP colored by HDBSCAN morphology state")

plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    borderaxespad=0,
    fontsize=9,
    frameon=True
)

plt.tight_layout()

cluster_fig_path = os.path.join(output_dir, "well_level_umap_colored_by_cluster.png")
plt.savefig(cluster_fig_path, dpi=300, bbox_inches="tight")
plt.close()

print("Saved figure:", cluster_fig_path)


# saving excel
features_used_df = pd.DataFrame({
    "feature_used_for_clustering": cluster_feature_cols
})

original_features_df = pd.DataFrame({
    "original_feature": feature_cols
})

summary_info_df = pd.DataFrame({
    "item": [
        "n_wells",
        "n_original_objects",
        "n_clean_wells",
        "n_clusters_detected",
        "n_noise_points",
        "use_std_for_clustering"
    ],
    "value": [
        well_df["well_id"].nunique(),
        df.shape[0],
        well_df.shape[0],
        n_clusters,
        n_noise,
        USE_STD_FOR_CLUSTERING
    ]
})






with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    well_df.to_excel(writer, sheet_name="well_umap_cluster", index=False)
    well_feature_summary.to_excel(writer, sheet_name="mean_std_by_well", index=False)
    death_summary.to_excel(writer, sheet_name="death_summary", index=False)
    features_used_df.to_excel(writer, sheet_name="features_used", index=False)
    original_features_df.to_excel(writer, sheet_name="original_features", index=False)
    summary_info_df.to_excel(writer, sheet_name="summary", index=False)

print("Saved Excel:", output_excel)


