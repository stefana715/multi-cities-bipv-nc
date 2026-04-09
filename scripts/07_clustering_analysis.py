#!/usr/bin/env python3
"""
Step 7: K-means + Hierarchical Clustering of 39 cities on D1-D5 dimensions.
Determines optimal k via silhouette score, names clusters, produces PCA/radar/dendrogram.
"""
import warnings
warnings.filterwarnings("ignore")
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "results_nc" / "clustering"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = PROJECT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

DIMS = ["D1_score", "D2_score", "D3_score", "D4_score", "D5_score"]
DIM_LABELS = ["D1 Climate", "D2 Morphology", "D3 Technical", "D4 Economic", "D5 Uncertainty"]

# Cluster name heuristics (applied after fitting)
CLUSTER_NAMES = {
    # will be assigned after profiling
}

PALETTE = ["#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261", "#A8DADC"]


def load_data():
    sm = pd.read_csv(PROJECT_DIR / "results" / "fdsi" / "suitability_matrix.csv")
    # city column varies; normalise
    if "city" not in sm.columns and "City" in sm.columns:
        sm = sm.rename(columns={"City": "city"})
    # name_en
    if "name_en" not in sm.columns:
        sm["name_en"] = sm["city"].str.capitalize()
    # Use Cityname col if exists
    for col in ["City", "city"]:
        if col in sm.columns:
            sm["city_key"] = sm[col].str.lower()
            break
    return sm


def pick_name(profile: pd.Series) -> str:
    """Heuristic cluster namer based on D1-D5 profile (0-1 scale)."""
    d1, d2, d3, d4, d5 = (profile[f"D{i}_score"] for i in range(1, 6))
    if d1 >= 0.70 and d4 >= 0.60:
        return "高辐照-经济型"
    if d1 >= 0.55 and d2 >= 0.45 and d4 >= 0.45:
        return "沿海均衡型"
    if d1 <= 0.45 and d5 >= 0.60:
        return "低辐照-高确定型"
    if d1 <= 0.40 and d4 <= 0.45:
        return "盆地低适宜型"
    if d2 >= 0.45 and d3 >= 0.55:
        return "形态-技术友好型"
    return "中等综合型"


def main():
    log.info("=" * 60)
    log.info("Step 7: Clustering Analysis")
    log.info("=" * 60)

    df = load_data()
    log.info(f"  Loaded {len(df)} cities, cols: {list(df.columns[:8])}...")

    X = df[DIMS].values
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    city_labels = df["city"].tolist() if "city" in df.columns else df.index.tolist()
    # Try to get English names
    if "name_en" in df.columns:
        name_labels = df["name_en"].tolist()
    else:
        name_labels = city_labels

    # ── 1. Silhouette scores ──────────────────────────────────────────────────
    log.info("\n[1] Silhouette score sweep k=2..6")
    sil_rows = []
    best_k, best_sil = 3, -1
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(Xz)
        sil = silhouette_score(Xz, labels)
        sil_rows.append({"k": k, "silhouette": round(sil, 4)})
        log.info(f"  k={k}: silhouette={sil:.4f}")
        if sil > best_sil:
            best_sil, best_k = sil, k

    log.info(f"  → Best k = {best_k} (silhouette={best_sil:.4f})")
    sil_df = pd.DataFrame(sil_rows)
    sil_df.to_csv(OUT_DIR / "silhouette_scores.csv", index=False)

    # ── 2. K-means with best k ────────────────────────────────────────────────
    log.info(f"\n[2] K-means (k={best_k})")
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    km_labels = km_final.fit_predict(Xz)

    # ── 3. Hierarchical clustering (Ward) ────────────────────────────────────
    log.info(f"\n[3] Hierarchical clustering (Ward, k={best_k})")
    hier = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    hier_labels = hier.fit_predict(Xz)

    # Agreement between KM and HC
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(km_labels, hier_labels)
    log.info(f"  Adjusted Rand Index (KM vs HC): {ari:.3f}")

    # ── 4. Profile each cluster and name ─────────────────────────────────────
    df["km_cluster"] = km_labels
    df["hier_cluster"] = hier_labels

    # Use kmeans labels as primary (more reproducible)
    profiles = df.groupby("km_cluster")[DIMS].mean()
    cluster_names = {}
    for cid, row in profiles.iterrows():
        cluster_names[cid] = pick_name(row)
    df["cluster_name"] = df["km_cluster"].map(cluster_names)

    log.info("\n  Cluster profiles:")
    for cid in sorted(profiles.index):
        name = cluster_names[cid]
        cities_in = df[df["km_cluster"] == cid]["city"].tolist()
        log.info(f"  Cluster {cid} [{name}]: {cities_in}")
        for dim in DIMS:
            log.info(f"    {dim}: {profiles.loc[cid, dim]:.3f}")

    # Save assignments
    assign_cols = ["city"] + (["name_en"] if "name_en" in df.columns else []) + \
                  DIMS + ["fdsi_score", "km_cluster", "hier_cluster", "cluster_name"]
    assign_cols = [c for c in assign_cols if c in df.columns]
    df[assign_cols].to_csv(OUT_DIR / "cluster_assignments.csv", index=False)

    # Save profiles
    prof_out = profiles.copy()
    prof_out["cluster_name"] = pd.Series(cluster_names)
    prof_out["n_cities"] = df.groupby("km_cluster").size()
    prof_out["fdsi_mean"] = df.groupby("km_cluster")["fdsi_score"].mean()
    prof_out.to_csv(OUT_DIR / "cluster_profiles.csv")
    log.info(f"  Saved: cluster_assignments.csv, cluster_profiles.csv, silhouette_scores.csv")

    # ── 5. Silhouette bar chart ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(sil_df["k"], sil_df["silhouette"], color=PALETTE[:len(sil_df)], width=0.6)
    ax.axvline(best_k, color="red", linestyle="--", linewidth=1.5, label=f"Best k={best_k}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Optimal Cluster Count — Silhouette Method")
    ax.legend(fontsize=9)
    for bar, val in zip(bars, sil_df["silhouette"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "silhouette_barchart.png", dpi=150)
    plt.close(fig)

    # ── 6. PCA scatter ───────────────────────────────────────────────────────
    log.info("\n[4] PCA scatter plot")
    pca = PCA(n_components=2, random_state=42)
    Xpca = pca.fit_transform(Xz)
    var_exp = pca.explained_variance_ratio_
    log.info(f"  PCA: PC1={var_exp[0]:.1%}, PC2={var_exp[1]:.1%} ({sum(var_exp):.1%} total)")

    fig, ax = plt.subplots(figsize=(9, 7))
    for cid in sorted(df["km_cluster"].unique()):
        mask = df["km_cluster"] == cid
        idx = df.index[mask]
        ax.scatter(Xpca[mask, 0], Xpca[mask, 1],
                   color=PALETTE[cid % len(PALETTE)], s=80,
                   label=f"C{cid}: {cluster_names[cid]}", zorder=3, alpha=0.85)
        for i in idx:
            ax.annotate(name_labels[i], (Xpca[i, 0], Xpca[i, 1]),
                        fontsize=6.5, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel(f"PC1 ({var_exp[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1%} variance)")
    ax.set_title(f"PCA Projection — {best_k}-Cluster K-means")
    ax.legend(loc="best", fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pca_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # PCA loadings annotation
    log.info("  PCA loadings:")
    for i, dim in enumerate(DIM_LABELS):
        log.info(f"    {dim}: PC1={pca.components_[0,i]:+.3f}, PC2={pca.components_[1,i]:+.3f}")

    # ── 7. Dendrogram ────────────────────────────────────────────────────────
    log.info("\n[5] Dendrogram")
    Z = linkage(Xz, method="ward")
    fig, ax = plt.subplots(figsize=(14, 5))
    dendrogram(Z, labels=name_labels, ax=ax, leaf_rotation=45, leaf_font_size=8,
               color_threshold=Z[-best_k+1, 2])
    ax.set_title(f"Hierarchical Clustering Dendrogram (Ward Linkage) — {best_k} clusters")
    ax.set_ylabel("Ward distance")
    ax.axhline(y=Z[-best_k+1, 2], color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dendrogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── 8. Radar by cluster ───────────────────────────────────────────────────
    log.info("\n[6] Radar chart by cluster")
    angles = np.linspace(0, 2 * np.pi, len(DIMS), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, best_k, figsize=(4 * best_k, 4.5),
                             subplot_kw=dict(polar=True))
    if best_k == 1:
        axes = [axes]

    for ax, cid in zip(axes, sorted(profiles.index)):
        vals = profiles.loc[cid, DIMS].tolist()
        vals += vals[:1]
        color = PALETTE[cid % len(PALETTE)]
        ax.plot(angles, vals, color=color, linewidth=2)
        ax.fill(angles, vals, alpha=0.25, color=color)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(["D1", "D2", "D3", "D4", "D5"], fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=6)
        n_c = (df["km_cluster"] == cid).sum()
        ax.set_title(f"C{cid}: {cluster_names[cid]}\n(n={n_c})", fontsize=9, pad=12)

    fig.suptitle("Cluster D1–D5 Radar Profiles", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "radar_by_cluster.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info(f"\n✓ All clustering outputs → {OUT_DIR}")
    return best_k, cluster_names, df, ari


if __name__ == "__main__":
    main()
