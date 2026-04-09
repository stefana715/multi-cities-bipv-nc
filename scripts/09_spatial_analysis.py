#!/usr/bin/env python3
"""
Step 9: Spatial autocorrelation — Global + Local Moran's I for FDSI and D1-D5.
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
import yaml

import libpysal
from libpysal.weights import KNN, lat2W
from esda.moran import Moran, Moran_Local

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "results_nc" / "spatial"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CONFIGS_DIR = PROJECT_DIR / "configs"

DIMS = {
    "FDSI": "fdsi_score",
    "D1_Climate": "D1_score",
    "D2_Morphology": "D2_score",
    "D3_Technical": "D3_score",
    "D4_Economic": "D4_score",
    "D5_Uncertainty": "D5_score",
}

LISA_COLORS = {
    "HH": "#E63946",   # High-High hotspot
    "LL": "#457B9D",   # Low-Low coldspot
    "HL": "#F4A261",   # High-Low outlier
    "LH": "#2A9D8F",   # Low-High outlier
    "ns": "#CCCCCC",   # Not significant
}


def load_coords():
    coords = {}
    for yf in CONFIGS_DIR.glob("*.yaml"):
        if yf.stem.startswith("_"):
            continue
        try:
            with open(yf) as f:
                cfg = yaml.safe_load(f)
            coords[yf.stem] = (cfg["city"]["latitude"], cfg["city"]["longitude"])
        except Exception:
            pass
    return coords


def main():
    log.info("=" * 60)
    log.info("Step 9: Spatial Autocorrelation Analysis")
    log.info("=" * 60)

    # ── Load ─────────────────────────────────────────────────────────────────
    sm = pd.read_csv(PROJECT_DIR / "results" / "fdsi" / "suitability_matrix.csv")
    sm["city_key"] = sm["city"].str.lower()
    coords = load_coords()
    sm["lat"] = sm["city_key"].map({k: v[0] for k, v in coords.items()})
    sm["lon"] = sm["city_key"].map({k: v[1] for k, v in coords.items()})

    # Drop rows with missing coords or scores
    sm = sm.dropna(subset=["lat", "lon"] + list(DIMS.values()))
    log.info(f"  {len(sm)} cities with coords + D1-D5 scores")

    # ── Spatial weights (KNN k=5) ─────────────────────────────────────────────
    log.info("\n[1] Building spatial weights matrix (KNN k=5)")
    pts = list(zip(sm["lon"].values, sm["lat"].values))
    W = KNN.from_array(np.array(pts), k=5)
    W.transform = "r"   # row-standardise
    log.info(f"  W: {W.n} observations, min neighbours={min(W.cardinalities.values())}")

    # ── Global Moran's I ─────────────────────────────────────────────────────
    log.info("\n[2] Global Moran's I")
    global_rows = []
    for dim_name, col in DIMS.items():
        y = sm[col].values
        mi = Moran(y, W, permutations=999)
        sig = "***" if mi.p_sim < 0.001 else "**" if mi.p_sim < 0.01 else \
              "*" if mi.p_sim < 0.05 else "ns"
        interp = "positive clustering" if (mi.I > 0 and mi.p_sim < 0.05) else \
                 "negative clustering" if (mi.I < 0 and mi.p_sim < 0.05) else "random"
        log.info(f"  {dim_name:15s}: I={mi.I:+.4f}  E[I]={mi.EI:.4f}  "
                 f"p={mi.p_sim:.4f} {sig}  → {interp}")
        global_rows.append({
            "dimension": dim_name, "Moran_I": round(mi.I, 4),
            "Expected_I": round(mi.EI, 4), "z_score": round(mi.z_sim, 4),
            "p_value": round(mi.p_sim, 4), "significance": sig,
            "interpretation": interp,
        })

    global_df = pd.DataFrame(global_rows)
    global_df.to_csv(OUT_DIR / "global_morans_i.csv", index=False)

    # ── Local Moran's I (LISA) for FDSI ──────────────────────────────────────
    log.info("\n[3] Local Moran's I (LISA) — FDSI")
    y_fdsi = sm["fdsi_score"].values
    lisa = Moran_Local(y_fdsi, W, permutations=999, seed=42)

    # Classify: q=1→HH, q=2→LH, q=3→LL, q=4→HL
    quad_map = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    sm["lisa_quad"] = [quad_map.get(q, "ns") for q in lisa.q]
    sm["lisa_Is"] = lisa.Is
    sm["lisa_p"] = lisa.p_sim
    sm["lisa_cluster"] = sm.apply(
        lambda r: r["lisa_quad"] if r["lisa_p"] < 0.05 else "ns", axis=1
    )

    log.info("  LISA cluster counts:")
    for cat, cnt in sm["lisa_cluster"].value_counts().items():
        cities_in = sm[sm["lisa_cluster"] == cat]["city"].tolist()
        log.info(f"    {cat:3s}: {cnt:2d} cities — {cities_in}")

    # Save LISA results
    lisa_out = sm[["city_key", "city", "lat", "lon",
                   "fdsi_score", "lisa_Is", "lisa_p",
                   "lisa_quad", "lisa_cluster"]].copy()
    lisa_out.to_csv(OUT_DIR / "lisa_clusters.csv", index=False)

    # Also do LISA for all dimensions
    all_lisa = []
    for dim_name, col in DIMS.items():
        y = sm[col].values
        li = Moran_Local(y, W, permutations=999, seed=42)
        for i, row in enumerate(sm.itertuples()):
            quad = quad_map.get(li.q[i], "ns")
            cluster = quad if li.p_sim[i] < 0.05 else "ns"
            all_lisa.append({
                "city": row.city_key, "dimension": dim_name,
                "local_I": round(li.Is[i], 4),
                "p_value": round(li.p_sim[i], 4),
                "quad": quad, "cluster": cluster,
            })
    pd.DataFrame(all_lisa).to_csv(OUT_DIR / "lisa_all_dimensions.csv", index=False)

    # ── Moran scatter plot (FDSI) ─────────────────────────────────────────────
    log.info("\n[4] Moran scatter plot")
    y_std = (y_fdsi - y_fdsi.mean()) / y_fdsi.std()
    Wy_std = np.array([W.sparse.dot(y_fdsi)])[0]
    Wy_std = (Wy_std - Wy_std.mean()) / Wy_std.std()

    fig, ax = plt.subplots(figsize=(7, 6))
    cluster_colors = [LISA_COLORS[sm["lisa_cluster"].iloc[i]] for i in range(len(sm))]
    ax.scatter(y_std, Wy_std, c=cluster_colors, s=60, edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    # Regression line
    slope = float(np.polyfit(y_std, Wy_std, 1)[0])
    xl = np.linspace(y_std.min(), y_std.max(), 100)
    ax.plot(xl, slope * xl, color="black", linewidth=1.5, label=f"Slope = Moran's I ≈ {slope:.3f}")
    # Annotate cities
    for i, row in enumerate(sm.itertuples()):
        if sm["lisa_cluster"].iloc[i] != "ns":
            ax.annotate(row.city, (y_std[i], Wy_std[i]),
                        fontsize=6.5, xytext=(4, 3), textcoords="offset points")
    # Legend
    patches = [mpatches.Patch(color=v, label=k) for k, v in LISA_COLORS.items()]
    ax.legend(handles=patches, fontsize=8, loc="upper left")
    ax.set_xlabel("Standardized FDSI (z-score)")
    ax.set_ylabel("Spatially Lagged FDSI (z-score)")
    mi_fdsi = global_rows[0]
    ax.set_title(f"Moran Scatter Plot — FDSI\n"
                 f"Moran's I = {mi_fdsi['Moran_I']:.4f}, p = {mi_fdsi['p_value']:.4f}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "moran_scatterplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Global Moran's I bar chart ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    bar_colors = ["#E63946" if r["p_value"] < 0.05 else "#AAAAAA"
                  for _, r in global_df.iterrows()]
    bars = ax.bar(global_df["dimension"], global_df["Moran_I"],
                  color=bar_colors, alpha=0.85, width=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(global_df["Expected_I"].mean(), color="gray", linestyle="--",
               linewidth=1, alpha=0.6, label="E[I] (expected under randomness)")
    for bar, (_, row) in zip(bars, global_df.iterrows()):
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else \
              "*" if row["p_value"] < 0.05 else "ns"
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01 if bar.get_height() >= 0 else bar.get_height() - 0.03,
                f"{row['Moran_I']:.3f}\n({sig})", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Global Moran's I")
    ax.set_title("Global Moran's I by Dimension (KNN-5 Spatial Weights)")
    ax.legend(fontsize=8)
    red_patch = mpatches.Patch(color="#E63946", label="p < 0.05")
    gray_patch = mpatches.Patch(color="#AAAAAA", label="p ≥ 0.05 (not significant)")
    ax.legend(handles=[red_patch, gray_patch], fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "global_morans_barchart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── LISA map (lat/lon scatter) ────────────────────────────────────────────
    log.info("\n[5] LISA map")
    fig, ax = plt.subplots(figsize=(11, 7))
    for cluster_type, color in LISA_COLORS.items():
        mask = sm["lisa_cluster"] == cluster_type
        if mask.any():
            ax.scatter(sm.loc[mask, "lon"], sm.loc[mask, "lat"],
                       c=color, s=120, label=cluster_type,
                       edgecolor="white", linewidth=0.8, zorder=3, alpha=0.9)
    # Label significant cities
    for _, row in sm.iterrows():
        if row["lisa_cluster"] != "ns":
            ax.annotate(row["city"], (row["lon"], row["lat"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
    # ns cities as small dots
    mask_ns = sm["lisa_cluster"] == "ns"
    ax.scatter(sm.loc[mask_ns, "lon"], sm.loc[mask_ns, "lat"],
               c="#CCCCCC", s=40, edgecolor="white", linewidth=0.5,
               zorder=2, alpha=0.6, label="Not significant")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("LISA Clusters — FDSI Spatial Autocorrelation\n"
                 "(HH=hotspot, LL=coldspot, HL/LH=outliers)")
    ax.legend(title="LISA type", fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "lisa_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info(f"\n✓ All spatial outputs → {OUT_DIR}")
    mi_fdsi_val = global_rows[0]
    log.info(f"  FDSI Global Moran's I = {mi_fdsi_val['Moran_I']:.4f} "
             f"(p={mi_fdsi_val['p_value']:.4f}, {mi_fdsi_val['significance']})")


if __name__ == "__main__":
    main()
