#!/usr/bin/env python3
"""
Step 10: Bootstrap ranking stability + Leave-One-Out analysis for 39-city FDSI.
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
import matplotlib.colors as mcolors

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "results_nc" / "bootstrap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIMS = ["D1_score", "D2_score", "D3_score", "D4_score", "D5_score"]
DIM_WEIGHTS = [0.25, 0.20, 0.20, 0.20, 0.15]   # must match 05_fdsi_scoring weights
B = 1000
RNG = np.random.default_rng(42)


def load_data():
    sm = pd.read_csv(PROJECT_DIR / "results" / "fdsi" / "suitability_matrix.csv")
    sm["city_key"] = sm["city"].str.lower()
    # English name
    if "name_en" not in sm.columns:
        sm["name_en"] = sm["city"]
    return sm


def compute_fdsi(df_sub: pd.DataFrame) -> pd.Series:
    """Recompute FDSI after min-max renormalization on the subset."""
    X = df_sub[DIMS].copy()
    # Min-max across the subset
    for col in DIMS:
        col_min, col_max = X[col].min(), X[col].max()
        if col_max > col_min:
            X[col] = (X[col] - col_min) / (col_max - col_min)
        else:
            X[col] = 0.5
    fdsi = (X * DIM_WEIGHTS).sum(axis=1)
    return fdsi


def bootstrap_rankings(df: pd.DataFrame, B: int = 1000) -> pd.DataFrame:
    """Bootstrap with replacement, recompute FDSI & rank each iteration."""
    n = len(df)
    all_ranks = {city: [] for city in df["city_key"]}
    all_scores = {city: [] for city in df["city_key"]}

    for b in range(B):
        idx = RNG.choice(n, size=n, replace=True)
        sample = df.iloc[idx].copy().reset_index(drop=True)
        fdsi = compute_fdsi(sample)
        # Average score for each original city (a city can appear multiple times)
        sample["fdsi_boot"] = fdsi.values
        city_mean = sample.groupby("city_key")["fdsi_boot"].mean()
        # Rank all 39 original cities by their bootstrapped score
        # Cities not in sample get the last-seen score; assign NaN for missing
        scores_this = df["city_key"].map(city_mean)
        ranks_this = scores_this.rank(ascending=False, method="min")
        for i, city in enumerate(df["city_key"]):
            if not np.isnan(scores_this.iloc[i]):
                all_ranks[city].append(int(ranks_this.iloc[i]))
                all_scores[city].append(scores_this.iloc[i])

    rows = []
    for city in df["city_key"]:
        r = np.array(all_ranks[city])
        s = np.array(all_scores[city])
        rows.append({
            "city_key": city,
            "observed_rank": int(df[df["city_key"] == city]["fdsi_rank"].iloc[0])
                             if "fdsi_rank" in df.columns else None,
            "name_en": df[df["city_key"] == city]["name_en"].iloc[0],
            "boot_rank_mean": round(r.mean(), 2) if len(r) else np.nan,
            "boot_rank_p025": int(np.percentile(r, 2.5)) if len(r) else np.nan,
            "boot_rank_p975": int(np.percentile(r, 97.5)) if len(r) else np.nan,
            "boot_rank_ci_width": int(np.percentile(r, 97.5) - np.percentile(r, 2.5))
                                  if len(r) else np.nan,
            "boot_rank_std": round(r.std(), 2) if len(r) else np.nan,
            "boot_score_mean": round(s.mean(), 4) if len(s) else np.nan,
            "boot_score_std": round(s.std(), 4) if len(s) else np.nan,
            "n_bootstrap_appearances": len(r),
        })
    return pd.DataFrame(rows)


def leave_one_out(df: pd.DataFrame) -> pd.DataFrame:
    """For each city removed, recompute ranks for all remaining cities."""
    n = len(df)
    # shift[city][removed] = new_rank - original_rank
    shifts = {c: {} for c in df["city_key"]}

    orig_fdsi = compute_fdsi(df)
    orig_rank = orig_fdsi.rank(ascending=False, method="min").astype(int)
    df = df.copy()
    df["orig_rank"] = orig_rank.values

    for i in range(n):
        removed_city = df["city_key"].iloc[i]
        df_sub = df.drop(df.index[i]).reset_index(drop=True)
        fdsi_sub = compute_fdsi(df_sub)
        rank_sub = fdsi_sub.rank(ascending=False, method="min").astype(int)
        df_sub["new_rank"] = rank_sub.values

        for _, row in df_sub.iterrows():
            city = row["city_key"]
            old_r = df[df["city_key"] == city]["orig_rank"].iloc[0]
            new_r = row["new_rank"]
            shifts[city][removed_city] = int(new_r) - int(old_r)

    shift_df = pd.DataFrame(shifts).T  # rows=cities removed, cols=affected city
    max_shift = shift_df.abs().max(axis=1)
    loo_summary = pd.DataFrame({
        "city_key": shift_df.index,
        "max_rank_shift_when_removed": max_shift.values,
    })
    # Also: for each city, what's the biggest shift it experiences when ANY city is removed
    impact_on = shift_df.abs().max(axis=0)
    loo_summary2 = pd.DataFrame({
        "city_key": impact_on.index,
        "max_rank_shift_experienced": impact_on.values,
    })
    loo_combined = loo_summary.merge(loo_summary2, on="city_key")
    loo_combined = loo_combined.merge(df[["city_key","name_en","orig_rank"]], on="city_key")
    return loo_combined.sort_values("max_rank_shift_experienced", ascending=False)


def main():
    log.info("=" * 60)
    log.info("Step 10: Bootstrap Ranking Stability")
    log.info("=" * 60)

    df = load_data()
    # Add rank if missing
    if "fdsi_rank" not in df.columns:
        df["fdsi_rank"] = df["fdsi_score"].rank(ascending=False, method="min").astype(int)
    log.info(f"  Loaded {len(df)} cities")

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    log.info(f"\n[1] Bootstrap (B={B}, with replacement)")
    ci_df = bootstrap_rankings(df, B=B)
    ci_df = ci_df.sort_values("boot_rank_mean")

    # Flag unstable cities (CI width > 10)
    ci_df["unstable"] = ci_df["boot_rank_ci_width"] > 10

    log.info("\n  Ranking 95% CI (all cities):")
    log.info(f"  {'City':15s} {'Obs':>5} {'Boot':>6} {'p2.5':>5} {'p97.5':>6} {'CI':>4} {'Unstable'}")
    log.info("  " + "-" * 55)
    for _, row in ci_df.iterrows():
        flag = " ← UNSTABLE" if row["unstable"] else ""
        log.info(f"  {row['name_en']:15s} {row['observed_rank']:>5} {row['boot_rank_mean']:>6.1f} "
                 f"{row['boot_rank_p025']:>5} {row['boot_rank_p975']:>6} "
                 f"{row['boot_rank_ci_width']:>4}{flag}")

    ci_df.to_csv(OUT_DIR / "ranking_ci.csv", index=False)

    unstable = ci_df[ci_df["unstable"]]
    log.info(f"\n  Unstable cities (CI width > 10): {unstable['name_en'].tolist()}")
    log.info(f"  Top-5 stability (CI widths): "
             f"{ci_df.head(5)[['name_en','boot_rank_ci_width']].to_dict('records')}")
    log.info(f"  Bottom-5 stability (CI widths): "
             f"{ci_df.tail(5)[['name_en','boot_rank_ci_width']].to_dict('records')}")

    # ── Leave-one-out ─────────────────────────────────────────────────────────
    log.info(f"\n[2] Leave-One-Out sensitivity")
    loo_df = leave_one_out(df)
    loo_df.to_csv(OUT_DIR / "leave_one_out.csv", index=False)

    log.info("  Cities most affected by LOO removal (top 10):")
    for _, row in loo_df.head(10).iterrows():
        log.info(f"  {row['name_en']:15s} (rank #{int(row['orig_rank']):2d}): "
                 f"max shift = {int(row['max_rank_shift_experienced']):+d} positions")

    # ── Plot: CI bar chart ────────────────────────────────────────────────────
    log.info("\n[3] Ranking stability CI chart")
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ["#E63946" if u else "#457B9D" for u in ci_df["unstable"]]
    y_pos = range(len(ci_df))
    ax.barh(y_pos, ci_df["boot_rank_ci_width"], color=colors, alpha=0.75, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{r['name_en']} (#{r['observed_rank']})" for _, r in ci_df.iterrows()],
        fontsize=8
    )
    ax.axvline(10, color="red", linestyle="--", linewidth=1.2, label="Unstable threshold (CI=10)")
    ax.set_xlabel("95% CI Width (rank positions)")
    ax.set_title(f"Bootstrap Ranking Stability (B={B})\n"
                 "Red = unstable (CI > 10 positions)")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ranking_ci_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Plot: Heatmap of rank distribution ───────────────────────────────────
    log.info("[4] Bootstrap rank distribution heatmap")
    n = len(df)
    # Re-run bootstrap collecting full rank distributions
    rank_matrix = np.zeros((n, n))  # [city_idx, rank] = count
    city_order = ci_df["city_key"].tolist()  # sorted by boot_rank_mean
    city_idx = {c: i for i, c in enumerate(city_order)}

    for b in range(B):
        idx = RNG.choice(n, size=n, replace=True)
        sample = df.iloc[idx].copy().reset_index(drop=True)
        fdsi = compute_fdsi(sample)
        sample["fdsi_boot"] = fdsi.values
        city_mean = sample.groupby("city_key")["fdsi_boot"].mean()
        scores_this = df["city_key"].map(city_mean)
        ranks_this = scores_this.rank(ascending=False, method="min")
        for i_orig, city in enumerate(df["city_key"]):
            if city in city_idx and pd.notna(scores_this.iloc[i_orig]):
                rank_val = int(ranks_this.iloc[i_orig] - 1)  # 0-indexed
                rank_matrix[city_idx[city], rank_val] += 1

    # Normalize rows
    rank_matrix_norm = rank_matrix / rank_matrix.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(13, 10))
    im = ax.imshow(rank_matrix_norm, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=rank_matrix_norm.max())
    ax.set_xticks(range(0, n, 5))
    ax.set_xticklabels(range(1, n + 1, 5), fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [f"{ci_df.iloc[i]['name_en']} (#{ci_df.iloc[i]['observed_rank']})" for i in range(n)],
        fontsize=7
    )
    ax.set_xlabel("Bootstrap Rank")
    ax.set_ylabel("City (sorted by mean bootstrap rank)")
    ax.set_title(f"Bootstrap Rank Distribution Heatmap (B={B})\n"
                 "Color = probability of achieving each rank")
    plt.colorbar(im, ax=ax, label="Probability", shrink=0.6)
    # Mark observed ranks
    obs_ranks = [int(df[df["city_key"] == c]["fdsi_rank"].iloc[0]) - 1
                 if len(df[df["city_key"] == c]) > 0 else 0
                 for c in city_order]
    ax.scatter(obs_ranks, range(n), marker="|", color="blue", s=30, zorder=5,
               label="Observed rank")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ranking_stability_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info(f"\n✓ All bootstrap outputs → {OUT_DIR}")
    n_unstable = ci_df["unstable"].sum()
    log.info(f"  {n_unstable}/{n} cities have CI width > 10 (unstable ranking)")


if __name__ == "__main__":
    main()
