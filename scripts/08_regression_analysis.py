#!/usr/bin/env python3
"""
Step 8: OLS regression + LASSO — which city characteristics drive FDSI most?
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
import yaml

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "results_nc" / "regression"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CONFIGS_DIR = PROJECT_DIR / "configs"


def load_electricity_prices():
    prices = {}
    for yf in CONFIGS_DIR.glob("*.yaml"):
        if yf.stem.startswith("_"):
            continue
        try:
            with open(yf) as f:
                cfg = yaml.safe_load(f)
            prices[yf.stem] = cfg.get("economics", {}).get("electricity_price_cny_kwh", np.nan)
        except Exception:
            pass
    return prices


def main():
    log.info("=" * 60)
    log.info("Step 8: Regression Analysis")
    log.info("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    ind = pd.read_csv(PROJECT_DIR / "results" / "fdsi" / "integrated_indicators.csv")
    sm_df = pd.read_csv(PROJECT_DIR / "results" / "fdsi" / "suitability_matrix.csv")

    # Merge on city
    sm_df["city_key"] = sm_df["city"].str.lower()
    ind["city_key"] = ind["city"].str.lower()
    df = ind.merge(sm_df[["city_key", "fdsi_score"]], on="city_key", how="left",
                   suffixes=("", "_sm"))
    # Resolve duplicate fdsi_score
    if "fdsi_score_sm" in df.columns:
        df["fdsi_score"] = df["fdsi_score_sm"].fillna(df["fdsi_score"])
        df = df.drop(columns=["fdsi_score_sm"])

    # Add electricity prices from YAML
    prices = load_electricity_prices()
    df["electricity_price"] = df["city_key"].map(prices)

    log.info(f"  Dataset: {len(df)} cities")

    # ── Feature selection ─────────────────────────────────────────────────────
    candidate_features = {
        "ghi_annual":       "d1_1_ghi_annual_kwh",
        "temp_annual":      "d1_2_temp_annual_c",
        "sunshine_hours":   "d1_4_sunshine_hours",
        "height_mean":      "d2_1_height_mean",
        "building_density": "d2_2_building_density",
        "far":              "d2_5_far",
        "shading_factor":   "d3_2_shading_factor_mean",
        "deployable_mw":    "d3_4_total_deployable_mw",
        "lcoe":             "d4_1_lcoe_cny_kwh",
        "pbt_years":        "d4_2_pbt_years",
        "yield_cv":         "d5_1_yield_cv",
        "pbt_ci_width":     "d5_2_pbt_ci95_width",
        "elec_price":       "electricity_price",
        "latitude":         None,   # derive from coords or use PVGIS data
    }

    # Add latitude from YAML if not in df
    if "latitude" not in df.columns:
        lats = {}
        for yf in CONFIGS_DIR.glob("*.yaml"):
            if yf.stem.startswith("_"):
                continue
            try:
                with open(yf) as f:
                    cfg = yaml.safe_load(f)
                lats[yf.stem] = cfg["city"]["latitude"]
            except Exception:
                pass
        df["latitude"] = df["city_key"].map(lats)

    # Climate zone dummies
    df["cz"] = df.get("climate_zone_morph", df.get("climate_zone_energy",
                       pd.Series(["unknown"] * len(df), index=df.index)))
    cz_dummies = pd.get_dummies(df["cz"], prefix="cz", drop_first=True)

    # Build feature matrix
    feat_cols = []
    for feat_name, col in candidate_features.items():
        if col is None:
            col = feat_name
        if col in df.columns:
            feat_cols.append((feat_name, col))
        else:
            log.warning(f"  Feature '{feat_name}' col '{col}' not found — skipping")

    X_raw = pd.DataFrame(index=df.index)
    for feat_name, col in feat_cols:
        X_raw[feat_name] = pd.to_numeric(df[col], errors="coerce")

    # Add dummies
    X_raw = pd.concat([X_raw, cz_dummies], axis=1)

    y = pd.to_numeric(df["fdsi_score"], errors="coerce")

    # Drop rows with any NaN
    valid = X_raw.notna().all(axis=1) & y.notna()
    X_raw = X_raw[valid]
    y = y[valid]
    city_names = df.loc[valid, "city_key"].tolist()
    log.info(f"  Valid observations: {len(y)}")

    # ── VIF check ─────────────────────────────────────────────────────────────
    log.info("\n[1] VIF check (OLS features, standardized)")
    scaler = StandardScaler()
    Xz = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns, index=X_raw.index)
    Xz_const = sm.add_constant(Xz)

    vif_rows = []
    for i, col in enumerate(Xz.columns):
        try:
            vif = variance_inflation_factor(Xz_const.values, i + 1)
        except Exception:
            vif = np.nan
        vif_rows.append({"feature": col, "VIF": round(vif, 2)})
        flag = " ← HIGH" if vif > 10 else ""
        log.info(f"  {col:25s}: VIF={vif:.2f}{flag}")

    vif_df = pd.DataFrame(vif_rows)
    vif_df.to_csv(OUT_DIR / "vif_check.csv", index=False)

    # Drop high-VIF features (> 10) iteratively
    drop_cols = set()
    # Identify correlated pairs and remove the one more correlated with GHI
    high_vif = vif_df[vif_df["VIF"] > 10]["feature"].tolist()
    if high_vif:
        log.info(f"  High-VIF features to remove: {high_vif}")
        # Keep ghi_annual if it's there; drop others
        priority_keep = {"ghi_annual", "elec_price", "latitude", "far", "height_mean"}
        for feat in high_vif:
            if feat not in priority_keep:
                drop_cols.add(feat)
        # If still > 10, drop lcoe (highly correlated with ghi + elec_price)
        for feat in high_vif:
            if feat in {"lcoe", "pbt_years", "deployable_mw", "sunshine_hours"}:
                drop_cols.add(feat)

    X_ols = Xz.drop(columns=list(drop_cols), errors="ignore")
    log.info(f"  Features after VIF filter: {list(X_ols.columns)}")

    # ── OLS ──────────────────────────────────────────────────────────────────
    log.info("\n[2] OLS regression (standardized features)")
    X_const = sm.add_constant(X_ols)
    ols = sm.OLS(y.values, X_const.values).fit()
    log.info(f"  R²={ols.rsquared:.4f}, Adj-R²={ols.rsquared_adj:.4f}, n={len(y)}")

    coef_df = pd.DataFrame({
        "feature": ["const"] + list(X_ols.columns),
        "coef": ols.params,
        "std_err": ols.bse,
        "t_stat": ols.tvalues,
        "p_value": ols.pvalues,
    })
    coef_df["significant"] = coef_df["p_value"] < 0.05
    coef_df = coef_df.sort_values("coef", key=abs, ascending=False)
    coef_df.to_csv(OUT_DIR / "standardized_coefficients.csv", index=False)

    log.info("\n  Top standardized coefficients:")
    for _, row in coef_df[coef_df["feature"] != "const"].head(8).iterrows():
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else \
              "*" if row["p_value"] < 0.05 else "ns"
        log.info(f"  {row['feature']:25s}: β={row['coef']:+.4f}  p={row['p_value']:.4f} {sig}")

    # Save OLS summary
    with open(OUT_DIR / "ols_summary.txt", "w") as f:
        f.write(ols.summary().as_text())

    # ── LASSO ─────────────────────────────────────────────────────────────────
    log.info("\n[3] LASSO (cross-validated alpha)")
    lasso = LassoCV(cv=5, random_state=42, max_iter=5000, n_alphas=100)
    lasso.fit(X_ols.values, y.values)
    log.info(f"  Best alpha: {lasso.alpha_:.6f}")

    lasso_coef = pd.DataFrame({
        "feature": X_ols.columns,
        "lasso_coef": lasso.coef_,
    }).sort_values("lasso_coef", key=abs, ascending=False)
    lasso_coef["selected"] = lasso_coef["lasso_coef"] != 0
    lasso_coef.to_csv(OUT_DIR / "lasso_coefficients.csv", index=False)

    log.info("  LASSO selected features:")
    for _, row in lasso_coef[lasso_coef["selected"]].iterrows():
        log.info(f"    {row['feature']:25s}: {row['lasso_coef']:+.4f}")

    # ── Coefficient plot ───────────────────────────────────────────────────────
    log.info("\n[4] Coefficient plot")
    plot_df = coef_df[coef_df["feature"] != "const"].copy()
    plot_df = plot_df.sort_values("coef")

    fig, ax = plt.subplots(figsize=(8, max(5, len(plot_df) * 0.4 + 1)))
    colors = ["#E63946" if v > 0 else "#457B9D" for v in plot_df["coef"]]
    bars = ax.barh(plot_df["feature"], plot_df["coef"], color=colors, alpha=0.8)
    # Error bars
    ax.errorbar(plot_df["coef"], range(len(plot_df)),
                xerr=1.96 * plot_df["std_err"],
                fmt="none", color="black", capsize=3, linewidth=1)
    ax.axvline(0, color="black", linewidth=0.8)
    # Significance markers
    for i, (_, row) in enumerate(plot_df.iterrows()):
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else \
              "*" if row["p_value"] < 0.05 else ""
        if sig:
            ax.text(row["coef"] + (0.01 if row["coef"] >= 0 else -0.01),
                    i, sig, ha="left" if row["coef"] >= 0 else "right",
                    va="center", fontsize=10, color="black")
    ax.set_xlabel("Standardized coefficient β")
    ax.set_title(f"OLS Regression — Standardized Coefficients\n(R²={ols.rsquared:.3f}, n={len(y)})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "coefficient_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info(f"\n✓ All regression outputs → {OUT_DIR}")
    log.info(f"  R²={ols.rsquared:.4f}, Adj-R²={ols.rsquared_adj:.4f}")
    top3 = coef_df[coef_df["feature"] != "const"].head(3)["feature"].tolist()
    log.info(f"  Top-3 predictors: {top3}")


if __name__ == "__main__":
    main()
