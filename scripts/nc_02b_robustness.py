#!/usr/bin/env python3
"""
============================================================================
NC Phase 2b: 鲁棒性检验 — 核心发现是否经得起审稿人质疑
============================================================================
审稿人最可能的质疑：
  "你的结论是否只因为你选了特定的权重/维度/指数构造方式？"

四项检验：
  1. 等权 FDSI — 不用 entropy+AHP，五维度各 0.20
  2. 去 D5 的 FDSI — 只用 D1-D4 等权(各0.25)，排除确定性维度
  3. 去 D4 的 FDSI — 只用 D1-D3+D5 等权(各0.25)，排除经济维度
  4. PCA composite — 纯数据驱动，不做任何主观赋权

核心检验标准：
  ✓ 通过: 每种替代方案下 GHI rank vs 替代FDSI rank 的 Spearman r_s < 0.85
          → "适宜性 ≠ 资源" 的结论不依赖于特定指数构造
  ✓ 通过: 替代排名与原始排名的 Spearman r_s > 0.80
          → 排名结论是稳健的
  ✓ 通过: "高辐照陷阱" 和 "隐藏冠军" 城市在不同方案下持续出现

产出：
  results_nc/robustness/
    robustness_summary.csv               — 各方案的 Spearman 统计
    alternative_rankings.csv             — 39城 × 5种排名对比
    persistent_misclass.csv              — 在所有方案下均被误分类的城市
    robustness_report.json               — 完整结果

用法：
  cd ~/Desktop/multi-cities-bipv-nc
  python scripts/nc_02b_robustness.py
============================================================================
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FDSI_DIR = PROJECT_DIR / "results" / "fdsi"
OUTPUT_DIR = PROJECT_DIR / "results_nc" / "robustness"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load dimension scores and GHI."""
    matrix = pd.read_csv(FDSI_DIR / "suitability_matrix.csv")
    indicators = pd.read_csv(FDSI_DIR / "integrated_indicators.csv")

    df = pd.DataFrame({
        "city": matrix["city"].values if "city" in matrix.columns else matrix.iloc[:, 0].values,
    })

    # Try to get name_en
    if "name_en_energy" in indicators.columns:
        df["name_en"] = indicators["name_en_energy"].values
    elif "name_en" in indicators.columns:
        df["name_en"] = indicators["name_en"].values

    # Dimension scores
    for dim in ["D1", "D2", "D3", "D4", "D5"]:
        col = f"{dim}_score"
        if col in matrix.columns:
            df[dim] = matrix[col].values
        else:
            log.warning(f"  Column {col} not found in suitability_matrix.csv")

    # GHI
    if "d1_1_ghi_annual_kwh" in indicators.columns:
        df["ghi_annual"] = indicators["d1_1_ghi_annual_kwh"].values

    # Original FDSI
    if "fdsi_score" in indicators.columns:
        df["fdsi_original"] = indicators["fdsi_score"].values
    elif "fdsi_score" in matrix.columns:
        df["fdsi_original"] = matrix["fdsi_score"].values

    log.info(f"  Loaded {len(df)} cities with D1-D5 scores")
    return df


def compute_alternative_fdsi(df):
    """Compute 4 alternative composite scores."""
    dims = ["D1", "D2", "D3", "D4", "D5"]

    # --- Alt 1: Equal weights (0.20 each) ---
    df["fdsi_equal_wt"] = df[dims].mean(axis=1)

    # --- Alt 2: Drop D5 (equal weight D1-D4, each 0.25) ---
    df["fdsi_no_d5"] = df[["D1", "D2", "D3", "D4"]].mean(axis=1)

    # --- Alt 3: Drop D4 (equal weight D1-D3+D5, each 0.25) ---
    df["fdsi_no_d4"] = df[["D1", "D2", "D3", "D5"]].mean(axis=1)

    # --- Alt 4: PCA composite (first principal component) ---
    from numpy.linalg import eigh
    X = df[dims].values
    # Standardize
    X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    # Covariance matrix
    cov = np.cov(X_std, rowvar=False)
    eigenvalues, eigenvectors = eigh(cov)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # PC1 score
    pc1 = X_std @ eigenvectors[:, 0]
    # Flip sign if PC1 is negatively correlated with original FDSI
    if np.corrcoef(pc1, df["fdsi_original"])[0, 1] < 0:
        pc1 = -pc1

    # Normalize to [0, 1] for comparability
    pc1_norm = (pc1 - pc1.min()) / (pc1.max() - pc1.min())
    df["fdsi_pca"] = pc1_norm

    # Log PCA details
    var_explained = eigenvalues / eigenvalues.sum()
    log.info(f"\n  PCA 方差解释率:")
    for i, (ev, ve) in enumerate(zip(eigenvalues, var_explained)):
        log.info(f"    PC{i + 1}: eigenvalue={ev:.3f}, variance={ve:.1%}")
    log.info(f"  PC1 载荷:")
    for dim, loading in zip(dims, eigenvectors[:, 0]):
        log.info(f"    {dim}: {loading:.3f}")

    return df, {
        "pca_var_explained_pc1": round(var_explained[0] * 100, 1),
        "pca_loadings": {dim: round(float(l), 3) for dim, l in zip(dims, eigenvectors[:, 0])},
    }


def run_robustness_tests(df):
    """Compare each alternative with GHI ranking and original FDSI ranking."""
    log.info("\n" + "=" * 70)
    log.info("鲁棒性检验")
    log.info("=" * 70)

    alternatives = {
        "original": "fdsi_original",
        "equal_weight": "fdsi_equal_wt",
        "no_D5": "fdsi_no_d5",
        "no_D4": "fdsi_no_d4",
        "pca_composite": "fdsi_pca",
    }

    # GHI ranking
    df["ghi_rank"] = df["ghi_annual"].rank(ascending=False).astype(int)

    results = {}
    rank_cols = {}

    for name, col in alternatives.items():
        rank_col = f"rank_{name}"
        df[rank_col] = df[col].rank(ascending=False).astype(int)
        rank_cols[name] = rank_col

        # Test 1: r_s with GHI → should be < 0.85 (core finding survives)
        rs_ghi, p_ghi = stats.spearmanr(df["ghi_rank"], df[rank_col])

        # Test 2: r_s with original FDSI → should be > 0.80 (ranking stable)
        if name != "original":
            rs_orig, p_orig = stats.spearmanr(
                df["rank_original"], df[rank_col]
            )
        else:
            rs_orig, p_orig = 1.0, 0.0

        # Test 3: median rank shift from GHI
        shifts = (df["ghi_rank"] - df[rank_col]).abs()

        results[name] = {
            "rs_vs_ghi": round(rs_ghi, 4),
            "p_vs_ghi": round(p_ghi, 4),
            "rs_vs_original": round(rs_orig, 4),
            "p_vs_original": round(p_orig, 4),
            "median_shift_from_ghi": round(shifts.median(), 1),
            "mean_shift_from_ghi": round(shifts.mean(), 1),
            "core_finding_survives": rs_ghi < 0.85,
            "ranking_stable": rs_orig > 0.80 if name != "original" else True,
        }

        verdict_core = "PASS" if rs_ghi < 0.85 else "FAIL"
        verdict_stable = "PASS" if (name == "original" or rs_orig > 0.80) else "FAIL"

        log.info(f"\n  ── {name} ──")
        log.info(f"    r_s vs GHI:      {rs_ghi:.4f} (p={p_ghi:.4f})  [{verdict_core}]")
        if name != "original":
            log.info(f"    r_s vs original: {rs_orig:.4f} (p={p_orig:.4f})  [{verdict_stable}]")
        log.info(f"    Median shift from GHI: {shifts.median():.0f} positions")

    # ── Overall verdict ──
    log.info(f"\n  {'=' * 50}")
    log.info(f"  鲁棒性综合判定")
    log.info(f"  {'=' * 50}")

    all_core_pass = all(r["core_finding_survives"] for r in results.values())
    all_stable = all(r["ranking_stable"] for r in results.values())

    if all_core_pass:
        log.info(f"  ✓ 核心发现 (适宜性 ≠ 资源) 在所有 {len(alternatives)} 种方案下均成立")
    else:
        failed = [n for n, r in results.items() if not r["core_finding_survives"]]
        log.info(f"  ⚠ 核心发现在以下方案下不成立: {failed}")

    if all_stable:
        log.info(f"  ✓ 排名在所有替代方案下均稳定 (r_s > 0.80)")
    else:
        unstable = [n for n, r in results.items() if not r["ranking_stable"]]
        log.info(f"  ⚠ 排名在以下方案下不稳定: {unstable}")

    return df, results


def check_persistent_misclass(df):
    """Find cities misclassified under ALL alternative schemes."""
    log.info("\n" + "=" * 70)
    log.info("持续误分类城市 — 在所有方案下都被 GHI 错误判断")
    log.info("=" * 70)

    n = len(df)
    n_tercile = n // 3

    rank_schemes = ["rank_original", "rank_equal_weight", "rank_no_D5",
                    "rank_no_D4", "rank_pca_composite"]

    # For each scheme, identify cities where GHI and that scheme disagree on tercile
    misclass_counts = {}
    for _, row in df.iterrows():
        city = row["name_en"]
        ghi_tercile = "High" if row["ghi_rank"] <= n_tercile else (
            "Low" if row["ghi_rank"] > n - n_tercile else "Medium"
        )
        n_misclass = 0
        for scheme in rank_schemes:
            scheme_tercile = "High" if row[scheme] <= n_tercile else (
                "Low" if row[scheme] > n - n_tercile else "Medium"
            )
            if ghi_tercile != scheme_tercile:
                n_misclass += 1
        misclass_counts[city] = n_misclass

    df["n_schemes_misclassified"] = df["name_en"].map(misclass_counts)

    # Persistent = misclassified in ALL 5 schemes
    persistent = df[df["n_schemes_misclassified"] == len(rank_schemes)]
    log.info(f"\n  在全部 {len(rank_schemes)} 种方案下都被 GHI 误分类的城市: {len(persistent)}")
    for _, r in persistent.sort_values("ghi_rank").iterrows():
        log.info(f"    {r['name_en']:12s}: GHI#{r['ghi_rank']:2d}, "
                 f"原始FDSI#{r['rank_original']:2d}, 等权#{r['rank_equal_weight']:2d}, "
                 f"无D5#{r['rank_no_D5']:2d}, 无D4#{r['rank_no_D4']:2d}, "
                 f"PCA#{r['rank_pca_composite']:2d}")

    # Majority = misclassified in ≥ 3 schemes
    majority = df[df["n_schemes_misclassified"] >= 3]
    log.info(f"\n  在 ≥3 种方案下被误分类的城市: {len(majority)}")

    persistent[["city", "name_en", "ghi_rank", "rank_original", "rank_equal_weight",
                 "rank_no_D5", "rank_no_D4", "rank_pca_composite",
                 "n_schemes_misclassified"]].to_csv(
        OUTPUT_DIR / "persistent_misclass.csv", index=False, encoding="utf-8-sig"
    )

    return {
        "n_persistent_all": len(persistent),
        "n_persistent_majority": len(majority),
        "persistent_cities": ", ".join(persistent["name_en"].tolist()),
    }


def main():
    log.info("=" * 70)
    log.info("NC Phase 2b: 鲁棒性检验")
    log.info("核心问题: 误分类发现是否依赖于特定指数构造？")
    log.info("=" * 70)

    log.info("\n[0] 加载数据...")
    df = load_data()

    log.info("\n[1] 计算替代 FDSI...")
    df, pca_info = compute_alternative_fdsi(df)

    log.info("\n[2] 运行鲁棒性检验...")
    df, test_results = run_robustness_tests(df)

    log.info("\n[3] 检查持续误分类城市...")
    persist_results = check_persistent_misclass(df)

    # ── Save all ──
    # Summary table
    summary_rows = []
    for name, r in test_results.items():
        summary_rows.append({"scheme": name, **r})
    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "robustness_summary.csv", index=False, encoding="utf-8-sig"
    )

    # Full rankings
    rank_cols = ["city", "name_en", "ghi_annual", "ghi_rank",
                 "fdsi_original", "rank_original",
                 "fdsi_equal_wt", "rank_equal_weight",
                 "fdsi_no_d5", "rank_no_D5",
                 "fdsi_no_d4", "rank_no_D4",
                 "fdsi_pca", "rank_pca_composite",
                 "n_schemes_misclassified"]
    df[[c for c in rank_cols if c in df.columns]].sort_values("rank_original").to_csv(
        OUTPUT_DIR / "alternative_rankings.csv", index=False, encoding="utf-8-sig"
    )

    # JSON report
    report = {
        "tests": test_results,
        "pca": pca_info,
        "persistent_misclass": persist_results,
        "overall": {
            "core_finding_robust": all(r["core_finding_survives"] for r in test_results.values()),
            "rankings_stable": all(r["ranking_stable"] for r in test_results.values()),
        }
    }
    def _json_safe(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(OUTPUT_DIR / "robustness_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=_json_safe)

    # ── Final verdict ──
    log.info(f"\n{'=' * 70}")
    log.info(f"最终结论")
    log.info(f"{'=' * 70}")
    if report["overall"]["core_finding_robust"]:
        log.info(f"  ✓ 核心发现 (r_s < 0.85 in all schemes) 完全稳健")
        log.info(f"    → 可以自信地写: 'This finding is robust to alternative")
        log.info(f"       index constructions including equal weighting, dimension")
        log.info(f"       removal, and data-driven PCA composites.'")
    else:
        log.info(f"  ⚠ 核心发现在某些方案下不成立，需要谨慎表述")

    if persist_results["n_persistent_all"] > 0:
        log.info(f"\n  {persist_results['n_persistent_all']} 个城市在所有方案下均被 GHI 误分类")
        log.info(f"    → 这些城市是最强的 misclassification 证据")
        log.info(f"    → 城市: {persist_results['persistent_cities']}")

    log.info(f"\n  产出目录: {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.glob("*")):
        log.info(f"    {f.name}")
    log.info("\n完成。")


if __name__ == "__main__":
    main()
