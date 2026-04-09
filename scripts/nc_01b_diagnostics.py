#!/usr/bin/env python3
"""
============================================================================
NC Phase 1b: 15城数据诊断分析
============================================================================
三项诊断：
  1. GHI rank vs FDSI rank Spearman 相关 → 量化"适宜性 ≠ 资源"的核心发现
  2. D5 独立性检验 → D5 vs D1+D4 相关系数，揭示 D5 退化问题
  3. 数据质量 flag → 标记需要修复的城市

产出：
  results_nc/diagnostics/diagnostic_report.csv     — 汇总表
  results_nc/diagnostics/rank_comparison.csv       — GHI vs FDSI 排名对比
  results_nc/diagnostics/d5_independence.csv       — D5 独立性分析
  results_nc/diagnostics/data_quality_flags.csv    — 数据质量标记

用法：
  cd ~/Desktop/multi-cities-bipv-nc
  python scripts/nc_01b_diagnostics.py
============================================================================
"""

import logging
import sys
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
MORPH_DIR = PROJECT_DIR / "results" / "morphology"
ENERGY_DIR = PROJECT_DIR / "results" / "energy"
OUTPUT_DIR = PROJECT_DIR / "results_nc" / "diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    fdsi = pd.read_csv(FDSI_DIR / "fdsi_scores.csv")
    log.info(f"  FDSI: {len(fdsi)} cities")
    matrix = pd.read_csv(FDSI_DIR / "suitability_matrix.csv")
    indicators = pd.read_csv(FDSI_DIR / "integrated_indicators.csv")
    morph = pd.read_csv(MORPH_DIR / "cross_city_d2d3_summary.csv")
    return fdsi, matrix, indicators, morph


def diagnose_ghi_vs_fdsi(indicators, fdsi):
    log.info("\n" + "=" * 60)
    log.info("诊断 1: GHI rank vs FDSI rank 相关分析")
    log.info("=" * 60)

    df = pd.DataFrame({
        "city": indicators["city"].values,
        "name_en": indicators["name_en_energy"].values,
        "climate_zone": indicators["climate_zone_energy"].values,
        "ghi_annual": indicators["d1_1_ghi_annual_kwh"].values,
        "fdsi_score": indicators["fdsi_score"].values,
    })

    df["ghi_rank"] = df["ghi_annual"].rank(ascending=False).astype(int)
    df["fdsi_rank"] = df["fdsi_score"].rank(ascending=False).astype(int)
    df["rank_diff"] = df["ghi_rank"] - df["fdsi_rank"]
    df["abs_rank_diff"] = df["rank_diff"].abs()
    df = df.sort_values("fdsi_rank")

    r_s, p_value = stats.spearmanr(df["ghi_rank"], df["fdsi_rank"])
    tau, p_tau = stats.kendalltau(df["ghi_rank"], df["fdsi_rank"])
    r_p, p_pearson = stats.pearsonr(df["ghi_annual"], df["fdsi_score"])

    log.info(f"\n  Spearman r_s = {r_s:.4f}  (p = {p_value:.4f})")
    log.info(f"  Kendall  tau = {tau:.4f}  (p = {p_tau:.4f})")
    log.info(f"  Pearson  r   = {r_p:.4f}  (p = {p_pearson:.4f})")

    if p_value < 0.05:
        log.info(f"\n  → GHI排名与FDSI排名显著正相关 (p<0.05)")
        log.info(f"    但 r_s={r_s:.3f} 远低于 1.0，说明GHI只能解释部分变异")
    else:
        log.info(f"\n  → GHI排名与FDSI排名 **不显著相关** (p={p_value:.3f})")
        log.info(f"    这是一个很强的发现：太阳能资源丰裕度无法预测BIPV适宜性！")

    log.info(f"\n  排名偏差最大的城市（|ΔRANK| ≥ 3）:")
    big_diff = df[df["abs_rank_diff"] >= 3].sort_values("abs_rank_diff", ascending=False)
    for _, row in big_diff.iterrows():
        direction = "被低估" if row["rank_diff"] > 0 else "被高估"
        log.info(f"    {row['name_en']:12s}: GHI排名#{row['ghi_rank']} → "
                 f"FDSI排名#{row['fdsi_rank']}  (Δ={row['rank_diff']:+d}, "
                 f"按GHI{direction})")

    log.info(f"\n  「高辐照陷阱」城市（GHI前5但FDSI后7）:")
    ghi_top5 = set(df.nsmallest(5, "ghi_rank")["city"])
    fdsi_bottom7 = set(df.nlargest(7, "fdsi_rank")["city"])
    traps = ghi_top5 & fdsi_bottom7
    for city in traps:
        row = df[df["city"] == city].iloc[0]
        log.info(f"    {row['name_en']}: GHI={row['ghi_annual']:.0f} (#{row['ghi_rank']}) "
                 f"但 FDSI={row['fdsi_score']:.3f} (#{row['fdsi_rank']})")

    log.info(f"\n  「隐藏冠军」城市（GHI后7但FDSI前5）:")
    ghi_bottom7 = set(df.nlargest(7, "ghi_rank")["city"])
    fdsi_top5 = set(df.nsmallest(5, "fdsi_rank")["city"])
    champions = ghi_bottom7 & fdsi_top5
    for city in champions:
        row = df[df["city"] == city].iloc[0]
        log.info(f"    {row['name_en']}: GHI={row['ghi_annual']:.0f} (#{row['ghi_rank']}) "
                 f"但 FDSI={row['fdsi_score']:.3f} (#{row['fdsi_rank']})")

    log.info(f"\n  完整排名对比表:")
    display_cols = ["name_en", "ghi_annual", "ghi_rank", "fdsi_score", "fdsi_rank", "rank_diff"]
    log.info(f"\n{df[display_cols].to_string(index=False)}")

    df.to_csv(OUTPUT_DIR / "rank_comparison.csv", index=False, encoding="utf-8-sig")

    return {
        "spearman_rs": round(r_s, 4),
        "spearman_p": round(p_value, 4),
        "kendall_tau": round(tau, 4),
        "kendall_p": round(p_tau, 4),
        "pearson_r": round(r_p, 4),
        "pearson_p": round(p_pearson, 4),
        "mean_abs_rank_diff": round(df["abs_rank_diff"].mean(), 2),
        "max_abs_rank_diff": int(df["abs_rank_diff"].max()),
        "n_big_diff": int((df["abs_rank_diff"] >= 3).sum()),
    }


def diagnose_d5_independence(matrix, indicators):
    log.info("\n" + "=" * 60)
    log.info("诊断 2: D5 维度独立性检验")
    log.info("=" * 60)

    d_scores = pd.DataFrame({
        "city": matrix["city"].values if "city" in matrix.columns else matrix["city_cn"].values,
        "D1": matrix["D1_score"].values,
        "D2": matrix["D2_score"].values,
        "D3": matrix["D3_score"].values,
        "D4": matrix["D4_score"].values,
        "D5": matrix["D5_score"].values,
    })

    log.info(f"\n  D5 vs 其他维度的 Pearson 相关:")
    results = {}
    for dim in ["D1", "D2", "D3", "D4"]:
        r, p = stats.pearsonr(d_scores["D5"], d_scores[dim])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        log.info(f"    D5 vs {dim}: r={r:.4f}  p={p:.4f}  {sig}")
        results[f"D5_vs_{dim}_r"] = round(r, 4)
        results[f"D5_vs_{dim}_p"] = round(p, 4)

    from numpy.linalg import lstsq
    X = np.column_stack([d_scores["D1"].values, d_scores["D4"].values, np.ones(len(d_scores))])
    y = d_scores["D5"].values
    beta, residuals, _, _ = lstsq(X, y, rcond=None)
    y_pred = X @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    log.info(f"\n  D5 = f(D1, D4) 线性回归:")
    log.info(f"    D5 = {beta[0]:.3f}×D1 + {beta[1]:.3f}×D4 + {beta[2]:.3f}")
    log.info(f"    R² = {r_squared:.4f}")
    results["D5_from_D1D4_R2"] = round(r_squared, 4)

    if r_squared > 0.8:
        log.info(f"\n  ⚠ 严重问题: D5 可被 D1+D4 解释 {r_squared:.1%}")
        log.info(f"    D5 维度退化为 D1 和 D4 的影子，缺乏独立信息")
        log.info(f"    原因: MC_PARAMS 是全局常量，所有城市用相同的不确定性分布")
        log.info(f"    修复: Phase 2a 引入城市特异性 MC 参数")
    elif r_squared > 0.5:
        log.info(f"\n  ⚠ 中等问题: D5 部分依赖 D1+D4 (R²={r_squared:.2f})")
    else:
        log.info(f"\n  ✓ D5 相对独立 (R²={r_squared:.2f})")

    log.info(f"\n  D5 子指标跨城市变异:")
    d5_cols = {
        "d5_1_yield_cv": "发电量CV",
        "d5_2_pbt_ci95_width": "PBT CI宽度",
        "d5_3_prob_pbt_le_15yr": "P(PBT≤15yr)",
    }
    for col, label in d5_cols.items():
        if col in indicators.columns:
            vals = indicators[col].dropna()
            cv = vals.std() / vals.mean() if vals.mean() > 0 else 0
            unique_pct = vals.nunique() / len(vals) * 100
            log.info(f"    {label:15s}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
                     f"CV={cv:.4f}, 唯一值比例={unique_pct:.0f}%")
            if cv < 0.01:
                log.info(f"      → ⚠ CV<1%，该子指标几乎无区分度！")

    corr = d_scores[["D1", "D2", "D3", "D4", "D5"]].corr()
    log.info(f"\n  五维度相关矩阵:")
    log.info(f"\n{corr.round(3).to_string()}")

    d_scores.to_csv(OUTPUT_DIR / "d5_independence.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(OUTPUT_DIR / "dimension_correlation_matrix.csv", encoding="utf-8-sig")

    return results


def diagnose_data_quality(indicators, morph):
    log.info("\n" + "=" * 60)
    log.info("诊断 3: 数据质量标记")
    log.info("=" * 60)

    flags = []
    for _, row in morph.iterrows():
        city = row["city"]
        name = row["name_en"]
        f = {"city": city, "name_en": name, "flags": [], "severity": "OK"}

        n = row["n_buildings"]
        if n < 500:
            f["flags"].append(f"建筑数过少({n})")
            f["severity"] = "CRITICAL"
        elif n < 1000:
            f["flags"].append(f"建筑数偏少({n})")
            if f["severity"] != "CRITICAL":
                f["severity"] = "WARNING"

        h = row["d2_1_height_mean"]
        if h < 5:
            f["flags"].append(f"均高异常低({h:.1f}m)，可能抓错区域")
            f["severity"] = "CRITICAL"
        elif h > 80:
            f["flags"].append(f"均高异常高({h:.1f}m)，可能数据错误")
            f["severity"] = "CRITICAL"

        area = row["study_area_km2"]
        if area < 20:
            f["flags"].append(f"研究区偏小({area:.1f}km²)")
            if f["severity"] != "CRITICAL":
                f["severity"] = "WARNING"

        density = row["d2_2_building_density"]
        if density < 0.005:
            f["flags"].append(f"建筑密度极低({density:.4f})，可能是郊区")
            if f["severity"] != "CRITICAL":
                f["severity"] = "WARNING"

        far = row["d2_5_far"]
        if far > 1.0:
            f["flags"].append(f"FAR极高({far:.2f})，导致D2归一化为0")
            if f["severity"] != "CRITICAL":
                f["severity"] = "WARNING"

        if row.get("dominant_pct") == 0.5 and row.get("city_typology") == "mixed":
            f["flags"].append("形态分类为mixed(50/50)，代表性不足")

        if not f["flags"]:
            f["flags"] = ["无异常"]
        flags.append(f)

    for f in flags:
        icon = "✗" if f["severity"] == "CRITICAL" else "⚠" if f["severity"] == "WARNING" else "✓"
        log.info(f"  {icon} {f['name_en']:12s} [{f['severity']:8s}]: {'; '.join(f['flags'])}")

    n_critical = sum(1 for f in flags if f["severity"] == "CRITICAL")
    n_warning = sum(1 for f in flags if f["severity"] == "WARNING")
    log.info(f"\n  汇总: {n_critical} CRITICAL, {n_warning} WARNING, "
             f"{len(flags) - n_critical - n_warning} OK")

    flags_df = pd.DataFrame([
        {"city": f["city"], "name_en": f["name_en"],
         "severity": f["severity"], "flags": "; ".join(f["flags"])}
        for f in flags
    ])
    flags_df.to_csv(OUTPUT_DIR / "data_quality_flags.csv", index=False, encoding="utf-8-sig")
    return flags


def print_summary(ghi_results, d5_results, quality_flags):
    log.info("\n" + "=" * 60)
    log.info("NC 升级诊断汇总")
    log.info("=" * 60)

    log.info(f"\n  [核心发现] GHI排名 vs FDSI排名:")
    log.info(f"    Spearman r_s = {ghi_results['spearman_rs']:.3f} (p={ghi_results['spearman_p']:.4f})")
    log.info(f"    平均排名偏差 = {ghi_results['mean_abs_rank_diff']:.1f} 位")
    log.info(f"    最大排名偏差 = {ghi_results['max_abs_rank_diff']} 位")
    if ghi_results["spearman_rs"] < 0.7:
        log.info(f"    → 强证据: BIPV适宜性 ≠ 太阳能资源丰裕度 (r_s < 0.7)")
    elif ghi_results["spearman_rs"] < 0.85:
        log.info(f"    → 中等证据: 存在系统性偏差但有正相关")
    else:
        log.info(f"    → 弱证据: 排名偏差不够大")

    log.info(f"\n  [方法论风险] D5 独立性:")
    r2 = d5_results["D5_from_D1D4_R2"]
    log.info(f"    D5 = f(D1,D4) 的 R² = {r2:.3f}")
    if r2 > 0.8:
        log.info(f"    → 必须修复: D5 退化为 D1+D4 的影子，NC 审稿人会质疑")
    elif r2 > 0.5:
        log.info(f"    → 建议修复: D5 部分依赖 D1+D4")
    else:
        log.info(f"    → 可以接受")

    log.info(f"\n  [数据质量] 需要修复的城市:")
    critical = [f for f in quality_flags if f["severity"] == "CRITICAL"]
    for f in critical:
        log.info(f"    ✗ {f['name_en']}: {'; '.join(f['flags'])}")
    if not critical:
        log.info(f"    （无 CRITICAL 级别问题）")

    log.info(f"\n  [建议的下一步]:")
    if critical:
        log.info(f"    1. 修复 CRITICAL 城市数据")
    if r2 > 0.5:
        log.info(f"    2. 改进 D5: 引入城市特异性 MC 参数")
    log.info(f"    3. 政策情景分析（可与上述并行）")
    log.info(f"    4. 扩展到 25 城市")


def main():
    log.info("=" * 60)
    log.info("NC Phase 1b: 15城数据诊断分析")
    log.info("=" * 60)

    log.info("\n[0] 加载数据...")
    fdsi, matrix, indicators, morph = load_data()

    ghi_results = diagnose_ghi_vs_fdsi(indicators, fdsi)
    d5_results = diagnose_d5_independence(matrix, indicators)
    quality_flags = diagnose_data_quality(indicators, morph)

    print_summary(ghi_results, d5_results, quality_flags)

    summary = {**ghi_results, **d5_results}
    pd.DataFrame([summary]).to_csv(
        OUTPUT_DIR / "diagnostic_report.csv",
        index=False, encoding="utf-8-sig"
    )

    log.info(f"\n  输出目录: {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        log.info(f"    {f.name}")
    log.info("\n完成。")


if __name__ == "__main__":
    main()
