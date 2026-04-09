#!/usr/bin/env python3
"""
============================================================================
NC Phase 2a: GHI-only 误分类量化分析
============================================================================
核心问题：
  "To what extent does irradiance-only prioritization misclassify
   urban residential rooftop PV suitability, and which non-resource
   dimensions drive that misclassification?"

四项分析（按 GPT 建议 + 补充）：
  1. 三分位误分类矩阵 — 按 GHI 和 FDSI 各分 High/Med/Low，交叉统计
  2. 排名重排量化 — rank shift 分布、≥5/≥10 城市数
  3. 典型案例识别 — "高资源陷阱" & "低资源逆袭" 城市 + 驱动因子分解
  4. 政策误导模拟 — 若按 GHI 前 1/3 优先投放，会错过多少真正 High 城市

产出：
  results_nc/misclassification/
    misclass_confusion_matrix.csv      — 三分位交叉表
    rank_shift_analysis.csv            — 排名偏移全表
    extreme_cases.csv                  — 陷阱 & 逆袭案例 + 维度分解
    policy_misallocation.csv           — 政策误导量化
    misclassification_summary.json     — 可直接写入摘要的关键数字

用法：
  cd ~/Desktop/multi-cities-bipv-nc
  python scripts/nc_02a_misclassification.py
============================================================================
"""

import json
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

# ── paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
FDSI_DIR = PROJECT_DIR / "results" / "fdsi"
OUTPUT_DIR = PROJECT_DIR / "results_nc" / "misclassification"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── suitability thresholds (same as scenario analysis) ─────────────────
HIGH_THRESHOLD = 0.70
LOW_THRESHOLD = 0.45


def load_data():
    """Load FDSI scores and integrated indicators."""
    indicators = pd.read_csv(FDSI_DIR / "integrated_indicators.csv")
    matrix = pd.read_csv(FDSI_DIR / "suitability_matrix.csv")
    log.info(f"  Loaded {len(indicators)} cities")

    df = pd.DataFrame({
        "city": indicators["city"].values,
        "name_en": indicators["name_en_energy"].values,
        "climate_zone": indicators["climate_zone_energy"].values,
        "ghi_annual": indicators["d1_1_ghi_annual_kwh"].values,
        "fdsi_score": indicators["fdsi_score"].values,
    })

    # Merge D1-D5 scores
    for dim in ["D1", "D2", "D3", "D4", "D5"]:
        col = f"{dim}_score"
        if col in matrix.columns:
            df[dim] = matrix[col].values

    # Rankings (1 = best)
    df["ghi_rank"] = df["ghi_annual"].rank(ascending=False).astype(int)
    df["fdsi_rank"] = df["fdsi_score"].rank(ascending=False).astype(int)
    df["rank_shift"] = df["ghi_rank"] - df["fdsi_rank"]  # positive = GHI underestimates
    df["abs_rank_shift"] = df["rank_shift"].abs()

    return df, indicators


def assign_tercile(series, name="var"):
    """Assign High / Medium / Low based on tercile cutoffs."""
    t1 = series.quantile(1 / 3)
    t2 = series.quantile(2 / 3)
    labels = []
    for v in series:
        if v >= t2:
            labels.append("High")
        elif v >= t1:
            labels.append("Medium")
        else:
            labels.append("Low")
    return labels, t1, t2


def assign_absolute(series, high=HIGH_THRESHOLD, low=LOW_THRESHOLD):
    """Assign High / Medium / Low based on absolute thresholds."""
    labels = []
    for v in series:
        if v > high:
            labels.append("High")
        elif v > low:
            labels.append("Medium")
        else:
            labels.append("Low")
    return labels


# ══════════════════════════════════════════════════════════════════════
# Analysis 1: Tercile confusion matrix
# ══════════════════════════════════════════════════════════════════════
def analysis_1_confusion_matrix(df):
    log.info("\n" + "=" * 70)
    log.info("分析 1: GHI vs FDSI 三分位误分类矩阵")
    log.info("=" * 70)

    n = len(df)

    # --- Tercile-based classification ---
    df["ghi_class_tercile"], ghi_t1, ghi_t2 = assign_tercile(df["ghi_annual"], "GHI")
    df["fdsi_class_tercile"], fdsi_t1, fdsi_t2 = assign_tercile(df["fdsi_score"], "FDSI")

    log.info(f"\n  GHI 三分位阈值: Low < {ghi_t1:.0f} < Med < {ghi_t2:.0f} < High kWh/m²/yr")
    log.info(f"  FDSI 三分位阈值: Low < {fdsi_t1:.3f} < Med < {fdsi_t2:.3f} < High")

    # Confusion matrix
    order = ["High", "Medium", "Low"]
    ct_tercile = pd.crosstab(
        df["ghi_class_tercile"], df["fdsi_class_tercile"],
        rownames=["GHI_class"], colnames=["FDSI_class"]
    ).reindex(index=order, columns=order, fill_value=0)

    log.info(f"\n  三分位混淆矩阵:")
    log.info(f"\n{ct_tercile.to_string()}")

    # Misclassification count
    correct = sum(df["ghi_class_tercile"] == df["fdsi_class_tercile"])
    misclass = n - correct
    misclass_pct = misclass / n * 100

    log.info(f"\n  正确分类: {correct}/{n} ({correct / n * 100:.0f}%)")
    log.info(f"  误分类:   {misclass}/{n} ({misclass_pct:.0f}%)")

    # --- Absolute-threshold classification ---
    df["fdsi_class_abs"] = assign_absolute(df["fdsi_score"])
    # For GHI, define analogous absolute thresholds by mapping FDSI thresholds
    # But more useful: classify by GHI tercile, then check FDSI absolute class
    log.info(f"\n  按 FDSI 绝对阈值 (High>{HIGH_THRESHOLD}, Low≤{LOW_THRESHOLD}):")
    for cls in order:
        subset = df[df["fdsi_class_abs"] == cls]
        log.info(f"    {cls}: {len(subset)} 城市")

    # Specific misclassification types
    # Type A: GHI says High, FDSI says Low (worst overestimate)
    type_a = df[(df["ghi_class_tercile"] == "High") & (df["fdsi_class_tercile"] == "Low")]
    # Type B: GHI says Low, FDSI says High (worst underestimate)
    type_b = df[(df["ghi_class_tercile"] == "Low") & (df["fdsi_class_tercile"] == "High")]

    log.info(f"\n  极端误分类:")
    log.info(f"    GHI=High → FDSI=Low (严重高估): {len(type_a)} 城市")
    for _, r in type_a.iterrows():
        log.info(f"      {r['name_en']} (GHI={r['ghi_annual']:.0f}, FDSI={r['fdsi_score']:.3f})")
    log.info(f"    GHI=Low  → FDSI=High (严重低估): {len(type_b)} 城市")
    for _, r in type_b.iterrows():
        log.info(f"      {r['name_en']} (GHI={r['ghi_annual']:.0f}, FDSI={r['fdsi_score']:.3f})")

    ct_tercile.to_csv(OUTPUT_DIR / "misclass_confusion_matrix.csv", encoding="utf-8-sig")

    return {
        "n_cities": n,
        "n_correct_tercile": int(correct),
        "n_misclass_tercile": int(misclass),
        "pct_misclass_tercile": round(misclass_pct, 1),
        "n_ghi_high_fdsi_low": len(type_a),
        "n_ghi_low_fdsi_high": len(type_b),
        "cities_ghi_high_fdsi_low": ", ".join(type_a["name_en"].tolist()),
        "cities_ghi_low_fdsi_high": ", ".join(type_b["name_en"].tolist()),
    }


# ══════════════════════════════════════════════════════════════════════
# Analysis 2: Rank shift distribution
# ══════════════════════════════════════════════════════════════════════
def analysis_2_rank_shift(df):
    log.info("\n" + "=" * 70)
    log.info("分析 2: 排名重排量化")
    log.info("=" * 70)

    n = len(df)
    shifts = df["abs_rank_shift"]

    log.info(f"\n  排名偏移统计:")
    log.info(f"    均值:   {shifts.mean():.1f} 位")
    log.info(f"    中位数: {shifts.median():.1f} 位")
    log.info(f"    最大值: {shifts.max()} 位")
    log.info(f"    标准差: {shifts.std():.1f} 位")

    for threshold in [3, 5, 7, 10]:
        count = (shifts >= threshold).sum()
        pct = count / n * 100
        log.info(f"    |shift| ≥ {threshold:2d}: {count:2d} 城市 ({pct:.0f}%)")

    # Directional analysis
    overest = df[df["rank_shift"] < 0]  # GHI rank better than FDSI rank → overestimated by GHI
    underest = df[df["rank_shift"] > 0]  # GHI rank worse than FDSI rank → underestimated by GHI
    log.info(f"\n  方向性分析:")
    log.info(f"    被 GHI 高估 (GHI排名优于FDSI): {len(overest)} 城市, "
             f"平均偏差 {overest['abs_rank_shift'].mean():.1f} 位")
    log.info(f"    被 GHI 低估 (GHI排名劣于FDSI): {len(underest)} 城市, "
             f"平均偏差 {underest['abs_rank_shift'].mean():.1f} 位")

    # Top 10 biggest shifts
    log.info(f"\n  排名偏移最大的 10 个城市:")
    top10 = df.nlargest(10, "abs_rank_shift")[
        ["name_en", "climate_zone", "ghi_annual", "ghi_rank",
         "fdsi_score", "fdsi_rank", "rank_shift"]
    ]
    for _, r in top10.iterrows():
        direction = "高估" if r["rank_shift"] < 0 else "低估"
        log.info(f"    {r['name_en']:12s} [{r['climate_zone']:12s}]: "
                 f"GHI#{r['ghi_rank']:2d} → FDSI#{r['fdsi_rank']:2d}  "
                 f"(shift={r['rank_shift']:+3d}, 被{direction})")

    # Spearman decomposition
    r_s, p_val = stats.spearmanr(df["ghi_rank"], df["fdsi_rank"])
    r_s_sq = r_s ** 2  # proportion of rank variance explained
    log.info(f"\n  Spearman r_s = {r_s:.4f} (p={p_val:.4f})")
    log.info(f"  r_s² = {r_s_sq:.3f} → GHI排名解释FDSI排名变异的 {r_s_sq * 100:.1f}%")
    log.info(f"  → {(1 - r_s_sq) * 100:.1f}% 的排名变异来自非资源因素 (形态+经济+确定性)")

    # Save full table
    shift_df = df[["city", "name_en", "climate_zone", "ghi_annual", "ghi_rank",
                    "fdsi_score", "fdsi_rank", "rank_shift", "abs_rank_shift"]].copy()
    shift_df = shift_df.sort_values("abs_rank_shift", ascending=False)
    shift_df.to_csv(OUTPUT_DIR / "rank_shift_analysis.csv", index=False, encoding="utf-8-sig")

    return {
        "mean_abs_shift": round(shifts.mean(), 1),
        "median_abs_shift": round(shifts.median(), 1),
        "max_abs_shift": int(shifts.max()),
        "n_shift_ge5": int((shifts >= 5).sum()),
        "n_shift_ge10": int((shifts >= 10).sum()),
        "pct_shift_ge5": round((shifts >= 5).sum() / n * 100, 1),
        "spearman_rs": round(r_s, 4),
        "rank_variance_unexplained_pct": round((1 - r_s_sq) * 100, 1),
        "n_overestimated": len(overest),
        "n_underestimated": len(underest),
    }


# ══════════════════════════════════════════════════════════════════════
# Analysis 3: Extreme cases with dimension decomposition
# ══════════════════════════════════════════════════════════════════════
def analysis_3_extreme_cases(df):
    log.info("\n" + "=" * 70)
    log.info("分析 3: 典型误分类案例 + 维度驱动因子分解")
    log.info("=" * 70)

    dims = ["D1", "D2", "D3", "D4", "D5"]
    has_dims = all(d in df.columns for d in dims)

    # --- "Resource-rich trap" cities ---
    # GHI top tercile but FDSI bottom tercile
    n_tercile = len(df) // 3
    ghi_top = set(df.nsmallest(n_tercile, "ghi_rank")["city"])
    fdsi_bottom = set(df.nlargest(n_tercile, "fdsi_rank")["city"])
    traps = df[df["city"].isin(ghi_top & fdsi_bottom)].sort_values("rank_shift")

    log.info(f"\n  ━━━ 「高辐照陷阱」(Resource-Rich Traps) ━━━")
    log.info(f"  定义: GHI排名前1/3 但 FDSI排名后1/3")
    log.info(f"  发现: {len(traps)} 城市")
    for _, r in traps.iterrows():
        log.info(f"\n    ▸ {r['name_en']} [{r['climate_zone']}]")
        log.info(f"      GHI = {r['ghi_annual']:.0f} kWh/m²/yr (排名 #{r['ghi_rank']})")
        log.info(f"      FDSI = {r['fdsi_score']:.3f} (排名 #{r['fdsi_rank']})")
        log.info(f"      排名偏移 = {r['rank_shift']:+d} (被GHI严重高估)")
        if has_dims:
            log.info(f"      维度得分: D1={r['D1']:.3f} D2={r['D2']:.3f} "
                     f"D3={r['D3']:.3f} D4={r['D4']:.3f} D5={r['D5']:.3f}")
            # Identify which dimensions drag it down
            dim_medians = {d: df[d].median() for d in dims}
            weak_dims = [d for d in dims if r[d] < dim_medians[d] * 0.8]
            if weak_dims:
                log.info(f"      拖累维度: {', '.join(weak_dims)} (低于中位数20%以上)")

    # --- "Hidden champion" cities ---
    ghi_bottom = set(df.nlargest(n_tercile, "ghi_rank")["city"])
    fdsi_top = set(df.nsmallest(n_tercile, "fdsi_rank")["city"])
    champions = df[df["city"].isin(ghi_bottom & fdsi_top)].sort_values("rank_shift", ascending=False)

    log.info(f"\n  ━━━ 「低辐照逆袭」(Hidden Champions) ━━━")
    log.info(f"  定义: GHI排名后1/3 但 FDSI排名前1/3")
    log.info(f"  发现: {len(champions)} 城市")
    for _, r in champions.iterrows():
        log.info(f"\n    ▸ {r['name_en']} [{r['climate_zone']}]")
        log.info(f"      GHI = {r['ghi_annual']:.0f} kWh/m²/yr (排名 #{r['ghi_rank']})")
        log.info(f"      FDSI = {r['fdsi_score']:.3f} (排名 #{r['fdsi_rank']})")
        log.info(f"      排名偏移 = {r['rank_shift']:+d} (被GHI严重低估)")
        if has_dims:
            log.info(f"      维度得分: D1={r['D1']:.3f} D2={r['D2']:.3f} "
                     f"D3={r['D3']:.3f} D4={r['D4']:.3f} D5={r['D5']:.3f}")
            dim_medians = {d: df[d].median() for d in dims}
            strong_dims = [d for d in dims[1:] if r[d] > dim_medians[d] * 1.2]
            if strong_dims:
                log.info(f"      优势维度: {', '.join(strong_dims)} (高于中位数20%以上)")

    # --- "Biggest single shift" cases (for narrative) ---
    log.info(f"\n  ━━━ 排名偏移极端案例 (|shift| ≥ 10) ━━━")
    extreme = df[df["abs_rank_shift"] >= 10].sort_values("abs_rank_shift", ascending=False)
    if len(extreme) == 0:
        extreme = df.nlargest(3, "abs_rank_shift")
        log.info(f"  (无 |shift|≥10 的城市，展示最大3个)")
    for _, r in extreme.iterrows():
        direction = "高估 (高辐照陷阱)" if r["rank_shift"] < 0 else "低估 (隐藏冠军)"
        log.info(f"    {r['name_en']:12s}: shift={r['rank_shift']:+3d} → {direction}")

    # Save extreme cases
    cases = []
    for _, r in traps.iterrows():
        row = {"city": r["city"], "name_en": r["name_en"], "type": "resource_rich_trap",
               "ghi": r["ghi_annual"], "ghi_rank": r["ghi_rank"],
               "fdsi": r["fdsi_score"], "fdsi_rank": r["fdsi_rank"],
               "rank_shift": r["rank_shift"]}
        if has_dims:
            for d in dims:
                row[d] = r[d]
        cases.append(row)
    for _, r in champions.iterrows():
        row = {"city": r["city"], "name_en": r["name_en"], "type": "hidden_champion",
               "ghi": r["ghi_annual"], "ghi_rank": r["ghi_rank"],
               "fdsi": r["fdsi_score"], "fdsi_rank": r["fdsi_rank"],
               "rank_shift": r["rank_shift"]}
        if has_dims:
            for d in dims:
                row[d] = r[d]
        cases.append(row)

    pd.DataFrame(cases).to_csv(OUTPUT_DIR / "extreme_cases.csv", index=False, encoding="utf-8-sig")

    return {
        "n_resource_rich_traps": len(traps),
        "trap_cities": ", ".join(traps["name_en"].tolist()),
        "n_hidden_champions": len(champions),
        "champion_cities": ", ".join(champions["name_en"].tolist()),
    }


# ══════════════════════════════════════════════════════════════════════
# Analysis 4: Policy misallocation simulation
# ══════════════════════════════════════════════════════════════════════
def analysis_4_policy_misallocation(df):
    log.info("\n" + "=" * 70)
    log.info("分析 4: GHI-only 政策优先投放误导量化")
    log.info("=" * 70)

    n = len(df)

    # --- Scenario: Prioritize GHI top-1/3 for BIPV deployment ---
    n_top = n // 3  # ~13 cities
    if n_top == 0:
        n_top = 1

    ghi_priority = set(df.nsmallest(n_top, "ghi_rank")["city"])
    fdsi_priority = set(df.nsmallest(n_top, "fdsi_rank")["city"])

    overlap = ghi_priority & fdsi_priority
    missed_by_ghi = fdsi_priority - ghi_priority  # truly suitable but not selected
    false_positives = ghi_priority - fdsi_priority  # selected by GHI but not truly suitable

    log.info(f"\n  政策模拟: 选前 {n_top} 个城市优先部署 BIPV")
    log.info(f"  ─────────────────────────────────────────")
    log.info(f"  按 GHI 选中:  {sorted(df[df['city'].isin(ghi_priority)]['name_en'].tolist())}")
    log.info(f"  按 FDSI 选中: {sorted(df[df['city'].isin(fdsi_priority)]['name_en'].tolist())}")
    log.info(f"\n  吻合: {len(overlap)}/{n_top} ({len(overlap)/n_top*100:.0f}%)")
    log.info(f"  错选 (GHI选了但FDSI说不该选): {len(false_positives)} 城市")
    for city in false_positives:
        r = df[df["city"] == city].iloc[0]
        log.info(f"    ✗ {r['name_en']}: GHI#{r['ghi_rank']} 但 FDSI#{r['fdsi_rank']}")
    log.info(f"  遗漏 (FDSI说该选但GHI没选): {len(missed_by_ghi)} 城市")
    for city in missed_by_ghi:
        r = df[df["city"] == city].iloc[0]
        log.info(f"    ✗ {r['name_en']}: FDSI#{r['fdsi_rank']} 但 GHI#{r['ghi_rank']}")

    # --- Misallocation rate at different selection sizes ---
    log.info(f"\n  不同选择规模下的误导率:")
    misalloc_data = []
    for frac in [0.25, 0.33, 0.50]:
        k = max(1, int(n * frac))
        ghi_set = set(df.nsmallest(k, "ghi_rank")["city"])
        fdsi_set = set(df.nsmallest(k, "fdsi_rank")["city"])
        overlap_k = len(ghi_set & fdsi_set)
        precision = overlap_k / k * 100
        missed_k = len(fdsi_set - ghi_set)
        log.info(f"    选前 {frac:.0%} ({k}城): 吻合 {overlap_k}/{k} ({precision:.0f}%), "
                 f"遗漏 {missed_k} 个真正高适宜城市")
        misalloc_data.append({
            "selection_fraction": frac,
            "k": k,
            "overlap": overlap_k,
            "precision_pct": round(precision, 1),
            "n_missed": missed_k,
            "n_false_positive": k - overlap_k,
        })

    # --- "Opportunity cost" of misallocation ---
    # Compare mean FDSI of GHI-selected vs FDSI-selected
    ghi_selected_fdsi = df[df["city"].isin(ghi_priority)]["fdsi_score"].mean()
    fdsi_selected_fdsi = df[df["city"].isin(fdsi_priority)]["fdsi_score"].mean()
    opportunity_cost = fdsi_selected_fdsi - ghi_selected_fdsi
    log.info(f"\n  机会成本 (前1/3选择):")
    log.info(f"    按GHI选的{n_top}城平均FDSI: {ghi_selected_fdsi:.4f}")
    log.info(f"    按FDSI选的{n_top}城平均FDSI: {fdsi_selected_fdsi:.4f}")
    log.info(f"    机会损失: {opportunity_cost:.4f} ({opportunity_cost/fdsi_selected_fdsi*100:.1f}%)")

    pd.DataFrame(misalloc_data).to_csv(
        OUTPUT_DIR / "policy_misallocation.csv", index=False, encoding="utf-8-sig"
    )

    return {
        "n_priority": n_top,
        "n_overlap_top3rd": len(overlap),
        "pct_overlap_top3rd": round(len(overlap) / n_top * 100, 1),
        "n_missed_top3rd": len(missed_by_ghi),
        "n_false_positive_top3rd": len(false_positives),
        "missed_cities": ", ".join(
            df[df["city"].isin(missed_by_ghi)]["name_en"].tolist()
        ),
        "false_positive_cities": ", ".join(
            df[df["city"].isin(false_positives)]["name_en"].tolist()
        ),
        "mean_fdsi_ghi_selected": round(ghi_selected_fdsi, 4),
        "mean_fdsi_fdsi_selected": round(fdsi_selected_fdsi, 4),
        "opportunity_cost_pct": round(
            opportunity_cost / fdsi_selected_fdsi * 100, 1
        ),
    }


# ══════════════════════════════════════════════════════════════════════
# Summary: Abstract-ready numbers
# ══════════════════════════════════════════════════════════════════════
def generate_abstract_numbers(r1, r2, r3, r4):
    log.info("\n" + "=" * 70)
    log.info("可直接写入摘要的关键数字")
    log.info("=" * 70)

    sentences = [
        f"Across {r1['n_cities']} Chinese cities spanning five climate zones, "
        f"irradiance-only prioritization misclassifies {r1['pct_misclass_tercile']}% "
        f"of cities into wrong suitability terciles.",

        f"The median rank shift between GHI-based and multidimensional rankings "
        f"is {r2['median_abs_shift']:.0f} positions (max {r2['max_abs_shift']}), "
        f"with {r2['pct_shift_ge5']}% of cities shifting by ≥5 ranks.",

        f"GHI ranking explains only {100 - r2['rank_variance_unexplained_pct']:.0f}% "
        f"of suitability rank variance; the remaining "
        f"{r2['rank_variance_unexplained_pct']:.0f}% is driven by urban morphology, "
        f"economic conditions, and assessment certainty.",

        f"If the top third of cities are selected for priority BIPV deployment "
        f"based on solar resource alone, {r4['n_missed_top3rd']} of {r4['n_priority']} "
        f"truly high-suitability cities would be overlooked, representing a "
        f"{r4['opportunity_cost_pct']}% opportunity cost in expected suitability.",

        f"We identify {r3['n_resource_rich_traps']} 'resource-rich trap' cities "
        f"and {r3['n_hidden_champions']} 'hidden champion' cities where conventional "
        f"irradiance-based assessment produces the most misleading conclusions.",
    ]

    for i, s in enumerate(sentences, 1):
        log.info(f"\n  [{i}] {s}")

    return sentences


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 70)
    log.info("NC Phase 2a: GHI-only 误分类量化分析")
    log.info("核心问题: To what extent does irradiance-only prioritization")
    log.info("          misclassify urban residential rooftop PV suitability?")
    log.info("=" * 70)

    log.info("\n[0] 加载数据...")
    df, indicators = load_data()

    r1 = analysis_1_confusion_matrix(df)
    r2 = analysis_2_rank_shift(df)
    r3 = analysis_3_extreme_cases(df)
    r4 = analysis_4_policy_misallocation(df)

    sentences = generate_abstract_numbers(r1, r2, r3, r4)

    # Save all results
    summary = {**r1, **r2, **r3, **r4, "abstract_sentences": sentences}
    with open(OUTPUT_DIR / "misclassification_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log.info(f"\n\n  产出目录: {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.glob("*")):
        log.info(f"    {f.name}")
    log.info("\n完成。")


if __name__ == "__main__":
    main()
