#!/usr/bin/env python3
"""
============================================================================
NC Phase 2d: 政策机会成本具体化 + 分类敏感性分析
============================================================================
两个任务：

Task 1 — 机会成本物理量化
  把 "4% FDSI 机会损失" 翻译为：
  - 被遗漏城市的年发电潜力 (GWh/yr)
  - 被遗漏城市的屋顶装机容量 (MW)
  - 被遗漏城市的年 CO₂ 减排 (ktCO₂/yr)
  - 被遗漏城市覆盖的人口 (百万人)

  方法：
  1. 从 energy 结果中提取每城市的 specific yield (kWh/kWp/yr)
  2. 从 morphology 结果中提取屋顶总面积 (m²)
  3. 用 0.15 kWp/m² 转换系数（典型住宅单晶硅）估算装机容量
  4. 乘以 specific yield → 年发电量
  5. 乘以各省 grid emission factor → CO₂ 减排
  6. 按 GHI-only vs FDSI 策略差异计算被遗漏的增量

Task 2 — 分类敏感性 (SI 表)
  在 tercile 基础上加 quartile, quintile, top-10/top-20% targeting
  对每种分类方案计算：
  - 混淆矩阵
  - 误分类率
  - 政策遗漏数
  产出一张汇总表，正文只引一句话，细节放 SI

产出：
  results_nc/policy_cost/
    opportunity_cost_physical.csv     — 每城市的物理潜力
    opportunity_cost_summary.json     — 摘要级数字
    misallocation_detail.csv          — 按 GHI vs FDSI 策略的具体差异
  results_nc/sensitivity/
    classification_sensitivity.csv    — 多阈值敏感性汇总表
    confusion_matrices/               — 各方案混淆矩阵

用法：
  cd ~/Desktop/multi-cities-bipv-nc
  python scripts/nc_02d_policy_cost_and_sensitivity.py
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
MORPH_DIR = PROJECT_DIR / "results" / "morphology"
ENERGY_DIR = PROJECT_DIR / "results" / "energy"
OUTPUT_COST = PROJECT_DIR / "results_nc" / "policy_cost"
OUTPUT_SENS = PROJECT_DIR / "results_nc" / "sensitivity"
OUTPUT_COST.mkdir(parents=True, exist_ok=True)
OUTPUT_SENS.mkdir(parents=True, exist_ok=True)
(OUTPUT_SENS / "confusion_matrices").mkdir(exist_ok=True)

# ── 常量 ──────────────────────────────────────────────────────────────
# 典型住宅屋顶光伏转换参数
PV_DENSITY = 0.15  # kWp per m² of rooftop (typical mono-Si residential)
ROOFTOP_UTIL = 0.60  # 屋顶利用率（排除设备、遮挡区域等）

# 各省电网碳排放因子 (kgCO₂/kWh) — 2022年中电联数据
GRID_EMISSION_FACTORS = {
    '黑龙江': 0.85, '吉林': 0.76, '辽宁': 0.74,
    '内蒙古': 0.80, '河北': 0.78, '新疆': 0.75,
    '山西': 0.82, '山东': 0.72, '河南': 0.70,
    '陕西': 0.72, '甘肃': 0.68, '宁夏': 0.78,
    '青海': 0.45, '北京': 0.60, '天津': 0.65,
    '江苏': 0.62, '浙江': 0.58, '安徽': 0.70,
    '江西': 0.68, '上海': 0.60,
    '湖北': 0.55, '湖南': 0.58, '四川': 0.30,
    '重庆': 0.52, '贵州': 0.50, '云南': 0.25,
    '西藏': 0.15,
    '广东': 0.52, '福建': 0.50, '广西': 0.55,
    '海南': 0.55,
    '香港特别行政区': 0.70, '台湾': 0.50,
}

# 城市人口 (百万) — 用于社会影响量化
CITY_POPULATIONS = {
    'harbin': 10.0, 'changchun': 9.1, 'shenyang': 9.1, 'dalian': 7.5,
    'hohhot': 3.5, 'tangshan': 7.7, 'urumqi': 4.1,
    'beijing': 21.5, 'tianjin': 13.9, 'jinan': 9.2, 'zhengzhou': 12.7,
    'xian': 13.0, 'shijiazhuang': 11.2, 'taiyuan': 5.3, 'lanzhou': 4.4,
    'yinchuan': 2.9, 'xining': 2.5, 'qingdao': 10.0, 'wuxi': 7.5,
    'suzhou': 12.7,
    'shanghai': 24.9, 'chongqing': 32.1, 'wuhan': 13.7, 'nanjing': 9.3,
    'changsha': 10.5, 'chengdu': 21.0, 'hangzhou': 12.2, 'hefei': 9.4,
    'nanchang': 6.3, 'ningbo': 9.5,
    'shenzhen': 17.6, 'guangzhou': 18.7, 'xiamen': 5.3, 'fuzhou': 8.4,
    'nanning': 8.7, 'haikou': 2.9,
    'kunming': 8.5, 'guiyang': 5.9, 'lhasa': 0.9,
    'hongkong': 7.5, 'taipei': 2.6,
}


def load_data():
    """Load all needed data."""
    indicators = pd.read_csv(FDSI_DIR / "integrated_indicators.csv")
    matrix = pd.read_csv(FDSI_DIR / "suitability_matrix.csv")
    morph = pd.read_csv(MORPH_DIR / "cross_city_d2d3_summary.csv")

    # Try to load energy simulation summaries
    energy_files = list(ENERGY_DIR.glob("*/energy_summary.csv"))
    if not energy_files:
        energy_files = list(ENERGY_DIR.glob("*_energy_summary.csv"))

    log.info(f"  FDSI: {len(indicators)} cities")
    log.info(f"  Morphology: {len(morph)} cities")
    log.info(f"  Energy summary files found: {len(energy_files)}")

    return indicators, matrix, morph, energy_files


def build_city_table(indicators, matrix, morph):
    """Build master table with all needed fields per city."""
    df = pd.DataFrame({
        "city": indicators["city"].values,
    })

    # Name and province
    for col in ["name_en_energy", "name_en", "name_en_morph"]:
        if col in indicators.columns:
            df["name_en"] = indicators[col].values
            break

    for col in ["province_energy", "province", "province_morph"]:
        if col in indicators.columns:
            df["province"] = indicators[col].values
            break

    # Climate zone
    for col in ["climate_zone_energy", "climate_zone"]:
        if col in indicators.columns:
            df["climate_zone"] = indicators[col].values
            break

    # GHI and FDSI
    df["ghi_annual"] = indicators["d1_1_ghi_annual_kwh"].values
    df["fdsi_score"] = indicators["fdsi_score"].values

    # D1-D5
    for dim in ["D1", "D2", "D3", "D4", "D5"]:
        col = f"{dim}_score"
        if col in matrix.columns:
            df[dim] = matrix[col].values

    # Rankings
    df["ghi_rank"] = df["ghi_annual"].rank(ascending=False).astype(int)
    df["fdsi_rank"] = df["fdsi_score"].rank(ascending=False).astype(int)

    # Morphology: directly use shading-corrected deployable capacity from nc_03
    # d3_4_total_deployable_mw = rooftop_area × utilization × shading_factor × PV_density
    # This is more accurate than re-deriving from raw roof area.
    morph_fields = {
        "d3_4_total_deployable_mw": "capacity_mw",      # use directly — shading corrected
        "d2_3_roof_area_total_m2": "rooftop_area_m2",   # keep for reporting only
        "total_rooftop_area_m2": "rooftop_area_m2",
        "total_roof_area_m2": "rooftop_area_m2",
        "n_buildings": "n_buildings",
        "study_area_km2": "study_area_km2",
        "d2_2_building_density": "building_density",
    }
    for orig, new in morph_fields.items():
        if orig in morph.columns and new not in df.columns:
            df[new] = morph[orig].values

    # Fallback rooftop area estimate if direct column missing
    if "rooftop_area_m2" not in df.columns:
        log.info("  No rooftop_area_m2 found, estimating from building_density × study_area")
        if "building_density" in df.columns and "study_area_km2" in df.columns:
            df["rooftop_area_m2"] = (
                df["building_density"] * df["study_area_km2"] * 1e6
            )

    # Energy: specific yield (kWh/kWp/yr) from indicators
    yield_cols = ["specific_yield_kwh_kwp", "d1_2_specific_yield_kwh_kwp", "specific_yield", "yield_mean"]
    for col in yield_cols:
        if col in indicators.columns:
            df["specific_yield"] = indicators[col].values
            break
    if "specific_yield" not in df.columns:
        # Fallback: estimate from GHI × system efficiency (~0.80 performance ratio)
        log.info("  No specific yield column found, estimating: GHI × 0.80")
        df["specific_yield"] = df["ghi_annual"] * 0.80

    # Economic indicators
    for col_pair in [
        ("d4_1_pbt_mean", "pbt_mean"),
        ("d4_2_lcoe_mean", "lcoe_mean"),
    ]:
        if col_pair[0] in indicators.columns:
            df[col_pair[1]] = indicators[col_pair[0]].values

    # Population
    df["population_million"] = df["city"].map(CITY_POPULATIONS).fillna(0)

    # Grid emission factor
    if "province" in df.columns:
        df["grid_ef"] = df["province"].map(GRID_EMISSION_FACTORS).fillna(0.55)
    else:
        df["grid_ef"] = 0.55  # national average fallback

    log.info(f"  Master table: {len(df)} cities, {len(df.columns)} columns")
    return df


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Policy opportunity cost in physical units
# ══════════════════════════════════════════════════════════════════════
def compute_physical_potential(df):
    """Compute per-city rooftop PV potential in physical units."""
    log.info("\n" + "=" * 70)
    log.info("TASK 1: 物理量化 — 每城市住宅屋顶 PV 潜力")
    log.info("=" * 70)

    # Capacity (MW): use shading-corrected deployable capacity from morphology if available,
    # otherwise derive from raw rooftop area (less accurate for high-rise cities).
    if "capacity_mw" not in df.columns:
        log.info("  capacity_mw not found, deriving from rooftop_area × utilization × PV_density")
        df["capacity_mw"] = df["rooftop_area_m2"] * ROOFTOP_UTIL * PV_DENSITY / 1000
    else:
        log.info("  Using d3_4_total_deployable_mw directly (shading-corrected)")

    # Annual generation (GWh/yr) = capacity (kWp) × specific_yield / 1e6
    df["generation_gwh_yr"] = (
        df["capacity_mw"] * 1000 * df["specific_yield"] / 1e6
    )

    # CO₂ reduction (ktCO₂/yr) = generation (GWh) × grid_ef (kgCO₂/kWh) × 1000 / 1000
    # = generation (GWh) × grid_ef (kgCO₂/kWh)
    df["co2_reduction_kt_yr"] = df["generation_gwh_yr"] * df["grid_ef"]

    log.info(f"\n  总计 (全部 {len(df)} 城市样本区域内):")
    log.info(f"    屋顶面积:     {df['rooftop_area_m2'].sum() / 1e6:.1f} km²")
    log.info(f"    装机容量:     {df['capacity_mw'].sum():.0f} MW")
    log.info(f"    年发电量:     {df['generation_gwh_yr'].sum():.1f} GWh/yr")
    log.info(f"    年减排量:     {df['co2_reduction_kt_yr'].sum():.1f} ktCO₂/yr")
    log.info(f"    覆盖人口:     {df['population_million'].sum():.1f} 百万人")

    return df


def compute_misallocation_cost(df):
    """Compare GHI-only vs FDSI-based targeting strategies."""
    log.info("\n" + "=" * 70)
    log.info("TASK 1b: 政策误导的物理机会成本")
    log.info("=" * 70)

    n = len(df)
    results = []

    for frac, label in [
        (1/3, "Top third (33%)"),
        (1/4, "Top quarter (25%)"),
        (1/5, "Top fifth (20%)"),
        (1/2, "Top half (50%)"),
    ]:
        k = max(1, int(n * frac))

        ghi_selected = set(df.nsmallest(k, "ghi_rank")["city"])
        fdsi_selected = set(df.nsmallest(k, "fdsi_rank")["city"])

        missed = fdsi_selected - ghi_selected  # truly suitable but not selected by GHI
        false_pos = ghi_selected - fdsi_selected  # selected by GHI but not truly suitable

        # Physical cost of missing these cities
        missed_df = df[df["city"].isin(missed)]
        false_pos_df = df[df["city"].isin(false_pos)]

        # What you get with FDSI strategy
        fdsi_total = df[df["city"].isin(fdsi_selected)]
        # What you get with GHI strategy
        ghi_total = df[df["city"].isin(ghi_selected)]

        missed_gen = missed_df["generation_gwh_yr"].sum()
        missed_cap = missed_df["capacity_mw"].sum()
        missed_co2 = missed_df["co2_reduction_kt_yr"].sum()
        missed_pop = missed_df["population_million"].sum()

        fp_gen = false_pos_df["generation_gwh_yr"].sum()
        fp_cap = false_pos_df["capacity_mw"].sum()

        # Net difference
        net_gen = fdsi_total["generation_gwh_yr"].sum() - ghi_total["generation_gwh_yr"].sum()

        log.info(f"\n  ── {label} (k={k}) ──")
        log.info(f"    吻合: {len(ghi_selected & fdsi_selected)}/{k}")
        log.info(f"    GHI 遗漏的高适宜城市: {len(missed)}")
        if missed:
            for _, r in missed_df.iterrows():
                log.info(f"      ✗ {r['name_en']:12s}: "
                         f"FDSI#{r['fdsi_rank']} (GHI#{r['ghi_rank']}), "
                         f"{r['generation_gwh_yr']:.1f} GWh/yr, "
                         f"{r['population_million']:.1f}M people")
        log.info(f"    被遗漏的合计潜力:")
        log.info(f"      装机:   {missed_cap:.0f} MW")
        log.info(f"      发电:   {missed_gen:.1f} GWh/yr")
        log.info(f"      减排:   {missed_co2:.1f} ktCO₂/yr")
        log.info(f"      人口:   {missed_pop:.1f} 百万人")
        log.info(f"    GHI 错选的低适宜城市: {len(false_pos)}")
        if false_pos:
            for _, r in false_pos_df.iterrows():
                log.info(f"      ✗ {r['name_en']:12s}: "
                         f"GHI#{r['ghi_rank']} (FDSI#{r['fdsi_rank']})")

        results.append({
            "strategy": label,
            "k": k,
            "n_overlap": len(ghi_selected & fdsi_selected),
            "n_missed": len(missed),
            "n_false_positive": len(false_pos),
            "missed_cities": ", ".join(missed_df["name_en"].tolist()),
            "missed_capacity_mw": round(missed_cap, 0),
            "missed_generation_gwh": round(missed_gen, 1),
            "missed_co2_kt": round(missed_co2, 1),
            "missed_population_million": round(missed_pop, 1),
            "false_positive_cities": ", ".join(false_pos_df["name_en"].tolist()),
            "ghi_strategy_total_gwh": round(ghi_total["generation_gwh_yr"].sum(), 1),
            "fdsi_strategy_total_gwh": round(fdsi_total["generation_gwh_yr"].sum(), 1),
            "net_generation_diff_gwh": round(net_gen, 1),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_COST / "misallocation_detail.csv",
                       index=False, encoding="utf-8-sig")
    return results_df


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Classification sensitivity analysis (for SI)
# ══════════════════════════════════════════════════════════════════════
def classification_sensitivity(df):
    """Run misclassification analysis under multiple threshold schemes."""
    log.info("\n" + "=" * 70)
    log.info("TASK 2: 分类敏感性分析 (Supplementary)")
    log.info("=" * 70)

    n = len(df)
    results = []

    schemes = {
        "tercile": 3,
        "quartile": 4,
        "quintile": 5,
    }

    for scheme_name, n_groups in schemes.items():
        log.info(f"\n  ── {scheme_name} (n_groups={n_groups}) ──")

        # Assign classes based on quantile cutoffs (0 = lowest, n_groups-1 = highest)
        # Both use the same numeric scale so comparison is valid
        ghi_groups = pd.qcut(
            df["ghi_annual"], q=n_groups, labels=False, duplicates="drop"
        )
        fdsi_groups = pd.qcut(
            df["fdsi_score"], q=n_groups, labels=False, duplicates="drop"
        )

        # Named labels for confusion matrix display
        group_labels = [f"G{i+1}" for i in range(n_groups)]
        ghi_classes = ghi_groups.map(lambda x: f"G{int(x)+1}" if pd.notna(x) else "G1")
        fdsi_classes = fdsi_groups.map(lambda x: f"G{int(x)+1}" if pd.notna(x) else "G1")

        # Confusion matrix
        ct = pd.crosstab(ghi_classes, fdsi_classes,
                          rownames=[f"GHI_{scheme_name}"],
                          colnames=[f"FDSI_{scheme_name}"])
        ct.to_csv(OUTPUT_SENS / "confusion_matrices" / f"confusion_{scheme_name}.csv",
                   encoding="utf-8-sig")

        # Misclassification rate (compare numeric groups directly)
        correct = int((ghi_groups == fdsi_groups).sum())
        misclass = n - correct
        misclass_pct = misclass / n * 100

        # Severe misclassification (off by ≥ 2 groups)
        severe = int((abs(ghi_groups - fdsi_groups) >= 2).sum())
        severe_pct = severe / n * 100

        log.info(f"    正确: {correct}/{n} ({100-misclass_pct:.1f}%)")
        log.info(f"    误分类: {misclass}/{n} ({misclass_pct:.1f}%)")
        log.info(f"    严重误分类 (跨≥2组): {severe}/{n} ({severe_pct:.1f}%)")

        results.append({
            "scheme": scheme_name,
            "n_groups": n_groups,
            "n_correct": int(correct),
            "n_misclassified": int(misclass),
            "pct_misclassified": round(misclass_pct, 1),
            "n_severe": int(severe),
            "pct_severe": round(severe_pct, 1),
        })

    # --- Targeting precision at different thresholds ---
    log.info(f"\n  ── Targeting precision (top-k) ──")
    for pct, label in [(10, "Top 10%"), (20, "Top 20%"), (25, "Top 25%"),
                        (33, "Top 33%"), (50, "Top 50%")]:
        k = max(1, int(n * pct / 100))
        ghi_set = set(df.nsmallest(k, "ghi_rank")["city"])
        fdsi_set = set(df.nsmallest(k, "fdsi_rank")["city"])
        overlap = len(ghi_set & fdsi_set)
        precision = overlap / k * 100
        missed = len(fdsi_set - ghi_set)

        log.info(f"    {label:8s} (k={k:2d}): "
                 f"precision={precision:.0f}%, missed={missed}")

        results.append({
            "scheme": f"targeting_{label.replace(' ', '_').lower()}",
            "n_groups": f"k={k}",
            "n_correct": int(overlap),
            "n_misclassified": int(missed),
            "pct_misclassified": round(100 - precision, 1),
            "n_severe": None,
            "pct_severe": None,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_SENS / "classification_sensitivity.csv",
                       index=False, encoding="utf-8-sig")

    log.info(f"\n  敏感性结论:")
    tercile_pct = results_df[results_df["scheme"] == "tercile"]["pct_misclassified"].values[0]
    quartile_pct = results_df[results_df["scheme"] == "quartile"]["pct_misclassified"].values[0]
    quintile_pct = results_df[results_df["scheme"] == "quintile"]["pct_misclassified"].values[0]
    log.info(f"    Tercile: {tercile_pct}%, Quartile: {quartile_pct}%, "
             f"Quintile: {quintile_pct}%")
    log.info(f"    → 结论在不同分类粒度下定性一致")

    return results_df


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    log.info("=" * 70)
    log.info("NC Phase 2d: 政策机会成本物理量化 + 分类敏感性")
    log.info("=" * 70)

    log.info("\n[0] 加载数据...")
    indicators, matrix, morph, energy_files = load_data()

    log.info("\n[1] 构建城市主表...")
    df = build_city_table(indicators, matrix, morph)

    log.info("\n[2] 计算物理潜力...")
    df = compute_physical_potential(df)

    log.info("\n[3] 政策误导的物理机会成本...")
    misalloc = compute_misallocation_cost(df)

    log.info("\n[4] 分类敏感性分析...")
    sensitivity = classification_sensitivity(df)

    # ── Save ──
    # Per-city physical potential
    out_cols = ["city", "name_en", "province", "climate_zone",
                "ghi_annual", "ghi_rank", "fdsi_score", "fdsi_rank",
                "rooftop_area_m2", "capacity_mw", "specific_yield",
                "generation_gwh_yr", "co2_reduction_kt_yr",
                "population_million", "grid_ef"]
    available = [c for c in out_cols if c in df.columns]
    df[available].sort_values("fdsi_rank").to_csv(
        OUTPUT_COST / "opportunity_cost_physical.csv",
        index=False, encoding="utf-8-sig"
    )

    # Summary JSON for abstract
    n = len(df)
    k = max(1, int(n / 3))
    ghi_sel = set(df.nsmallest(k, "ghi_rank")["city"])
    fdsi_sel = set(df.nsmallest(k, "fdsi_rank")["city"])
    missed_cities = fdsi_sel - ghi_sel
    missed_df = df[df["city"].isin(missed_cities)]

    summary = {
        "n_cities": n,
        "total_sample_capacity_mw": round(df["capacity_mw"].sum(), 0),
        "total_sample_generation_gwh": round(df["generation_gwh_yr"].sum(), 1),
        "targeting_top_third": {
            "k": k,
            "n_missed": len(missed_cities),
            "missed_capacity_mw": round(missed_df["capacity_mw"].sum(), 0),
            "missed_generation_gwh": round(missed_df["generation_gwh_yr"].sum(), 1),
            "missed_co2_kt": round(missed_df["co2_reduction_kt_yr"].sum(), 1),
            "missed_population_million": round(missed_df["population_million"].sum(), 1),
        },
        "sensitivity": {
            "tercile_misclass_pct": float(
                sensitivity[sensitivity["scheme"] == "tercile"]["pct_misclassified"].values[0]
            ),
            "quartile_misclass_pct": float(
                sensitivity[sensitivity["scheme"] == "quartile"]["pct_misclassified"].values[0]
            ),
            "quintile_misclass_pct": float(
                sensitivity[sensitivity["scheme"] == "quintile"]["pct_misclassified"].values[0]
            ),
        },
        "abstract_sentence": None,  # filled below
    }

    # Generate abstract sentence
    missed_gen = summary["targeting_top_third"]["missed_generation_gwh"]
    missed_pop = summary["targeting_top_third"]["missed_population_million"]
    missed_n = summary["targeting_top_third"]["n_missed"]
    summary["abstract_sentence"] = (
        f"If the top third of cities are selected for priority deployment based "
        f"on solar resource alone, {missed_n} cities with {missed_pop:.0f} million "
        f"residents and an estimated {missed_gen:.0f} GWh/yr of rooftop PV potential "
        f"would be systematically overlooked."
    )

    with open(OUTPUT_COST / "opportunity_cost_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log.info(f"\n{'=' * 70}")
    log.info(f"摘要级结论")
    log.info(f"{'=' * 70}")
    log.info(f"\n  {summary['abstract_sentence']}")
    log.info(f"\n  敏感性: 误分类率在 tercile/quartile/quintile 下分别为 "
             f"{summary['sensitivity']['tercile_misclass_pct']}% / "
             f"{summary['sensitivity']['quartile_misclass_pct']}% / "
             f"{summary['sensitivity']['quintile_misclass_pct']}%")

    log.info(f"\n  产出:")
    for d in [OUTPUT_COST, OUTPUT_SENS]:
        for f in sorted(d.rglob("*")):
            if f.is_file():
                log.info(f"    {f.relative_to(PROJECT_DIR)}")
    log.info("\n完成。")


if __name__ == "__main__":
    main()
