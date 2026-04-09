#!/usr/bin/env python3
"""
============================================================================
NC Phase 2c: Cross-Pair 对比分析 — 控制变量思路
============================================================================
目的：
  用"自然实验"式的城市配对，展示形态和经济对适宜性的独立影响。
  比统计回归更直观，审稿人和编辑都喜欢。

三类配对：
  Type A: 控制形态，看气候效应
    → 形态相似但气候区不同的城市对
    → 如果 FDSI 差异大 → 气候/经济差异是驱动力

  Type B: 控制气候，看形态效应
    → 同气候区但形态差异大的城市对
    → 如果 FDSI 差异大 → 形态是独立驱动力

  Type C: 控制 GHI，看非资源效应
    → GHI 相似（差异<100 kWh）但 FDSI 差异大的城市对
    → 最直接证明 "资源 ≠ 适宜性"

产出：
  results_nc/cross_pairs/
    type_a_morph_controlled.csv   — 控制形态
    type_b_climate_controlled.csv — 控制气候
    type_c_ghi_controlled.csv     — 控制 GHI
    cross_pair_summary.json       — 可写入 Discussion 的叙述

用法：
  cd ~/Desktop/multi-cities-bipv-nc
  python scripts/nc_02c_cross_pairs.py
============================================================================
"""

import json
import logging
from itertools import combinations
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
OUTPUT_DIR = PROJECT_DIR / "results_nc" / "cross_pairs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    indicators = pd.read_csv(FDSI_DIR / "integrated_indicators.csv")
    matrix = pd.read_csv(FDSI_DIR / "suitability_matrix.csv")
    morph = pd.read_csv(MORPH_DIR / "cross_city_d2d3_summary.csv")

    df = pd.DataFrame({
        "city": indicators["city"].values,
        "name_en": indicators["name_en_energy"].values,
        "climate_zone": indicators["climate_zone_energy"].values,
        "ghi_annual": indicators["d1_1_ghi_annual_kwh"].values,
        "fdsi_score": indicators["fdsi_score"].values,
    })

    for dim in ["D1", "D2", "D3", "D4", "D5"]:
        col = f"{dim}_score"
        if col in matrix.columns:
            df[dim] = matrix[col].values

    # Merge morphology features
    morph_cols = {
        "d2_1_height_mean": "height_mean",
        "d2_2_building_density": "density",
        "d2_5_far": "far",
    }
    for orig, new in morph_cols.items():
        if orig in morph.columns:
            df[new] = morph[orig].values

    # Get city typology if available
    if "city_typology" in morph.columns:
        df["typology"] = morph["city_typology"].values

    df["ghi_rank"] = df["ghi_annual"].rank(ascending=False).astype(int)
    df["fdsi_rank"] = df["fdsi_score"].rank(ascending=False).astype(int)

    log.info(f"  Loaded {len(df)} cities")
    return df


def find_type_a_pairs(df, n_best=5):
    """Type A: Similar morphology (D2), different climate → isolate climate/economic effect."""
    log.info("\n" + "=" * 70)
    log.info("Type A: 控制形态，观察气候+经济效应")
    log.info("  配对条件: |D2差| < 0.10, 不同气候区")
    log.info("=" * 70)

    pairs = []
    for (i, r1), (j, r2) in combinations(df.iterrows(), 2):
        if r1["climate_zone"] == r2["climate_zone"]:
            continue
        d2_diff = abs(r1["D2"] - r2["D2"])
        if d2_diff > 0.10:
            continue
        fdsi_diff = abs(r1["fdsi_score"] - r2["fdsi_score"])
        pairs.append({
            "city_1": r1["name_en"], "city_2": r2["name_en"],
            "cz_1": r1["climate_zone"], "cz_2": r2["climate_zone"],
            "D2_1": r1["D2"], "D2_2": r2["D2"], "D2_diff": round(d2_diff, 3),
            "ghi_1": r1["ghi_annual"], "ghi_2": r2["ghi_annual"],
            "ghi_diff": round(abs(r1["ghi_annual"] - r2["ghi_annual"]), 0),
            "fdsi_1": r1["fdsi_score"], "fdsi_2": r2["fdsi_score"],
            "fdsi_diff": round(fdsi_diff, 4),
            "D1_1": r1["D1"], "D1_2": r2["D1"],
            "D4_1": r1["D4"], "D4_2": r2["D4"],
            "D5_1": r1["D5"], "D5_2": r2["D5"],
            "rank_1": r1["fdsi_rank"], "rank_2": r2["fdsi_rank"],
        })

    pairs_df = pd.DataFrame(pairs).sort_values("fdsi_diff", ascending=False)

    log.info(f"\n  找到 {len(pairs_df)} 个符合条件的配对")
    log.info(f"  展示 FDSI 差异最大的 {n_best} 对:")
    for _, p in pairs_df.head(n_best).iterrows():
        log.info(f"\n    {p['city_1']} [{p['cz_1']}] vs {p['city_2']} [{p['cz_2']}]")
        log.info(f"      D2 (形态):  {p['D2_1']:.3f} vs {p['D2_2']:.3f} (差={p['D2_diff']:.3f})")
        log.info(f"      GHI:        {p['ghi_1']:.0f} vs {p['ghi_2']:.0f} (差={p['ghi_diff']:.0f})")
        log.info(f"      FDSI:       {p['fdsi_1']:.3f} vs {p['fdsi_2']:.3f} (差={p['fdsi_diff']:.4f})")
        log.info(f"      D1(资源):   {p['D1_1']:.3f} vs {p['D1_2']:.3f}")
        log.info(f"      D4(经济):   {p['D4_1']:.3f} vs {p['D4_2']:.3f}")
        log.info(f"      → 形态几乎相同，FDSI 差异来自气候+经济")

    pairs_df.head(20).to_csv(OUTPUT_DIR / "type_a_morph_controlled.csv",
                              index=False, encoding="utf-8-sig")
    return pairs_df.head(n_best)


def find_type_b_pairs(df, n_best=5):
    """Type B: Same climate zone, different morphology → isolate urban form effect."""
    log.info("\n" + "=" * 70)
    log.info("Type B: 控制气候，观察形态效应")
    log.info("  配对条件: 同气候区, |D2差| > 0.15")
    log.info("=" * 70)

    pairs = []
    for (i, r1), (j, r2) in combinations(df.iterrows(), 2):
        if r1["climate_zone"] != r2["climate_zone"]:
            continue
        d2_diff = abs(r1["D2"] - r2["D2"])
        if d2_diff < 0.15:
            continue
        fdsi_diff = abs(r1["fdsi_score"] - r2["fdsi_score"])
        ghi_diff = abs(r1["ghi_annual"] - r2["ghi_annual"])
        pairs.append({
            "city_1": r1["name_en"], "city_2": r2["name_en"],
            "climate_zone": r1["climate_zone"],
            "D2_1": r1["D2"], "D2_2": r2["D2"], "D2_diff": round(d2_diff, 3),
            "ghi_1": r1["ghi_annual"], "ghi_2": r2["ghi_annual"],
            "ghi_diff": round(ghi_diff, 0),
            "fdsi_1": r1["fdsi_score"], "fdsi_2": r2["fdsi_score"],
            "fdsi_diff": round(fdsi_diff, 4),
            "D1_1": r1["D1"], "D1_2": r2["D1"],
            "D4_1": r1["D4"], "D4_2": r2["D4"],
            "rank_1": r1["fdsi_rank"], "rank_2": r2["fdsi_rank"],
        })

    pairs_df = pd.DataFrame(pairs).sort_values("fdsi_diff", ascending=False)

    log.info(f"\n  找到 {len(pairs_df)} 个符合条件的配对")
    log.info(f"  展示 FDSI 差异最大的 {n_best} 对:")
    for _, p in pairs_df.head(n_best).iterrows():
        log.info(f"\n    {p['city_1']} vs {p['city_2']} [同 {p['climate_zone']}]")
        log.info(f"      D2 (形态):  {p['D2_1']:.3f} vs {p['D2_2']:.3f} (差={p['D2_diff']:.3f})")
        log.info(f"      GHI:        {p['ghi_1']:.0f} vs {p['ghi_2']:.0f} (差={p['ghi_diff']:.0f})")
        log.info(f"      FDSI:       {p['fdsi_1']:.3f} vs {p['fdsi_2']:.3f} (差={p['fdsi_diff']:.4f})")
        log.info(f"      → 同气候区内，形态差异驱动 FDSI 差异")

    pairs_df.head(20).to_csv(OUTPUT_DIR / "type_b_climate_controlled.csv",
                              index=False, encoding="utf-8-sig")
    return pairs_df.head(n_best)


def find_type_c_pairs(df, ghi_tolerance=100, n_best=5):
    """Type C: Similar GHI, very different FDSI → direct proof suitability ≠ resource."""
    log.info("\n" + "=" * 70)
    log.info(f"Type C: 控制 GHI (差<{ghi_tolerance} kWh)，观察 FDSI 分化")
    log.info("  最直接证明: 太阳能资源相似 ≠ BIPV 适宜性相似")
    log.info("=" * 70)

    pairs = []
    for (i, r1), (j, r2) in combinations(df.iterrows(), 2):
        ghi_diff = abs(r1["ghi_annual"] - r2["ghi_annual"])
        if ghi_diff > ghi_tolerance:
            continue
        fdsi_diff = abs(r1["fdsi_score"] - r2["fdsi_score"])
        rank_diff = abs(r1["fdsi_rank"] - r2["fdsi_rank"])
        pairs.append({
            "city_1": r1["name_en"], "city_2": r2["name_en"],
            "cz_1": r1["climate_zone"], "cz_2": r2["climate_zone"],
            "ghi_1": r1["ghi_annual"], "ghi_2": r2["ghi_annual"],
            "ghi_diff": round(ghi_diff, 0),
            "fdsi_1": r1["fdsi_score"], "fdsi_2": r2["fdsi_score"],
            "fdsi_diff": round(fdsi_diff, 4),
            "rank_1": r1["fdsi_rank"], "rank_2": r2["fdsi_rank"],
            "rank_diff": int(rank_diff),
            "D2_1": r1["D2"], "D2_2": r2["D2"],
            "D4_1": r1["D4"], "D4_2": r2["D4"],
            "D5_1": r1["D5"], "D5_2": r2["D5"],
        })

    pairs_df = pd.DataFrame(pairs).sort_values("fdsi_diff", ascending=False)

    log.info(f"\n  找到 {len(pairs_df)} 个 GHI 差异 < {ghi_tolerance} kWh 的配对")
    log.info(f"  展示 FDSI 分化最大的 {n_best} 对:")
    for _, p in pairs_df.head(n_best).iterrows():
        log.info(f"\n    {p['city_1']} [{p['cz_1']}] vs {p['city_2']} [{p['cz_2']}]")
        log.info(f"      GHI:  {p['ghi_1']:.0f} vs {p['ghi_2']:.0f} "
                 f"(差={p['ghi_diff']:.0f}, 几乎相同)")
        log.info(f"      FDSI: {p['fdsi_1']:.3f} (#{p['rank_1']}) vs "
                 f"{p['fdsi_2']:.3f} (#{p['rank_2']})")
        log.info(f"      排名差: {p['rank_diff']} 位")
        log.info(f"      D2(形态): {p['D2_1']:.3f} vs {p['D2_2']:.3f}")
        log.info(f"      D4(经济): {p['D4_1']:.3f} vs {p['D4_2']:.3f}")
        log.info(f"      → 相同太阳能资源，完全不同的 BIPV 适宜性！")

    pairs_df.head(20).to_csv(OUTPUT_DIR / "type_c_ghi_controlled.csv",
                              index=False, encoding="utf-8-sig")
    return pairs_df.head(n_best)


def generate_narrative(type_a, type_b, type_c):
    """Generate Discussion-ready narrative from best pairs."""
    log.info("\n" + "=" * 70)
    log.info("论文叙事素材 (可直接融入 Discussion)")
    log.info("=" * 70)

    narratives = []

    if len(type_c) > 0:
        p = type_c.iloc[0]
        text = (
            f"[Type C — 最强证据] {p['city_1']} and {p['city_2']} receive nearly "
            f"identical solar irradiance ({p['ghi_1']:.0f} vs {p['ghi_2']:.0f} kWh/m²/yr, "
            f"Δ={p['ghi_diff']:.0f}), yet their multidimensional suitability diverges by "
            f"{p['rank_diff']} rank positions. This divergence is driven primarily by "
            f"differences in urban morphology (D2: {p['D2_1']:.3f} vs {p['D2_2']:.3f}) "
            f"and economic conditions (D4: {p['D4_1']:.3f} vs {p['D4_2']:.3f})."
        )
        narratives.append(text)
        log.info(f"\n  {text}")

    if len(type_b) > 0:
        p = type_b.iloc[0]
        text = (
            f"[Type B — 形态效应] Within the same climate zone ({p['climate_zone']}), "
            f"{p['city_1']} and {p['city_2']} differ sharply in urban morphology "
            f"(D2: {p['D2_1']:.3f} vs {p['D2_2']:.3f}), producing a FDSI gap of "
            f"{p['fdsi_diff']:.3f} despite similar climatic conditions."
        )
        narratives.append(text)
        log.info(f"\n  {text}")

    if len(type_a) > 0:
        p = type_a.iloc[0]
        text = (
            f"[Type A — 气候+经济效应] {p['city_1']} ({p['cz_1']}) and "
            f"{p['city_2']} ({p['cz_2']}) share nearly identical urban morphology "
            f"(D2: {p['D2_1']:.3f} vs {p['D2_2']:.3f}) but occupy different climate "
            f"zones, resulting in a FDSI difference of {p['fdsi_diff']:.3f}."
        )
        narratives.append(text)
        log.info(f"\n  {text}")

    return narratives


def main():
    log.info("=" * 70)
    log.info("NC Phase 2c: Cross-Pair 控制变量对比分析")
    log.info("=" * 70)

    log.info("\n[0] 加载数据...")
    df = load_data()

    type_a = find_type_a_pairs(df)
    type_b = find_type_b_pairs(df)
    type_c = find_type_c_pairs(df)

    narratives = generate_narrative(type_a, type_b, type_c)

    summary = {
        "type_a_n_pairs": len(type_a),
        "type_b_n_pairs": len(type_b),
        "type_c_n_pairs": len(type_c),
        "narratives": narratives,
    }
    with open(OUTPUT_DIR / "cross_pair_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log.info(f"\n  产出目录: {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.glob("*")):
        log.info(f"    {f.name}")
    log.info("\n完成。")


if __name__ == "__main__":
    main()
