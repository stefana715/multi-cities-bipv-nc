#!/usr/bin/env python3
"""
============================================================================
NC Phase 1d: 修复贵阳 — 优化 bbox + 面积推断高度 + 重跑 morphology
============================================================================
问题根因：
  贵阳 OSM 数据覆盖率在中国主要城市中属于最低水平：
  - 原始 bbox（542km²）仅含 2,897 栋建筑，密度 5/km²
  - 住宅筛选后仅 495 栋（CRITICAL 阈值）
  - 中位高 3m → 大量农村/工业建筑混入，非城市住宅区

诊断结论：
  bbox 位置问题 + OSM 全市覆盖率偏低（双重问题）
  三个测试 bbox 中 old_city_wide 建筑数最多（831栋/69km²）

修复方案：
  1. 换用 old_city_wide bbox（26.63, 26.55, 106.75, 106.67）
  2. 纳入 building=yes（沈阳方针）
  3. 面积推断高度（替代中位数填充）
  4. 建筑数 < 800 时记为 WARNING（不强求 CRITICAL → OK）

贵阳面积-高度规则（参考 mid_rise 主导城市）：
  ≥800m²  → 21m (7F) — 中高层板楼
  400-800m² → 15m (5F) — 多层住宅
  200-400m² → 12m (4F) — 低多层
  <200m²   →  6m (2F) — 低层/附属

用法：
  cd ~/Desktop/multi-cities-bipv-nc
  python scripts/fix_guiyang.py
  python scripts/04_energy_simulation.py --city guiyang
  python scripts/05_fdsi_scoring.py
============================================================================
"""

import logging
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import osmnx as ox
    ox.settings.log_console = False
    ox.settings.use_cache = True
    ox.settings.timeout = 600
except ImportError:
    print("ERROR: pip install osmnx geopandas")
    sys.exit(1)

from scipy.spatial import cKDTree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results" / "morphology"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GUIYANG_INFO = {
    "name_en": "Guiyang",
    "name_cn": "贵阳",
    "city_key": "guiyang",
    "climate_zone": "mild",
    "default_floor_height": 3.0,
    # old_city_wide: 最优 bbox，建筑数最多（831栋）
    "bbox": (26.63, 26.55, 106.75, 106.67),  # N, S, E, W
}

RESIDENTIAL_TAGS = {
    "residential", "apartments", "house", "detached",
    "semidetached_house", "terrace", "dormitory",
}

# 贵阳面积-高度规则（mid_rise 主导）
AREA_HEIGHT_RULES = [
    (800,  99999, 21.0, 7, "中高层板楼 (6-8F)"),
    (400,  800,   15.0, 5, "多层住宅 (4-6F)"),
    (200,  400,   12.0, 4, "低多层 (3-4F)"),
    (0,    200,    6.0, 2, "低层/附属 (1-2F)"),
]


def infer_height_v2(gdf, areas, default_floor_h=3.0):
    height = pd.Series(np.nan, index=gdf.index, dtype=float)
    source = pd.Series("unknown", index=gdf.index)

    if "height" in gdf.columns:
        h = pd.to_numeric(gdf["height"], errors="coerce")
        mask = h.notna() & (h > 0)
        height = height.where(~mask, h)
        source = source.where(~mask, "osm_height")

    if "building:levels" in gdf.columns:
        levels = pd.to_numeric(gdf["building:levels"], errors="coerce")
        h_from_levels = levels * default_floor_h
        mask = height.isna() & h_from_levels.notna() & (h_from_levels > 0)
        height = height.where(~mask, h_from_levels)
        source = source.where(~mask, "osm_levels")

    n_total = len(height)
    n_osm = (source != "unknown").sum()
    coverage = n_osm / n_total
    log.info(f"  高度属性: {n_osm:,} 已知 ({coverage:.1%}), {n_total - n_osm:,} 缺失")

    n_area_inferred = 0
    log.info(f"  启用 footprint 面积高度推断")
    for min_a, max_a, h_val, _, desc in AREA_HEIGHT_RULES:
        mask = height.isna() & (areas >= min_a) & (areas < max_a)
        n_match = mask.sum()
        if n_match > 0:
            height = height.where(~mask, h_val)
            source = source.where(~mask, "area_inferred")
            n_area_inferred += n_match
            log.info(f"    面积 [{min_a}-{max_a}m²] → {h_val}m ({desc}): {n_match} 栋")
    log.info(f"  面积推断合计: {n_area_inferred} 栋")

    n_still_missing = height.isna().sum()
    if n_still_missing > 0:
        known = height[height.notna()]
        fallback = known.median() if len(known) > 0 else 6 * default_floor_h
        height = height.fillna(fallback)
        log.info(f"  兜底填充: {n_still_missing} 栋 → {fallback:.1f}m")

    log.info("  高度来源分布:")
    for s, c in source.value_counts().items():
        log.info(f"    {s:15s}: {c:5d} ({c/n_total:.1%})")

    return height, source


def classify_typology(heights, floor_h=3.0):
    levels = (heights / floor_h).round().astype(int).clip(lower=1)
    conditions = [
        levels <= 3,
        (levels >= 4) & (levels <= 6),
        (levels >= 7) & (levels <= 9),
        levels >= 10,
    ]
    choices = ["low_rise", "mid_rise", "mid_high", "high_rise"]
    return pd.Series(np.select(conditions, choices, default="unknown"), index=heights.index)


def compute_d2_indicators(gdf, study_area_m2=None):
    heights = gdf["height_m"]
    areas = gdf["footprint_area_m2"]
    floors = gdf["n_floors"]

    if study_area_m2 is None:
        try:
            gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
            hull = gdf_proj.geometry.unary_union.convex_hull
            study_area_m2 = hull.area
        except Exception:
            study_area_m2 = areas.sum() / 0.3
        log.info(f"  研究区面积: {study_area_m2/1e6:.2f} km²")

    total_footprint = areas.sum()
    gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
    perimeters = gdf_proj.geometry.length
    compactness = (perimeters ** 2) / (4 * np.pi * areas)
    total_gfa = (areas * floors).sum()

    return {
        "n_buildings": len(gdf),
        "study_area_km2": round(study_area_m2 / 1e6, 3),
        "d2_1_height_mean": round(heights.mean(), 2),
        "d2_1_height_median": round(heights.median(), 2),
        "d2_1_height_std": round(heights.std(), 2),
        "d2_2_building_density": round(total_footprint / study_area_m2, 4),
        "d2_3_roof_area_mean": round(areas.mean(), 1),
        "d2_3_roof_area_median": round(areas.median(), 1),
        "d2_3_roof_area_total_m2": round(total_footprint, 0),
        "d2_4_compactness_mean": round(compactness.mean(), 3),
        "d2_4_compactness_median": round(compactness.median(), 3),
        "d2_5_far": round(total_gfa / study_area_m2, 3),
    }


def compute_d3_indicators(gdf):
    areas = gdf["footprint_area_m2"]
    heights = gdf["height_m"]
    typologies = gdf["typology"]

    ROOF_UTIL = {"low_rise": 0.60, "mid_rise": 0.65, "mid_high": 0.66, "high_rise": 0.68}
    gdf = gdf.copy()
    gdf["roof_utilization"] = typologies.map(ROOF_UTIL).fillna(0.65)

    SHADING_BY_TYPE = {"low_rise": 0.50, "mid_rise": 0.85, "mid_high": 0.95, "high_rise": 0.97}
    gdf["shading_factor"] = typologies.map(SHADING_BY_TYPE).fillna(0.80)

    try:
        gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
        centroids = np.array([(g.centroid.x, g.centroid.y) for g in gdf_proj.geometry])
        if len(centroids) > 1:
            tree = cKDTree(centroids)
            sf = gdf["shading_factor"].values.copy()
            for i in range(len(gdf)):
                dists, indices = tree.query(centroids[i], k=min(6, len(centroids)))
                neighbors = indices[1:]
                neighbor_heights = heights.iloc[neighbors].values
                my_height = heights.iloc[i]
                taller_ratio = (neighbor_heights > my_height * 1.5).mean()
                sf[i] *= max(1 - taller_ratio * 0.3, 0.5)
            gdf["shading_factor"] = sf
    except Exception as e:
        log.warning(f"  KNN遮挡修正失败: {e}")

    gdf["effective_pv_area_m2"] = areas * gdf["roof_utilization"] * gdf["shading_factor"]

    return gdf, {
        "d3_1_roof_utilization_mean": round(gdf["roof_utilization"].mean(), 3),
        "d3_2_shading_factor_mean": round(gdf["shading_factor"].mean(), 3),
        "d3_3_effective_area_mean_m2": round(gdf["effective_pv_area_m2"].mean(), 1),
        "d3_4_total_deployable_m2": round(gdf["effective_pv_area_m2"].sum(), 0),
        "d3_4_total_deployable_mw": round(gdf["effective_pv_area_m2"].sum() * 0.20 / 1000, 2),
    }


def compute_typology_stats(gdf):
    stats = gdf.groupby("typology").agg(
        count=("height_m", "size"),
        height_mean=("height_m", "mean"),
        height_std=("height_m", "std"),
        area_mean=("footprint_area_m2", "mean"),
        area_total=("footprint_area_m2", "sum"),
        floors_mean=("n_floors", "mean"),
        roof_util_mean=("roof_utilization", "mean"),
        shading_mean=("shading_factor", "mean"),
        effective_pv_mean=("effective_pv_area_m2", "mean"),
        effective_pv_total=("effective_pv_area_m2", "sum"),
    ).round(2)
    stats["pct"] = (stats["count"] / stats["count"].sum() * 100).round(1)
    type_order = ["low_rise", "mid_rise", "mid_high", "high_rise"]
    stats = stats.reindex([t for t in type_order if t in stats.index])
    return stats


def main():
    log.info("=" * 60)
    log.info("NC Phase 1d: 修复贵阳形态数据")
    log.info("=" * 60)

    city_key = GUIYANG_INFO["city_key"]
    floor_h = GUIYANG_INFO["default_floor_height"]
    N, S, E, W = GUIYANG_INFO["bbox"]

    log.info(f"\n[1] 从 OSM 获取贵阳建筑数据 (old_city_wide bbox)...")
    log.info(f"  bbox: N={N}, S={S}, E={E}, W={W}")
    log.info(f"  面积: {(N-S)*111:.1f}km × {(E-W)*111*0.88:.1f}km")

    try:
        gdf = ox.features_from_bbox(bbox=(W, S, E, N), tags={"building": True})
        gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    except Exception as e:
        log.error(f"  获取失败: {e}")
        sys.exit(1)

    log.info(f"  获取到 {len(gdf):,} 栋建筑")

    # 住宅 + building=yes
    if "building" in gdf.columns:
        is_res = gdf["building"].isin(RESIDENTIAL_TAGS)
        n_res = is_res.sum()
        log.info(f"  住宅标签: {n_res:,} ({n_res/len(gdf):.1%})")
        is_res |= gdf["building"] == "yes"
        n_total = is_res.sum()
        log.info(f"  含 building=yes: {n_total:,}")
        gdf_res = gdf[is_res].copy()
    else:
        gdf_res = gdf.copy()

    n_analyzed = len(gdf_res)
    log.info(f"  分析建筑数: {n_analyzed:,}")

    if n_analyzed < 500:
        log.error(f"  ✗ CRITICAL: 建筑数 {n_analyzed} < 500，OSM覆盖率不足")
    elif n_analyzed < 800:
        log.warning(f"  ⚠ WARNING: 建筑数 {n_analyzed} < 800，贵阳OSM覆盖率偏低（论文需说明）")
    else:
        log.info(f"  ✓ 建筑数 {n_analyzed} >= 800，可接受")

    # 面积
    gdf_proj = gdf_res.to_crs(gdf_res.estimate_utm_crs())
    gdf_res["footprint_area_m2"] = gdf_proj.geometry.area.values

    # 高度推断
    log.info("\n[2] 高度推断（面积规则）...")
    heights, sources = infer_height_v2(gdf_res, areas=gdf_res["footprint_area_m2"],
                                       default_floor_h=floor_h)
    gdf_res["height_m"] = heights.values
    gdf_res["height_source"] = sources.values
    gdf_res["n_floors"] = (gdf_res["height_m"] / floor_h).round().clip(lower=1).astype(int)

    log.info(f"\n  高度汇总: mean={heights.mean():.1f}m  median={heights.median():.1f}m  "
             f"P90={heights.quantile(0.9):.1f}m  max={heights.max():.1f}m")

    # 验证指标
    h_mean = heights.mean()
    if not (10 <= h_mean <= 35):
        log.warning(f"  ⚠ 均高 {h_mean:.1f}m 超出预期范围 [10-35m]")
    else:
        log.info(f"  ✓ 均高 {h_mean:.1f}m 在合理范围 [10-35m]")

    # 形态分类
    log.info("\n[3] 形态分类...")
    gdf_res["typology"] = classify_typology(gdf_res["height_m"]).values
    type_counts = gdf_res["typology"].value_counts()
    log.info(f"  形态分类:\n{type_counts.to_string()}")

    dominant_type = type_counts.index[0]
    dominant_pct = type_counts.iloc[0] / len(gdf_res)
    city_typology = f"{dominant_type}_dominant" if dominant_pct > 0.6 else "mixed"
    log.info(f"  城市形态: {city_typology} ({dominant_type}: {dominant_pct:.1%})")

    # D2/D3
    log.info("\n[4] D2/D3 指标计算...")
    d2 = compute_d2_indicators(gdf_res)
    gdf_res, d3 = compute_d3_indicators(gdf_res)
    typo_stats = compute_typology_stats(gdf_res)

    # 验证
    log.info("\n=== 修复后验证 ===")
    checks = [
        ("建筑数 > 500",        n_analyzed > 500,          f"{n_analyzed}"),
        ("建筑密度 > 0.01",     d2["d2_2_building_density"] > 0.01,
                                f"{d2['d2_2_building_density']:.4f}"),
        ("均高 10-35m",         10 <= d2["d2_1_height_mean"] <= 35,
                                f"{d2['d2_1_height_mean']:.1f}m"),
        ("FAR < 1.0",           d2["d2_5_far"] < 1.0,      f"{d2['d2_5_far']:.3f}"),
    ]
    all_pass = True
    for label, ok, val in checks:
        sym = "✓" if ok else "✗"
        log.info(f"  {sym} {label}: {val}")
        if not ok:
            all_pass = False

    # 对比
    log.info("\n" + "=" * 60)
    log.info("修复前后对比")
    log.info("=" * 60)
    old_d2_path = RESULTS_DIR / f"{city_key}_d2_indicators.csv"
    if old_d2_path.exists():
        old_d2 = pd.read_csv(old_d2_path).iloc[0]
        log.info(f"  {'指标':15s} {'旧值':>10s} {'新值':>10s}")
        log.info(f"  {'-'*40}")
        for label, old_k, new_v in [
            ("建筑数",    "n_buildings",           d2["n_buildings"]),
            ("均高(m)",   "d2_1_height_mean",      d2["d2_1_height_mean"]),
            ("中位高(m)", "d2_1_height_median",    d2["d2_1_height_median"]),
            ("建筑密度",  "d2_2_building_density", d2["d2_2_building_density"]),
            ("FAR",      "d2_5_far",              d2["d2_5_far"]),
        ]:
            log.info(f"  {label:15s} {str(old_d2.get(old_k,'?')):>10s} {str(new_v):>10s}")
    log.info(f"\n{typo_stats.to_string()}")

    # 保存
    log.info("\n[5] 保存修复数据...")
    for fname in [f"{city_key}_d2_indicators.csv", f"{city_key}_d3_indicators.csv",
                  f"{city_key}_typology_stats.csv", f"{city_key}_buildings_classified.gpkg"]:
        old = RESULTS_DIR / fname
        if old.exists():
            bak = RESULTS_DIR / f"{fname}.bak_v1"
            if not bak.exists():
                old.rename(bak)

    save_cols = ["geometry", "height_m", "height_source", "n_floors",
                 "footprint_area_m2", "typology", "roof_utilization",
                 "shading_factor", "effective_pv_area_m2"]
    save_cols = [c for c in save_cols if c in gdf_res.columns]
    gdf_res[save_cols].to_file(RESULTS_DIR / f"{city_key}_buildings_classified.gpkg", driver="GPKG")
    log.info(f"  已保存 gpkg: {len(gdf_res)} 栋")

    pd.DataFrame([{"city": city_key, **d2}]).to_csv(
        RESULTS_DIR / f"{city_key}_d2_indicators.csv", index=False)
    pd.DataFrame([{"city": city_key, **d3}]).to_csv(
        RESULTS_DIR / f"{city_key}_d3_indicators.csv", index=False)
    typo_stats.to_csv(RESULTS_DIR / f"{city_key}_typology_stats.csv")

    summary_path = RESULTS_DIR / "cross_city_d2d3_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        summary = summary[summary["city"] != city_key]
        new_row = {
            "city": city_key,
            "name_en": GUIYANG_INFO["name_en"],
            "name_cn": GUIYANG_INFO["name_cn"],
            "climate_zone": GUIYANG_INFO["climate_zone"],
            "status": "OK" if all_pass else "WARNING",
            "n_buildings_analyzed": len(gdf_res),
            "city_typology": city_typology,
            "dominant_type": dominant_type,
            "dominant_pct": round(dominant_pct, 3),
            "elapsed_s": 0,
            **d2, **d3,
        }
        summary = pd.concat([summary, pd.DataFrame([new_row])], ignore_index=True)
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        log.info(f"  已更新: {summary_path.name}")

    log.info(f"\n{'='*60}")
    log.info(f"修复{'完成' if all_pass else '完成（WARNING 状态）'}！")
    log.info(f"{'='*60}")
    log.info(f"  贵阳 OSM 覆盖率偏低属已知限制，建筑数={n_analyzed}，论文中需在 Section 2.3 说明。")


if __name__ == "__main__":
    main()
