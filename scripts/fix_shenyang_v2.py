#!/usr/bin/env python3
"""
============================================================================
NC Phase 1a (v2): 修复沈阳 — 改进高度推断 + 重跑 morphology
============================================================================
问题根因：
  沈阳 OSM 高度覆盖率仅 2.1%，现有 infer_height() 用中位数填充缺失值，
  导致 98% 的建筑被赋予低层高度，形态分析完全失真。

修复方案：
  在 03_morphology_analysis.py 的 infer_height() 中增加第三优先级：
  基于 footprint 面积推断高度（在中位数填充之前）。

  东北住宅区的经验规则（参考哈尔滨/长春的有高度建筑数据）：
  - footprint ≥ 800m² → 中高层板楼 (7层, 21m) — 典型东北6-8层板式住宅
  - footprint 400-800m² → 多层 (5层, 15m)
  - footprint 200-400m² → 低多层 (4层, 12m)
  - footprint < 200m² → 低层 (2层, 6m)

  这个推断只在 height 和 building:levels 都缺失时启用。
  论文中需说明这个辅助推断方法及其局限性。

用法：
  cd ~/Desktop/multi-cities-bipv-nc

  # Step 1: 用 heping_wide bbox 重新获取沈阳数据 + 改进高度推断 + morphology
  python scripts/fix_shenyang_v2.py

  # Step 2: 查看修复前后对比
  # (脚本会自动输出对比表)

  # Step 3: 确认后，重跑后续 pipeline
  python scripts/04_energy_simulation.py --city shenyang
  python scripts/05_fdsi_scoring.py
============================================================================
"""

import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Optional

import geopandas as gpd
import numpy as np
import pandas as pd

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

# ── 沈阳配置 ──
SHENYANG_BBOX = (41.820, 41.770, 123.450, 123.375)  # heping_wide: N, S, E, W
SHENYANG_INFO = {
    "name_en": "Shenyang", "name_cn": "沈阳",
    "climate_zone": "severe_cold",
    "default_floor_height": 3.0,
}

# ── 住宅标签 ──
RESIDENTIAL_TAGS = {
    "residential", "apartments", "house", "detached",
    "semidetached_house", "terrace", "dormitory",
}

# ── 面积-高度推断规则 ──
# 基于哈尔滨/长春有高度标注建筑的统计分析
AREA_HEIGHT_RULES = [
    # (min_area, max_area, inferred_height, inferred_floors, description)
    (800,  99999, 21.0, 7, "中高层板楼 (典型东北6-8层)"),
    (400,  800,   15.0, 5, "多层 (4-6层)"),
    (200,  400,   12.0, 4, "低多层 (3-4层)"),
    (0,    200,    6.0, 2, "低层 (1-2层)"),
]


# ============================================================================
# 改进版 infer_height
# ============================================================================

def infer_height_v2(
    gdf: gpd.GeoDataFrame,
    areas: pd.Series,
    default_floor_h: float = 3.0,
    use_area_rules: bool = True,
) -> pd.Series:
    """
    改进版高度推断。优先级：
    1. height 标签 → 直接使用
    2. building:levels 标签 → 乘以层高
    3. 基于 footprint 面积推断（仅当高度覆盖率 < 30% 时启用）
    4. 缺失 → 用已知高度的中位数填充
    """
    height = pd.Series(np.nan, index=gdf.index, dtype=float)
    source = pd.Series("unknown", index=gdf.index)

    # 优先级 1: height 标签
    if "height" in gdf.columns:
        h = pd.to_numeric(gdf["height"], errors="coerce")
        mask = h.notna()
        height = height.where(~mask, h)
        source = source.where(~mask, "osm_height")

    # 优先级 2: building:levels
    if "building:levels" in gdf.columns:
        levels = pd.to_numeric(gdf["building:levels"], errors="coerce")
        h_from_levels = levels * default_floor_h
        mask = height.isna() & h_from_levels.notna()
        height = height.where(~mask, h_from_levels)
        source = source.where(~mask, "osm_levels")

    # 统计
    n_total = len(height)
    n_osm = (source != "unknown").sum()
    coverage = n_osm / n_total
    log.info(f"  高度属性: {n_osm:,} 已知 ({coverage:.1%}), {n_total - n_osm:,} 缺失")

    # 优先级 3: 基于面积推断（高度覆盖率 < 30% 时启用）
    n_area_inferred = 0
    if use_area_rules and coverage < 0.30:
        log.info(f"  ⚠ 高度覆盖率 < 30%，启用基于 footprint 面积的高度推断")
        for min_a, max_a, h_val, _, desc in AREA_HEIGHT_RULES:
            mask = height.isna() & (areas >= min_a) & (areas < max_a)
            n_match = mask.sum()
            if n_match > 0:
                height = height.where(~mask, h_val)
                source = source.where(~mask, "area_inferred")
                n_area_inferred += n_match
                log.info(f"    面积 [{min_a}-{max_a}m²] → {h_val}m ({desc}): {n_match} 栋")

        log.info(f"  面积推断合计: {n_area_inferred} 栋")

    # 优先级 4: 中位数填充
    n_still_missing = height.isna().sum()
    if n_still_missing > 0:
        known = height.dropna()
        if len(known) > 0:
            median_h = known.median()
            height = height.fillna(median_h)
            source = source.fillna("median_fill")
            log.info(f"  中位数填充: {n_still_missing} 栋 → {median_h:.1f}m")
        else:
            default_h = 6 * default_floor_h
            height = height.fillna(default_h)
            source = source.fillna("default")
            log.warning(f"  ⚠ 无任何高度信息，用默认值: {default_h:.1f}m")

    # 高度来源统计
    source_counts = source.value_counts()
    log.info(f"  高度来源分布:")
    for s, c in source_counts.items():
        log.info(f"    {s:15s}: {c:5d} ({c/n_total:.1%})")

    return height, source


# ============================================================================
# 复用 03_morphology_analysis.py 的其他函数
# ============================================================================

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
    n = len(gdf)
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
        "n_buildings": n,
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
    n_floors = gdf["n_floors"]
    typologies = gdf["typology"]

    ROOF_UTIL = {"low_rise": 0.60, "mid_rise": 0.65, "mid_high": 0.66, "high_rise": 0.68}
    gdf["roof_utilization"] = typologies.map(ROOF_UTIL).fillna(0.65)

    # 遮挡系数（简化proxy）
    SHADING_BY_TYPE = {"low_rise": 0.50, "mid_rise": 0.85, "mid_high": 0.95, "high_rise": 0.97}
    gdf["shading_factor"] = typologies.map(SHADING_BY_TYPE).fillna(0.80)

    # KNN遮挡修正
    try:
        gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
        centroids = np.array([(g.centroid.x, g.centroid.y) for g in gdf_proj.geometry])
        if len(centroids) > 1:
            tree = cKDTree(centroids)
            for i in range(len(gdf)):
                dists, indices = tree.query(centroids[i], k=min(6, len(centroids)))
                neighbors = indices[1:]
                neighbor_heights = heights.iloc[neighbors].values
                my_height = heights.iloc[i]
                taller_ratio = (neighbor_heights > my_height * 1.5).mean()
                shading_penalty = 1 - taller_ratio * 0.3
                gdf.iloc[i, gdf.columns.get_loc("shading_factor")] *= max(shading_penalty, 0.5)
    except Exception as e:
        log.warning(f"  KNN遮挡修正失败: {e}")

    gdf["effective_pv_area_m2"] = areas * gdf["roof_utilization"] * gdf["shading_factor"]
    d3_4_total = gdf["effective_pv_area_m2"].sum()

    return {
        "d3_1_roof_utilization_mean": round(gdf["roof_utilization"].mean(), 3),
        "d3_2_shading_factor_mean": round(gdf["shading_factor"].mean(), 3),
        "d3_3_effective_area_mean_m2": round(gdf["effective_pv_area_m2"].mean(), 1),
        "d3_4_total_deployable_m2": round(d3_4_total, 0),
        "d3_4_total_deployable_mw": round(d3_4_total * 0.20 / 1000, 2),
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


# ============================================================================
# 主流程
# ============================================================================

def main():
    log.info("=" * 60)
    log.info("NC Phase 1a (v2): 修复沈阳形态数据")
    log.info("=" * 60)

    # ── 1. 获取数据 ──
    log.info("\n[1] 从 OSM 获取沈阳建筑数据 (heping_wide bbox)...")
    north, south, east, west = SHENYANG_BBOX
    log.info(f"  bbox: N={north}, S={south}, E={east}, W={west}")

    try:
        gdf = ox.features_from_bbox(bbox=(west, south, east, north), tags={"building": True})
        gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    except Exception as e:
        log.error(f"  获取失败: {e}")
        log.info("  Overpass API 可能过载，请稍后重试")
        sys.exit(1)

    log.info(f"  获取到 {len(gdf):,} 栋建筑")

    # ── 2. 筛选住宅 ──
    is_res = gdf["building"].isin(RESIDENTIAL_TAGS) if "building" in gdf.columns else pd.Series(False, index=gdf.index)
    n_res = is_res.sum()
    log.info(f"  住宅建筑: {n_res:,} / {len(gdf):,} ({n_res/len(gdf):.1%})")

    # 中国OSM中 building=yes 是住宅的主要标签，始终纳入
    if "building" in gdf.columns:
        is_res |= gdf["building"] == "yes"
        n_res = is_res.sum()
        log.info(f"  含 building=yes 后: {n_res:,}")

    if n_res < 100:
        log.warning(f"  住宅不足100栋，使用全部建筑")
        gdf_res = gdf.copy()
    else:
        gdf_res = gdf[is_res].copy()

    log.info(f"  分析建筑数: {len(gdf_res):,}")

    # ── 3. 计算面积 ──
    gdf_proj = gdf_res.to_crs(gdf_res.estimate_utm_crs())
    gdf_res["footprint_area_m2"] = gdf_proj.geometry.area.values

    # ── 4. 改进版高度推断 ──
    log.info("\n[2] 高度推断（改进版 v2）...")
    heights, sources = infer_height_v2(
        gdf_res,
        areas=gdf_res["footprint_area_m2"],
        default_floor_h=SHENYANG_INFO["default_floor_height"],
        use_area_rules=True,
    )
    gdf_res["height_m"] = heights.values
    gdf_res["height_source"] = sources.values
    gdf_res["n_floors"] = (gdf_res["height_m"] / SHENYANG_INFO["default_floor_height"]).round().clip(lower=1).astype(int)

    # ── 5. 形态分类 ──
    log.info("\n[3] 形态分类...")
    gdf_res["typology"] = classify_typology(gdf_res["height_m"]).values
    type_counts = gdf_res["typology"].value_counts()
    log.info(f"  形态分类:\n{type_counts.to_string()}")

    dominant_type = type_counts.index[0]
    dominant_pct = type_counts.iloc[0] / len(gdf_res)
    city_typology = f"{dominant_type}_dominant" if dominant_pct > 0.6 else "mixed"
    log.info(f"  城市形态: {city_typology} ({dominant_type}: {dominant_pct:.1%})")

    # ── 6. D2/D3 指标 ──
    log.info("\n[4] D2/D3 指标计算...")
    d2 = compute_d2_indicators(gdf_res)
    d3 = compute_d3_indicators(gdf_res)
    typo_stats = compute_typology_stats(gdf_res)

    # ── 7. 对比旧结果 ──
    log.info("\n" + "=" * 60)
    log.info("修复前后对比")
    log.info("=" * 60)

    old_d2_path = RESULTS_DIR / "shenyang_d2_indicators.csv"
    if old_d2_path.exists():
        old_d2 = pd.read_csv(old_d2_path).iloc[0]
        comparisons = [
            ("建筑数", old_d2.get("n_buildings", "?"), d2["n_buildings"]),
            ("均高(m)", old_d2.get("d2_1_height_mean", "?"), d2["d2_1_height_mean"]),
            ("中位高(m)", old_d2.get("d2_1_height_median", "?"), d2["d2_1_height_median"]),
            ("建筑密度", old_d2.get("d2_2_building_density", "?"), d2["d2_2_building_density"]),
            ("屋顶面积(m²)", old_d2.get("d2_3_roof_area_mean", "?"), d2["d2_3_roof_area_mean"]),
            ("FAR", old_d2.get("d2_5_far", "?"), d2["d2_5_far"]),
        ]
        log.info(f"  {'指标':15s} {'旧值':>10s} {'新值':>10s} {'参考(哈尔滨)':>12s}")
        log.info(f"  {'-'*50}")
        refs = [6577, 21.26, 21.0, 0.052, 1513.3, 0.374]
        for (label, old_v, new_v), ref in zip(comparisons, refs):
            log.info(f"  {label:15s} {str(old_v):>10s} {str(new_v):>10s} {ref:>12}")
    else:
        log.info("  （无旧数据可对比）")

    log.info(f"\n  形态类型统计:")
    log.info(f"\n{typo_stats.to_string()}")

    log.info(f"\n  参考:")
    log.info(f"    哈尔滨: mid_high 98.9%, 均高 21.3m")
    log.info(f"    长春:   mid_rise 85.4%, 均高 19.3m")

    # ── 8. 保存 ──
    log.info("\n[5] 保存修复数据...")

    # 备份旧文件
    for fname in ["shenyang_d2_indicators.csv", "shenyang_d3_indicators.csv",
                   "shenyang_typology_stats.csv", "shenyang_buildings_classified.gpkg"]:
        old = RESULTS_DIR / fname
        if old.exists():
            bak = RESULTS_DIR / f"{fname}.bak_v1"
            if not bak.exists():  # 只备份一次
                old.rename(bak)
                log.info(f"  备份: {fname}")

    # 保存新文件
    save_cols = ["geometry", "height_m", "height_source", "n_floors",
                 "footprint_area_m2", "typology", "roof_utilization",
                 "shading_factor", "effective_pv_area_m2"]
    save_cols = [c for c in save_cols if c in gdf_res.columns]
    gdf_res[save_cols].to_file(RESULTS_DIR / "shenyang_buildings_classified.gpkg", driver="GPKG")

    pd.DataFrame([{"city": "shenyang", **d2}]).to_csv(
        RESULTS_DIR / "shenyang_d2_indicators.csv", index=False)
    pd.DataFrame([{"city": "shenyang", **d3}]).to_csv(
        RESULTS_DIR / "shenyang_d3_indicators.csv", index=False)
    typo_stats.to_csv(RESULTS_DIR / "shenyang_typology_stats.csv")

    # 更新 cross_city summary
    summary_path = RESULTS_DIR / "cross_city_d2d3_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        # 删除旧沈阳行
        summary = summary[summary["city"] != "shenyang"]
        # 添加新行
        new_row = {
            "city": "shenyang",
            "name_en": "Shenyang", "name_cn": "沈阳",
            "climate_zone": "severe_cold",
            "status": "OK",
            "n_buildings_analyzed": len(gdf_res),
            "city_typology": city_typology,
            "dominant_type": dominant_type,
            "dominant_pct": round(dominant_pct, 3),
            "elapsed_s": 0,
            **d2, **d3,
        }
        summary = pd.concat([summary, pd.DataFrame([new_row])], ignore_index=True)
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        log.info(f"  已更新: {summary_path}")

    log.info(f"\n{'='*60}")
    log.info("修复完成！")
    log.info(f"{'='*60}")
    log.info(f"\n后续步骤:")
    log.info(f"  1. python scripts/04_energy_simulation.py --city shenyang")
    log.info(f"  2. python scripts/05_fdsi_scoring.py")
    log.info(f"  3. git add -A && git commit -m 'fix: shenyang OSM bbox + area-based height inference'")


if __name__ == "__main__":
    main()
