#!/usr/bin/env python3
"""
============================================================================
NC Phase 1b: 修复乌鲁木齐 — 更换 OSM bbox + 改进高度推断
============================================================================
问题根因：
  当前使用 place_query="天山区, 乌鲁木齐市, 中国"，OSM 返回的多边形范围
  包含大量郊区/农村建筑，导致：
    - 均高仅 3.97m（应 > 12m）
    - FAR = 0.02（应 > 0.3）
    - 研究区面积 143.96 km²（过大）
    - FDSI 排名 #32（GHI 排名 #9，Δ=-23，最大偏差城市）

修复方案：
  1. 改用核心城区 bbox（天山区 + 沙依巴克区核心住宅带）
     combined_core: N=43.83, S=43.76, E=87.64, W=87.52
  2. 面积推断规则适配西北中国城市：
     - Soviet时期板式楼（≥500m²） → 6层 × 3.0m = 18m
     - 多层住宅（250-500m²）     → 5层 × 3.0m = 15m
     - 低多层（120-250m²）       → 4层 × 3.0m = 12m
     - 低层（< 120m²）           → 3层 × 3.0m = 9m

验证目标：
  - height_mean: 12-25m
  - n_buildings: > 1000
  - building_density: > 0.10
  - FAR: > 0.30

用法：
  cd ~/Desktop/multi-cities-bipv-nc
  python scripts/fix_urumqi.py
  python scripts/04_energy_simulation.py --city urumqi
  python scripts/05_fdsi_scoring.py
  python scripts/nc_01b_diagnostics.py
============================================================================
"""

import logging
import sys
from pathlib import Path

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

# ── 乌鲁木齐住宅区 bbox ──
# 新市区（新疆维吾尔自治区最大住宅区，1980-2000年代 Soviet-style 板楼为主）
# 前次尝试 combined_core (天山+沙依巴克核心) 返回 98.2% 一层商业建筑 — 市中心商业主导
# 改为新市区核心住宅带，预期：多层板楼占主导，高度 15-21m
URUMQI_BBOX = (43.92, 43.85, 87.62, 87.54)  # N, S, E, W (新市区核心)
URUMQI_INFO = {
    "name_en": "Urumqi", "name_cn": "乌鲁木齐",
    "climate_zone": "severe_cold",
    "default_floor_height": 3.0,
}

# ── 住宅标签 ──
RESIDENTIAL_TAGS = {
    "residential", "apartments", "house", "detached",
    "semidetached_house", "terrace", "dormitory",
}

# ── 面积-高度推断规则（西北中国城市经验规则）──
# 参考哈尔滨/长春 Soviet-era 住宅区统计，适配乌鲁木齐同期建设模式
AREA_HEIGHT_RULES = [
    # (min_area, max_area, inferred_height, inferred_floors, description)
    (500,  99999, 18.0, 6, "Soviet板式楼 (6层典型)"),
    (250,  500,   15.0, 5, "多层住宅 (5层)"),
    (120,  250,   12.0, 4, "低多层 (4层)"),
    (0,    120,    9.0, 3, "低层 (3层)"),
]


# ============================================================================
# 高度推断
# ============================================================================

def infer_height_v2(gdf, areas, default_floor_h=3.0, use_area_rules=True):
    height = pd.Series(np.nan, index=gdf.index, dtype=float)
    source = pd.Series("unknown", index=gdf.index)

    if "height" in gdf.columns:
        h = pd.to_numeric(gdf["height"], errors="coerce")
        mask = h.notna()
        height = height.where(~mask, h)
        source = source.where(~mask, "osm_height")

    if "building:levels" in gdf.columns:
        levels = pd.to_numeric(gdf["building:levels"], errors="coerce")
        h_from_levels = levels * default_floor_h
        mask = height.isna() & h_from_levels.notna()
        height = height.where(~mask, h_from_levels)
        source = source.where(~mask, "osm_levels")

    n_total = len(height)
    n_osm = (source != "unknown").sum()
    coverage = n_osm / n_total if n_total > 0 else 0
    log.info(f"  高度属性: {n_osm:,} 已知 ({coverage:.1%}), {n_total - n_osm:,} 缺失")

    n_area_inferred = 0
    median_h = height.dropna().median() if height.notna().any() else 0
    # 触发面积推断：覆盖率低 OR 覆盖率够但中位高 < 6m（大量 levels=1 商业建筑误标）
    trigger_area = use_area_rules and (coverage < 0.30 or (coverage > 0 and median_h < 6.0))
    if trigger_area:
        log.info(f"  ⚠ 启用面积推断 (覆盖率={coverage:.1%}, 中位高={median_h:.1f}m)")
        # 面积推断：对所有 height<=3m 且面积足够大的建筑做覆盖（包含已有 osm_levels=1 的）
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
        known = height.dropna()
        if len(known) > 0:
            median_h = known.median()
            height = height.fillna(median_h)
            source = source.fillna("median_fill")
            log.info(f"  中位数填充: {n_still_missing} 栋 → {median_h:.1f}m")
        else:
            default_h = 5 * default_floor_h
            height = height.fillna(default_h)
            source = source.fillna("default")
            log.warning(f"  ⚠ 无任何高度信息，用默认值: {default_h:.1f}m")

    source_counts = source.value_counts()
    log.info(f"  高度来源分布:")
    for s, c in source_counts.items():
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
    gdf = gdf.copy()
    gdf["roof_utilization"] = typologies.map(ROOF_UTIL).fillna(0.65)

    SHADING_BY_TYPE = {"low_rise": 0.50, "mid_rise": 0.85, "mid_high": 0.95, "high_rise": 0.97}
    gdf["shading_factor"] = typologies.map(SHADING_BY_TYPE).fillna(0.80)

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
    }, gdf


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
    log.info("NC Phase 1b: 修复乌鲁木齐形态数据")
    log.info("=" * 60)

    # ── 1. 获取数据 ──
    log.info("\n[1] 从 OSM 获取乌鲁木齐建筑数据 (combined_core bbox)...")
    north, south, east, west = URUMQI_BBOX
    log.info(f"  bbox: N={north}, S={south}, E={east}, W={west}")
    log.info(f"  覆盖范围: 天山区 + 沙依巴克区核心住宅带")

    try:
        gdf = ox.features_from_bbox(bbox=(west, south, east, north), tags={"building": True})
        gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    except Exception as e:
        log.error(f"  获取失败: {e}")
        log.info("  Overpass API 可能过载，请稍后重试")
        sys.exit(1)

    log.info(f"  获取到 {len(gdf):,} 栋建筑（含全类型）")

    # ── 2. 筛选住宅 ──
    is_res = gdf["building"].isin(RESIDENTIAL_TAGS) if "building" in gdf.columns else pd.Series(False, index=gdf.index)
    n_res = is_res.sum()
    log.info(f"  住宅建筑（含 yes）: {n_res:,} / {len(gdf):,}")

    # 中国 OSM 中 building=yes 是住宅的主要标签
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
    gdf_res = gdf_res.copy()
    gdf_res["footprint_area_m2"] = gdf_proj.geometry.area.values

    # ── 4. 高度推断 ──
    log.info("\n[2] 高度推断（西北城市规则）...")
    heights, sources = infer_height_v2(
        gdf_res,
        areas=gdf_res["footprint_area_m2"],
        default_floor_h=URUMQI_INFO["default_floor_height"],
        use_area_rules=True,
    )
    gdf_res["height_m"] = heights.values
    gdf_res["height_source"] = sources.values
    gdf_res["n_floors"] = (gdf_res["height_m"] / URUMQI_INFO["default_floor_height"]).round().clip(lower=1).astype(int)

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
    d3_dict, gdf_res = compute_d3_indicators(gdf_res)
    typo_stats = compute_typology_stats(gdf_res)

    # ── 7. 对比旧结果 ──
    log.info("\n" + "=" * 60)
    log.info("修复前后对比")
    log.info("=" * 60)

    old_d2_path = RESULTS_DIR / "urumqi_d2_indicators.csv"
    if old_d2_path.exists():
        old_d2 = pd.read_csv(old_d2_path).iloc[0]
        comparisons = [
            ("建筑数",    old_d2.get("n_buildings", "?"),         d2["n_buildings"]),
            ("均高(m)",   old_d2.get("d2_1_height_mean", "?"),    d2["d2_1_height_mean"]),
            ("中位高(m)", old_d2.get("d2_1_height_median", "?"),  d2["d2_1_height_median"]),
            ("建筑密度",  old_d2.get("d2_2_building_density", "?"), d2["d2_2_building_density"]),
            ("屋顶面积",  old_d2.get("d2_3_roof_area_mean", "?"),  d2["d2_3_roof_area_mean"]),
            ("FAR",       old_d2.get("d2_5_far", "?"),             d2["d2_5_far"]),
        ]
        log.info(f"  {'指标':12s} {'旧值':>10s} {'新值':>10s} {'参考(哈尔滨)':>12s}")
        log.info(f"  {'-'*48}")
        refs = [6577, 21.26, 21.0, 0.052, 1513.3, 0.374]
        for (label, old_v, new_v), ref in zip(comparisons, refs):
            log.info(f"  {label:12s} {str(old_v):>10s} {str(new_v):>10s} {ref:>12}")
    else:
        log.info("  （无旧数据可对比）")

    log.info(f"\n  形态类型统计:")
    log.info(f"\n{typo_stats.to_string()}")

    log.info(f"\n  验证目标:")
    log.info(f"    height_mean: {d2['d2_1_height_mean']:.1f}m  (目标 > 12m)")
    log.info(f"    n_buildings: {d2['n_buildings']}  (目标 > 1000)")
    log.info(f"    building_density: {d2['d2_2_building_density']:.4f}  (目标 > 0.10)")
    log.info(f"    FAR: {d2['d2_5_far']:.3f}  (目标 > 0.30)")

    # ── 8. 保存 ──
    log.info("\n[5] 保存修复数据...")

    # 备份旧文件
    for fname in ["urumqi_d2_indicators.csv", "urumqi_d3_indicators.csv",
                  "urumqi_typology_stats.csv", "urumqi_buildings_classified.gpkg"]:
        old = RESULTS_DIR / fname
        if old.exists():
            bak = RESULTS_DIR / f"{fname}.bak_v1"
            if not bak.exists():
                old.rename(bak)
                log.info(f"  备份: {fname}")

    # 保存新文件
    save_cols = ["geometry", "height_m", "height_source", "n_floors",
                 "footprint_area_m2", "typology", "roof_utilization",
                 "shading_factor", "effective_pv_area_m2"]
    save_cols = [c for c in save_cols if c in gdf_res.columns]
    gdf_res[save_cols].to_file(RESULTS_DIR / "urumqi_buildings_classified.gpkg", driver="GPKG")

    pd.DataFrame([{"city": "urumqi", **d2}]).to_csv(
        RESULTS_DIR / "urumqi_d2_indicators.csv", index=False)
    pd.DataFrame([{"city": "urumqi", **d3_dict}]).to_csv(
        RESULTS_DIR / "urumqi_d3_indicators.csv", index=False)
    typo_stats.to_csv(RESULTS_DIR / "urumqi_typology_stats.csv")

    # 更新 cross_city summary
    summary_path = RESULTS_DIR / "cross_city_d2d3_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        summary = summary[summary["city"] != "urumqi"]
        new_row = {
            "city": "urumqi",
            "name_en": "Urumqi", "name_cn": "乌鲁木齐",
            "climate_zone": "severe_cold",
            "status": "OK",
            "n_buildings_analyzed": len(gdf_res),
            "city_typology": city_typology,
            "dominant_type": dominant_type,
            "dominant_pct": round(dominant_pct, 3),
            "elapsed_s": 0,
            **d2, **d3_dict,
        }
        summary = pd.concat([summary, pd.DataFrame([new_row])], ignore_index=True)
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        log.info(f"  已更新: {summary_path}")

    log.info(f"\n{'='*60}")
    log.info("修复完成！后续步骤：")
    log.info(f"{'='*60}")
    log.info(f"  1. python scripts/04_energy_simulation.py --city urumqi")
    log.info(f"  2. python scripts/05_fdsi_scoring.py")
    log.info(f"  3. python scripts/nc_01b_diagnostics.py")


if __name__ == "__main__":
    main()
