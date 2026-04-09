#!/usr/bin/env python3
"""
============================================================================
NC Phase 2: 批量形态分析 — 24 个新城市
============================================================================
从 configs/*.yaml 读取城市配置，使用面积推断高度（替代中位数填充），
输出与现有城市相同格式的 morphology 文件。

用法:
  python scripts/nc_03_morphology_new_cities.py              # 全部新城市
  python scripts/nc_03_morphology_new_cities.py --city dalian
  python scripts/nc_03_morphology_new_cities.py --skip-existing
============================================================================
"""

import argparse
import logging
import sys
import time
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
    ox.settings.timeout = 180
except ImportError:
    sys.exit("ERROR: pip install osmnx")

try:
    import yaml
except ImportError:
    sys.exit("ERROR: pip install pyyaml")

from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROJECT_DIR  = Path(__file__).resolve().parent.parent
CONFIGS_DIR  = PROJECT_DIR / "configs"
RESULTS_DIR  = PROJECT_DIR / "results" / "morphology"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 24 个新城市（原始 15 城不在此列表）
NEW_CITIES = [
    # severe_cold
    "dalian", "hohhot", "tangshan", "urumqi",
    # cold
    "taiyuan", "shijiazhuang", "lanzhou", "yinchuan", "xining",
    "qingdao", "wuxi", "suzhou", "tianjin", "zhengzhou",
    # hscw
    "hangzhou", "hefei", "nanchang", "ningbo", "shanghai", "chongqing",
    # hsww
    "fuzhou", "nanning", "haikou",
    # mild
    "lhasa",
    # non-mainland (NC extension)
    "hongkong", "taipei",
]

RESIDENTIAL_TAGS = {
    "residential", "apartments", "house", "detached",
    "semidetached_house", "terrace", "dormitory",
}

# 通用面积-高度规则（适用于中国主要城市住宅区）
AREA_HEIGHT_RULES = [
    # (min_m2, max_m2, height_m, floors, label)
    (1500, 99999, 33.0, 11, "大型商住 (10-12F)"),
    (800,  1500,  27.0,  9, "中高层住宅 (8-10F)"),
    (400,  800,   18.0,  6, "多层住宅 (5-7F)"),
    (200,  400,   12.0,  4, "低多层 (3-4F)"),
    (0,    200,    6.0,  2, "低层/附属 (1-2F)"),
]

ROOF_UTIL    = {"low_rise": 0.60, "mid_rise": 0.65, "mid_high": 0.66, "high_rise": 0.68}
SHADING_BASE = {"low_rise": 0.50, "mid_rise": 0.85, "mid_high": 0.95, "high_rise": 0.97}


# ── helpers ──────────────────────────────────────────────────────────────────

def load_yaml(city_key: str) -> dict:
    path = CONFIGS_DIR / f"{city_key}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def fetch_buildings(cfg: dict, city_key: str) -> gpd.GeoDataFrame:
    osm = cfg["osm"]
    if osm.get("place_query"):
        log.info(f"  place_query: {osm['place_query']}")
        gdf = ox.features_from_place(osm["place_query"], tags={"building": True})
    elif osm.get("bbox"):
        bb = osm["bbox"]  # [N, S, E, W]
        N, S, E, W = bb
        log.info(f"  bbox: N={N} S={S} E={E} W={W}")
        gdf = ox.features_from_bbox(bbox=(W, S, E, N), tags={"building": True})
    else:
        raise ValueError(f"{city_key}: no place_query or bbox in config")
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    log.info(f"  获取到 {len(gdf):,} 栋建筑")
    return gdf


def filter_residential(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "building" not in gdf.columns:
        return gdf.copy()
    is_res = gdf["building"].isin(RESIDENTIAL_TAGS) | (gdf["building"] == "yes")
    n_res = is_res.sum()
    log.info(f"  住宅+yes 筛选: {n_res:,} / {len(gdf):,}")
    return gdf[is_res].copy() if n_res >= 100 else gdf.copy()


def infer_height(gdf: gpd.GeoDataFrame, areas: pd.Series,
                 floor_h: float = 3.0) -> tuple[pd.Series, pd.Series]:
    height = pd.Series(np.nan, index=gdf.index, dtype=float)
    source = pd.Series("unknown", index=gdf.index)

    # 1. OSM height tag
    if "height" in gdf.columns:
        h = pd.to_numeric(gdf["height"], errors="coerce").clip(upper=500)
        mask = h.notna() & (h > 0)
        height = height.where(~mask, h)
        source = source.where(~mask, "osm_height")

    # 2. building:levels tag
    if "building:levels" in gdf.columns:
        lv = pd.to_numeric(gdf["building:levels"], errors="coerce")
        hf = (lv * floor_h).clip(upper=500)
        mask = height.isna() & hf.notna() & (hf > 0)
        height = height.where(~mask, hf)
        source = source.where(~mask, "osm_levels")

    n_total = len(height)
    n_known = (source != "unknown").sum()
    coverage = n_known / n_total
    log.info(f"  高度覆盖: {n_known}/{n_total} ({coverage:.1%})")

    # 3. 面积推断（覆盖率 < 50% 时启用，避免高层离群值拉偏中位数）
    if coverage < 0.50:
        log.info(f"  覆盖率 < 50%，启用面积推断")
        for min_a, max_a, h_val, _, desc in AREA_HEIGHT_RULES:
            mask = height.isna() & (areas >= min_a) & (areas < max_a)
            n = mask.sum()
            if n:
                height = height.where(~mask, h_val)
                source = source.where(~mask, "area_inferred")
                log.info(f"    [{min_a}-{max_a}m²] → {h_val}m ({desc}): {n}")

    # 4. 兜底中位数
    still_missing = height.isna().sum()
    if still_missing:
        known = height.dropna()
        fb = known.median() if len(known) else 6 * floor_h
        height = height.fillna(fb)
        log.info(f"  兜底中位数: {still_missing} 栋 → {fb:.1f}m")

    log.info("  来源: " + " | ".join(
        f"{s}={c}" for s, c in source.value_counts().items()))
    return height, source


def classify_typology(heights: pd.Series, floor_h: float = 3.0) -> pd.Series:
    lv = (heights / floor_h).round().astype(int).clip(lower=1)
    cond = [lv <= 3, (lv >= 4) & (lv <= 6), (lv >= 7) & (lv <= 9), lv >= 10]
    return pd.Series(
        np.select(cond, ["low_rise", "mid_rise", "mid_high", "high_rise"], "unknown"),
        index=heights.index)


def compute_d2(gdf: gpd.GeoDataFrame) -> dict:
    h = gdf["height_m"]; a = gdf["footprint_area_m2"]; fl = gdf["n_floors"]
    try:
        gp = gdf.to_crs(gdf.estimate_utm_crs())
        hull = gp.geometry.unary_union.convex_hull
        study_area = hull.area
    except Exception:
        study_area = a.sum() / 0.3
    log.info(f"  研究区面积: {study_area/1e6:.2f} km²")
    gp = gdf.to_crs(gdf.estimate_utm_crs())
    perim = gp.geometry.length
    compact = (perim ** 2) / (4 * np.pi * a)
    return {
        "n_buildings": len(gdf),
        "study_area_km2": round(study_area / 1e6, 3),
        "d2_1_height_mean": round(h.mean(), 2),
        "d2_1_height_median": round(h.median(), 2),
        "d2_1_height_std": round(h.std(), 2),
        "d2_2_building_density": round(a.sum() / study_area, 4),
        "d2_3_roof_area_mean": round(a.mean(), 1),
        "d2_3_roof_area_median": round(a.median(), 1),
        "d2_3_roof_area_total_m2": round(a.sum(), 0),
        "d2_4_compactness_mean": round(compact.mean(), 3),
        "d2_4_compactness_median": round(compact.median(), 3),
        "d2_5_far": round((a * fl).sum() / study_area, 3),
    }


def compute_d3(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict]:
    gdf = gdf.copy()
    gdf["roof_utilization"] = gdf["typology"].map(ROOF_UTIL).fillna(0.65)
    gdf["shading_factor"]   = gdf["typology"].map(SHADING_BASE).fillna(0.80)

    # KNN shading correction
    try:
        gp = gdf.to_crs(gdf.estimate_utm_crs())
        cen = np.array([(g.centroid.x, g.centroid.y) for g in gp.geometry])
        if len(cen) > 1:
            tree = cKDTree(cen)
            sf = gdf["shading_factor"].values.copy()
            h  = gdf["height_m"].values
            for i in range(len(gdf)):
                _, idx = tree.query(cen[i], k=min(6, len(cen)))
                nh = h[idx[1:]]
                sf[i] *= max(1 - (nh > h[i] * 1.5).mean() * 0.3, 0.5)
            gdf["shading_factor"] = sf
    except Exception as e:
        log.warning(f"  KNN 修正失败: {e}")

    gdf["effective_pv_area_m2"] = (
        gdf["footprint_area_m2"] * gdf["roof_utilization"] * gdf["shading_factor"])
    return gdf, {
        "d3_1_roof_utilization_mean": round(gdf["roof_utilization"].mean(), 3),
        "d3_2_shading_factor_mean":   round(gdf["shading_factor"].mean(), 3),
        "d3_3_effective_area_mean_m2":round(gdf["effective_pv_area_m2"].mean(), 1),
        "d3_4_total_deployable_m2":   round(gdf["effective_pv_area_m2"].sum(), 0),
        "d3_4_total_deployable_mw":   round(gdf["effective_pv_area_m2"].sum() * 0.20 / 1000, 2),
    }


def process_city(city_key: str, skip_existing: bool = False) -> dict:
    log.info(f"\n{'='*60}")
    log.info(f"  {city_key.upper()}")
    log.info(f"{'='*60}")

    gpkg_path = RESULTS_DIR / f"{city_key}_buildings_classified.gpkg"
    if skip_existing and gpkg_path.exists():
        log.info(f"  已存在，跳过（--skip-existing）")
        return {"city": city_key, "status": "SKIPPED"}

    try:
        cfg      = load_yaml(city_key)
        city_cfg = cfg["city"]
        floor_h  = cfg["osm"].get("default_floor_height", 3.0)

        # 获取建筑
        raw = fetch_buildings(cfg, city_key)
        gdf = filter_residential(raw)

        # 面积
        gp = gdf.to_crs(gdf.estimate_utm_crs())
        gdf["footprint_area_m2"] = gp.geometry.area.values

        # 高度
        heights, sources = infer_height(gdf, gdf["footprint_area_m2"], floor_h)
        gdf["height_m"]      = heights.values
        gdf["height_source"] = sources.values
        gdf["n_floors"]      = (heights / floor_h).round().clip(lower=1).astype(int)

        h_mean = heights.mean()
        log.info(f"  高度: mean={h_mean:.1f}m  median={heights.median():.1f}m  "
                 f"P90={heights.quantile(0.9):.1f}m")

        # 形态分类
        gdf["typology"] = classify_typology(heights, floor_h).values
        tc = gdf["typology"].value_counts()
        log.info(f"  形态:\n{tc.to_string()}")
        dom = tc.index[0]; dom_pct = tc.iloc[0] / len(gdf)
        city_typology = f"{dom}_dominant" if dom_pct > 0.6 else "mixed"

        # D2 / D3
        d2 = compute_d2(gdf)
        gdf, d3 = compute_d3(gdf)

        # 验证
        warns = []
        if len(gdf) < 500:   warns.append(f"建筑数偏少({len(gdf)})")
        if d2["d2_2_building_density"] < 0.01: warns.append("密度偏低")
        if not (8 <= h_mean <= 60):  warns.append(f"均高异常({h_mean:.1f}m)")
        if d2["d2_5_far"] > 1.5:     warns.append(f"FAR过高({d2['d2_5_far']})")
        status = "WARNING" if warns else "OK"
        if warns: log.warning(f"  ⚠ {'; '.join(warns)}")
        else:     log.info(f"  ✓ 验证通过")

        # 保存
        save_cols = [c for c in ["geometry", "height_m", "height_source", "n_floors",
                                  "footprint_area_m2", "typology", "roof_utilization",
                                  "shading_factor", "effective_pv_area_m2"]
                     if c in gdf.columns]
        gdf[save_cols].to_file(gpkg_path, driver="GPKG")

        pd.DataFrame([{"city": city_key, **d2}]).to_csv(
            RESULTS_DIR / f"{city_key}_d2_indicators.csv", index=False)
        pd.DataFrame([{"city": city_key, **d3}]).to_csv(
            RESULTS_DIR / f"{city_key}_d3_indicators.csv", index=False)

        row = {
            "city": city_key,
            "name_en": city_cfg["name_en"], "name_cn": city_cfg["name_cn"],
            "climate_zone": city_cfg["climate_zone"],
            "status": status,
            "n_buildings_analyzed": len(gdf),
            "city_typology": city_typology,
            "dominant_type": dom, "dominant_pct": round(dom_pct, 3),
            "elapsed_s": 0,
            **d2, **d3,
        }
        log.info(f"  → {city_key}: {len(gdf)} 栋, mean_h={h_mean:.1f}m, "
                 f"FAR={d2['d2_5_far']}, status={status}")
        return row

    except Exception as e:
        log.error(f"  ✗ {city_key} 失败: {e}")
        import traceback; traceback.print_exc()
        return {"city": city_key, "status": "FAILED", "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, default=None,
                        help="只处理指定城市")
    parser.add_argument("--skip-existing", action="store_true",
                        help="跳过已有 gpkg 的城市")
    args = parser.parse_args()

    cities = [args.city.lower()] if args.city else NEW_CITIES
    for c in cities:
        if c not in NEW_CITIES:
            log.error(f"未知城市: {c}，可选: {NEW_CITIES}")
            sys.exit(1)

    log.info(f"处理 {len(cities)} 个城市: {cities}")
    results = []
    for i, city_key in enumerate(cities):
        row = process_city(city_key, skip_existing=args.skip_existing)
        results.append(row)
        if i < len(cities) - 1:
            log.info("  等待 10s (API 限速)...")
            time.sleep(10)

    # 更新 cross_city summary
    ok_rows = [r for r in results if r.get("status") not in ("FAILED", "SKIPPED")]
    if ok_rows:
        summary_path = RESULTS_DIR / "cross_city_d2d3_summary.csv"
        new_df = pd.DataFrame(ok_rows)
        if summary_path.exists():
            old = pd.read_csv(summary_path)
            old = old[~old["city"].isin(new_df["city"])]
            combined = pd.concat([old, new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(summary_path, index=False, encoding="utf-8-sig")
        log.info(f"\n汇总表已更新: {summary_path}  ({len(combined)} 城市)")

    # 最终报告
    log.info("\n" + "="*60)
    log.info("批量形态分析完成")
    log.info("="*60)
    for r in results:
        sym = {"OK": "✓", "WARNING": "⚠", "FAILED": "✗", "SKIPPED": "–"}.get(
              r.get("status", "?"), "?")
        n = r.get("n_buildings_analyzed", "-")
        h = r.get("d2_1_height_mean", "-")
        log.info(f"  {sym} {r['city']:<15} 建筑={n}  均高={h}m  {r.get('status','?')}")


if __name__ == "__main__":
    main()
