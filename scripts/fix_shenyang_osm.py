#!/usr/bin/env python3
"""
============================================================================
NC Phase 1a: 修复沈阳 OSM 数据
============================================================================
问题诊断：
  旧 bbox [41.85, 41.75, 123.50, 123.38] 覆盖了沈河区郊区，
  仅获取687栋建筑，98%为低层(均高3.01m)。
  对比哈尔滨(98.9% mid_high, 均高21m)和长春(85.4% mid_rise, 均高17.7m)，
  沈阳的数据明显不代表其真实住宅形态。

修复方案：
  使用和平区核心住宅区 bbox [41.815, 41.775, 123.435, 123.385]
  该区域是沈阳最典型的多层/中高层住宅集中区。

  如果数据量仍不足，脚本会自动尝试两个备选 bbox：
  - 备选A：皇姑区住宅区
  - 备选B：和平区更大范围

用法：
  cd ~/Desktop/multi-cities-bipv-nc
  python scripts/fix_shenyang_osm.py

  # 只做 audit（不重跑 morphology）
  python scripts/fix_shenyang_osm.py --audit-only

  # 用自定义 bbox
  python scripts/fix_shenyang_osm.py --bbox 41.82,41.77,123.44,123.38
============================================================================
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

try:
    import osmnx as ox
    import geopandas as gpd
except ImportError:
    print("ERROR: pip install osmnx geopandas")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"

ox.settings.log_console = False
ox.settings.use_cache = True
ox.settings.timeout = 300

# ── bbox 候选方案 ──────────────────────────────────────────────

BBOX_CANDIDATES = {
    "heping_core": {
        "bbox": (41.815, 41.775, 123.435, 123.385),  # N, S, E, W
        "desc": "和平区核心住宅区（南湖-浑河-青年大街-南京街）",
    },
    "huanggu": {
        "bbox": (41.835, 41.795, 123.445, 123.390),
        "desc": "皇姑区住宅区（北陵附近）",
    },
    "heping_wide": {
        "bbox": (41.820, 41.770, 123.450, 123.375),
        "desc": "和平区扩大范围",
    },
}

# 旧 bbox（用于对比）
OLD_BBOX = (41.85, 41.75, 123.50, 123.38)

RESIDENTIAL_TAGS = {
    "residential", "apartments", "house", "detached",
    "semidetached_house", "terrace", "dormitory",
}


# ── 核心函数 ──────────────────────────────────────────────────

def fetch_and_audit(bbox, label=""):
    """获取建筑数据并返回审计摘要。"""
    north, south, east, west = bbox
    log.info(f"  [{label}] bbox: N={north}, S={south}, E={east}, W={west}")

    try:
        gdf = ox.features_from_bbox(bbox=(west, south, east, north), tags={"building": True})
        gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    except Exception as e:
        log.error(f"  [{label}] 获取失败: {e}")
        return None, None

    n_total = len(gdf)
    log.info(f"  [{label}] 获取建筑: {n_total}")

    if n_total == 0:
        return None, None

    # 住宅分类
    is_res = gdf["building"].isin(RESIDENTIAL_TAGS) if "building" in gdf.columns else pd.Series(False, index=gdf.index)
    n_res = is_res.sum()

    # 高度
    heights = pd.Series(np.nan, index=gdf.index)
    if "height" in gdf.columns:
        heights = heights.fillna(pd.to_numeric(gdf["height"], errors="coerce"))
    if "building:levels" in gdf.columns:
        levels = pd.to_numeric(gdf["building:levels"], errors="coerce")
        heights = heights.fillna(levels * 3.0)

    has_height = heights.notna()
    n_height = has_height.sum()

    # 面积
    gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
    areas = gdf_proj.geometry.area
    gdf["area_m2"] = areas.values
    gdf["height_m"] = heights.values

    # 建筑类型分布
    if "building" in gdf.columns:
        type_counts = gdf["building"].value_counts().head(5)
    else:
        type_counts = pd.Series()

    # 高度分布（有高度的建筑）
    h_valid = heights.dropna()
    if len(h_valid) > 0:
        h_mean = h_valid.mean()
        h_median = h_valid.median()
        # 类型统计
        n_low = (h_valid <= 10).sum()
        n_mid = ((h_valid > 10) & (h_valid <= 24)).sum()
        n_high = (h_valid > 24).sum()
    else:
        h_mean = h_median = 0
        n_low = n_mid = n_high = 0

    summary = {
        "label": label,
        "n_total": n_total,
        "n_residential": n_res,
        "res_ratio": round(n_res / n_total, 3),
        "n_with_height": n_height,
        "height_ratio": round(n_height / n_total, 3),
        "height_mean": round(h_mean, 1),
        "height_median": round(h_median, 1),
        "area_mean": round(areas.mean(), 1),
        "n_low_rise": n_low,
        "n_mid_rise": n_mid,
        "n_high_rise": n_high,
        "top_types": dict(type_counts),
    }

    log.info(f"  [{label}] 住宅: {n_res} ({summary['res_ratio']:.1%}), "
             f"有高度: {n_height} ({summary['height_ratio']:.1%})")
    log.info(f"  [{label}] 均高: {h_mean:.1f}m, 中位高: {h_median:.1f}m")
    log.info(f"  [{label}] 低层/多层中高层/高层: {n_low}/{n_mid}/{n_high}")

    return gdf, summary


def compare_results(old_summary, new_summary):
    """对比修复前后。"""
    log.info("\n" + "=" * 60)
    log.info("修复前后对比")
    log.info("=" * 60)

    metrics = ["n_total", "n_residential", "height_ratio", "height_mean", "height_median"]
    for m in metrics:
        old_v = old_summary.get(m, "N/A") if old_summary else "FAILED"
        new_v = new_summary.get(m, "N/A")
        log.info(f"  {m:20s}:  旧={old_v}  →  新={new_v}")

    # 与其他严寒区城市对比
    log.info("\n  参考（其他严寒区城市）:")
    log.info(f"    哈尔滨: 6577栋, 均高21.3m, 98.9% mid_high")
    log.info(f"    长春:   2936栋, 均高19.3m, 85.4% mid_rise")

    new_h = new_summary.get("height_mean", 0)
    if new_h >= 15:
        log.info(f"\n  ✓ 修复后均高 {new_h:.1f}m，与哈尔滨/长春一致，数据合理")
    elif new_h >= 10:
        log.info(f"\n  ⚠ 修复后均高 {new_h:.1f}m，偏低但可接受，建议检查建筑类型分布")
    else:
        log.info(f"\n  ✗ 修复后均高 {new_h:.1f}m，仍然偏低！建议尝试其他 bbox")


def save_fixed_data(gdf, summary):
    """保存修复后的数据。"""
    morph_dir = RESULTS_DIR / "morphology"

    # 备份旧文件
    for fname in ["shenyang_d2_indicators.csv", "shenyang_d3_indicators.csv",
                   "shenyang_typology_stats.csv", "shenyang_buildings_classified.gpkg"]:
        old_path = morph_dir / fname
        if old_path.exists():
            backup = morph_dir / f"{fname}.bak_prefixnc"
            old_path.rename(backup)
            log.info(f"  备份: {fname} → {fname}.bak_prefixnc")

    # 保存新的 gpkg
    gpkg_path = morph_dir / "shenyang_buildings_classified.gpkg"
    gdf.to_file(gpkg_path, driver="GPKG")
    log.info(f"  保存: {gpkg_path}")

    log.info("\n  ⚠ 注意：D2/D3 指标需要重新运行 03_morphology_analysis.py --city shenyang")
    log.info("  然后重新运行 04_energy_simulation.py --city shenyang")
    log.info("  最后重新运行 05_fdsi_scoring.py")


# ── 主流程 ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NC Phase 1a: 修复沈阳 OSM 数据")
    parser.add_argument("--bbox", type=str, default=None,
                        help="自定义 bbox: 'north,south,east,west'")
    parser.add_argument("--audit-only", action="store_true",
                        help="只做审计对比，不保存数据")
    parser.add_argument("--try-all", action="store_true",
                        help="尝试所有候选 bbox 并对比")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("NC Phase 1a: 修复沈阳 OSM 数据")
    log.info("=" * 60)

    # ── Step 1: 用旧bbox重新获取（对照组） ──
    log.info("\n[1/3] 重新获取旧 bbox 数据（对照组）...")
    _, old_summary = fetch_and_audit(OLD_BBOX, label="旧bbox")

    time.sleep(5)  # 礼貌等待

    # ── Step 2: 尝试新 bbox ──
    if args.try_all:
        log.info("\n[2/3] 尝试所有候选 bbox...")
        best_gdf = None
        best_summary = None
        best_score = 0

        for name, info in BBOX_CANDIDATES.items():
            log.info(f"\n  --- {name}: {info['desc']} ---")
            gdf, summary = fetch_and_audit(info["bbox"], label=name)
            time.sleep(5)

            if summary is not None:
                # 打分：住宅数量 × 高度覆盖率 × 均高合理性
                h = summary["height_mean"]
                h_score = min(h / 21.0, 1.0) if h > 3 else 0  # 21m是哈尔滨参考值
                score = summary["n_residential"] * summary["height_ratio"] * h_score
                log.info(f"  [{name}] 综合评分: {score:.1f}")

                if score > best_score:
                    best_score = score
                    best_gdf = gdf
                    best_summary = summary

        if best_summary:
            log.info(f"\n  最佳方案: {best_summary['label']}")
            compare_results(old_summary, best_summary)

            if not args.audit_only:
                save_fixed_data(best_gdf, best_summary)
        else:
            log.error("  所有候选 bbox 都失败了！")

    else:
        # 使用指定或默认的新 bbox
        if args.bbox:
            parts = [float(x.strip()) for x in args.bbox.split(",")]
            new_bbox = tuple(parts)
            label = "自定义bbox"
        else:
            new_bbox = BBOX_CANDIDATES["heping_core"]["bbox"]
            label = "和平区核心"

        log.info(f"\n[2/3] 获取新 bbox 数据 ({label})...")
        new_gdf, new_summary = fetch_and_audit(new_bbox, label=label)

        if new_summary is not None:
            compare_results(old_summary, new_summary)

            if not args.audit_only:
                log.info(f"\n[3/3] 保存修复数据...")
                save_fixed_data(new_gdf, new_summary)
        else:
            log.error("  新 bbox 获取失败！")
            log.info("  建议尝试: python scripts/fix_shenyang_osm.py --try-all")

    log.info("\n完成。")
    log.info("\n后续步骤:")
    log.info("  1. python scripts/03_morphology_analysis.py --city shenyang")
    log.info("  2. python scripts/04_energy_simulation.py --city shenyang")
    log.info("  3. python scripts/05_fdsi_scoring.py")


if __name__ == "__main__":
    main()
