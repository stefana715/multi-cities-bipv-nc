#!/usr/bin/env python3
"""
============================================================================
Paper 4 - D2 密度/容积率修复脚本
============================================================================
问题：Step 3 用 convex hull 估算 study_area，导致跨城市密度不可比。
      深圳 hull=61km² vs 北京 hull=431km²，尺度差 7 倍。

修复：
  方案 A: 用官方行政区面积重算区域密度和容积率
  方案 B: 从已有 gpkg 计算 100m 缓冲区局部密度均值

输出：
  results/morphology/cross_city_d2d3_summary_fixed.csv  — 修正后的汇总表
  （原文件备份为 _backup.csv）

用法：
  python scripts/fix_d2_density.py
============================================================================
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
MORPHOLOGY_DIR = PROJECT_DIR / "results" / "morphology"

# ============================================================================
# 官方行政区面积 (km²)
# 来源：各区政府官网/统计年鉴，论文中需注明
# ============================================================================

OFFICIAL_AREA = {
    "harbin":   {"district": "Nangang", "district_cn": "南岗区",
                 "area_km2": 182.87, "source": "哈尔滨市统计年鉴"},
    "beijing":  {"district": "Haidian", "district_cn": "海淀区",
                 "area_km2": 430.77, "source": "北京市统计年鉴"},
    "changsha": {"district": "Yuelu", "district_cn": "岳麓区",
                 "area_km2": 552.02, "source": "长沙市统计年鉴"},
    "shenzhen": {"district": "Futian", "district_cn": "福田区",
                 "area_km2": 78.66, "source": "深圳市统计年鉴"},
    "kunming":  {"district": "Central 4 districts", "district_cn": "主城四区",
                 "area_km2": 300.00, "source": "估算值，论文需注明bbox范围"},
}


def fix_density_and_far():
    """用官方面积重算密度和FAR。"""

    summary_path = MORPHOLOGY_DIR / "cross_city_d2d3_summary.csv"
    if not summary_path.exists():
        log.error(f"找不到: {summary_path}")
        sys.exit(1)

    # 备份原文件
    backup_path = MORPHOLOGY_DIR / "cross_city_d2d3_summary_backup.csv"
    df = pd.read_csv(summary_path)
    df.to_csv(backup_path, index=False)
    log.info(f"原文件已备份: {backup_path}")

    log.info("\n修正前后对比:")
    log.info(f"{'城市':>8s}  {'原面积km²':>10s} {'官方面积km²':>10s}  "
             f"{'原密度':>8s} {'新密度':>8s}  {'原FAR':>8s} {'新FAR':>8s}")
    log.info("-" * 75)

    for _, row in df.iterrows():
        city = row["city"]
        if city not in OFFICIAL_AREA:
            continue

        official = OFFICIAL_AREA[city]
        official_area_m2 = official["area_km2"] * 1e6

        # 原值
        old_area_km2 = row["study_area_km2"]
        old_density = row["d2_2_building_density"]
        old_far = row["d2_5_far"]

        # 从原始数据反推 total_footprint 和 total_gfa
        total_footprint = old_density * old_area_km2 * 1e6  # m²
        total_gfa = old_far * old_area_km2 * 1e6  # m²

        # 用官方面积重算
        new_density = total_footprint / official_area_m2
        new_far = total_gfa / official_area_m2

        # 更新 DataFrame
        idx = df[df["city"] == city].index[0]
        df.loc[idx, "study_area_km2_convex_hull"] = old_area_km2  # 保留原值
        df.loc[idx, "study_area_km2"] = official["area_km2"]
        df.loc[idx, "study_area_source"] = "official_admin"
        df.loc[idx, "d2_2_building_density_hull"] = old_density  # 保留原值
        df.loc[idx, "d2_2_building_density"] = round(new_density, 4)
        df.loc[idx, "d2_5_far_hull"] = old_far  # 保留原值
        df.loc[idx, "d2_5_far"] = round(new_far, 3)

        log.info(f"{city:>8s}  {old_area_km2:>10.2f} {official['area_km2']:>10.2f}  "
                 f"{old_density:>8.4f} {new_density:>8.4f}  "
                 f"{old_far:>8.3f} {new_far:>8.3f}")

    # 保存修正后的文件
    fixed_path = MORPHOLOGY_DIR / "cross_city_d2d3_summary.csv"
    df.to_csv(fixed_path, index=False, encoding="utf-8-sig")
    log.info(f"\n修正后文件已保存: {fixed_path}")

    # 打印新的 D2 指标汇总
    log.info("\n" + "=" * 60)
    log.info("修正后 D2 指标汇总")
    log.info("=" * 60)
    display_cols = ["city", "n_buildings_analyzed", "d2_1_height_mean",
                    "d2_2_building_density", "d2_5_far", "d2_3_roof_area_mean",
                    "study_area_km2"]
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].to_string(index=False))

    # 合理性检查
    log.info("\n合理性检查:")
    for _, row in df.iterrows():
        city = row["city"]
        density = row["d2_2_building_density"]
        far = row["d2_5_far"]
        height = row["d2_1_height_mean"]
        implied_floors = far / density if density > 0 else 0
        log.info(f"  {city:>8s}: density={density:.4f}, FAR={far:.3f}, "
                 f"implied avg floors={implied_floors:.1f}, "
                 f"actual avg height={height:.1f}m ({height/3:.1f} floors)")


def compute_local_density_summary():
    """
    方案 B: 从 gpkg 计算局部密度（如果 gpkg 文件存在）。
    局部密度 = 100m 缓冲区内所有建筑 footprint 面积之和 / 缓冲区面积
    """
    import geopandas as gpd
    from scipy.spatial import cKDTree

    log.info("\n" + "=" * 60)
    log.info("方案 B: 局部邻域密度计算")
    log.info("=" * 60)

    results = []
    for city in OFFICIAL_AREA.keys():
        gpkg_path = MORPHOLOGY_DIR / f"{city}_buildings_classified.gpkg"
        if not gpkg_path.exists():
            log.warning(f"  {city}: gpkg 不存在，跳过局部密度计算")
            continue

        log.info(f"  处理 {city}...")
        gdf = gpd.read_file(gpkg_path)
        gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())

        centroids = np.column_stack([
            gdf_proj.geometry.centroid.x,
            gdf_proj.geometry.centroid.y,
        ])
        areas = gdf_proj.geometry.area.values

        # KD-Tree 100m 邻域密度
        tree = cKDTree(centroids)
        buffer_r = 100  # meters
        buffer_area = np.pi * buffer_r ** 2

        # 抽样计算（大数据集）
        n = len(gdf)
        sample_size = min(n, 2000)
        sample_idx = np.random.choice(n, sample_size, replace=False)

        local_densities = []
        for i in sample_idx:
            neighbors = tree.query_ball_point(centroids[i], buffer_r)
            neighbor_footprint = areas[neighbors].sum()
            local_densities.append(neighbor_footprint / buffer_area)

        local_density_mean = np.mean(local_densities)
        local_density_median = np.median(local_densities)
        local_density_std = np.std(local_densities)

        results.append({
            "city": city,
            "local_density_100m_mean": round(local_density_mean, 4),
            "local_density_100m_median": round(local_density_median, 4),
            "local_density_100m_std": round(local_density_std, 4),
            "n_sampled": sample_size,
        })

        log.info(f"    局部密度(100m): mean={local_density_mean:.4f}, "
                 f"median={local_density_median:.4f}, std={local_density_std:.4f}")

    if results:
        local_df = pd.DataFrame(results)
        local_path = MORPHOLOGY_DIR / "local_density_100m.csv"
        local_df.to_csv(local_path, index=False)
        log.info(f"\n局部密度保存: {local_path}")

        # 合并到主表
        summary_path = MORPHOLOGY_DIR / "cross_city_d2d3_summary.csv"
        df = pd.read_csv(summary_path)
        df = df.merge(local_df, on="city", how="left")
        df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        log.info("已合并到主汇总表")


def main():
    log.info("=" * 60)
    log.info("D2 密度/容积率修复")
    log.info("=" * 60)

    # 方案 A: 官方面积修正
    fix_density_and_far()

    # 方案 B: 局部密度（需要 gpkg）
    try:
        compute_local_density_summary()
    except ImportError:
        log.warning("geopandas 未安装，跳过局部密度计算")
    except Exception as e:
        log.warning(f"局部密度计算失败: {e}")

    log.info("\n修复完成。请重跑 Step 5:")
    log.info("  python scripts/05_fdsi_scoring.py")


if __name__ == "__main__":
    main()
