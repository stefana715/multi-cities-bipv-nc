#!/usr/bin/env python3
"""
============================================================================
Paper 4 - Script 03: Urban Morphology & Technical Deployment Analysis
============================================================================
Step 3: 对五个城市批量计算 D2（城市形态）和 D3（技术部署）维度的全部子指标。

工作流程：
  1. 从 OSM 获取建筑数据（或从已有 GeoPackage 读取）
  2. 筛选住宅建筑
  3. 推断缺失高度
  4. 形态分类（高层/多层/低层/混合）
  5. 计算 D2 子指标：建筑高度、密度、屋顶面积、紧凑度、容积率
  6. 计算 D3 子指标：可部署面积比、遮挡损失系数（proxy）
  7. 按形态类型分组统计
  8. 输出汇总表 + 分城市详细表

输出：
  results/morphology/{city}_buildings_classified.gpkg  — 分类后的建筑数据
  results/morphology/{city}_d2_indicators.csv          — D2 子指标
  results/morphology/{city}_d3_indicators.csv          — D3 子指标
  results/morphology/{city}_typology_stats.csv         — 形态类型统计
  results/morphology/cross_city_d2d3_summary.csv       — 五城市汇总对比

用法：
  python scripts/03_morphology_analysis.py              # 全部五城市
  python scripts/03_morphology_analysis.py --city beijing  # 单个城市
  python scripts/03_morphology_analysis.py --from-gpkg     # 从已有gpkg读取

依赖：
  pip install osmnx geopandas pandas numpy shapely scipy pyyaml tabulate
============================================================================
"""

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import box

try:
    import osmnx as ox
    ox.settings.log_console = False
    ox.settings.use_cache = True
    ox.settings.timeout = 600
except ImportError:
    print("ERROR: osmnx not installed. Run: pip install osmnx")
    sys.exit(1)

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# ── Setup ──────────────────────────────────────────────────────────────────

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
AUDIT_DIR = PROJECT_DIR / "results" / "osm_audit"


# ============================================================================
# 城市定义
# ============================================================================

CITIES = {
    "harbin": {
        "name_en": "Harbin", "name_cn": "哈尔滨",
        "climate_zone": "severe_cold",
        "place_query": "南岗区, 哈尔滨市, 中国",
        "bbox": None,
        "default_floor_height": 3.0,
    },
    "beijing": {
        "name_en": "Beijing", "name_cn": "北京",
        "climate_zone": "cold",
        "place_query": "海淀区, 北京市, 中国",
        "bbox": None,
        "default_floor_height": 3.0,
    },
    "changsha": {
        "name_en": "Changsha", "name_cn": "长沙",
        "climate_zone": "hscw",
        "place_query": "岳麓区, 长沙市, 中国",
        "bbox": None,
        "default_floor_height": 3.0,
    },
    "shenzhen": {
        "name_en": "Shenzhen", "name_cn": "深圳",
        "climate_zone": "hsww",
        "place_query": "福田区, 深圳市, 中国",
        "bbox": None,
        "default_floor_height": 3.0,
    },
    "kunming": {
        "name_en": "Kunming", "name_cn": "昆明",
        "climate_zone": "mild",
        "place_query": None,
        "bbox": (25.10, 24.90, 102.85, 102.55),  # north, south, east, west
        "default_floor_height": 3.0,
    },
    # ── 10 new cities (Phase 2 expansion) ──
    "changchun": {
        "name_en": "Changchun", "name_cn": "长春",
        "climate_zone": "severe_cold",
        "place_query": "朝阳区, 长春市, 中国",
        "bbox": None,
        "default_floor_height": 3.0,
    },
    "shenyang": {
        "name_en": "Shenyang", "name_cn": "沈阳",
        "climate_zone": "severe_cold",
        "place_query": None,
        "bbox": (41.85, 41.75, 123.50, 123.38),  # north, south, east, west; DCS=0.401
        "default_floor_height": 3.0,
    },
    "jinan": {
        "name_en": "Jinan", "name_cn": "济南",
        "climate_zone": "cold",
        "place_query": "历下区, 济南市, 中国",
        "bbox": None,
        "default_floor_height": 3.0,
    },
    "xian": {
        "name_en": "Xian", "name_cn": "西安",
        "climate_zone": "cold",
        # 原 碑林区 FAR=1.09（古城超密集），改为未央区住宅区（更具代表性）
        "place_query": None,
        "bbox": (34.38, 34.30, 108.96, 108.88),  # N,S,E,W weiyang_residential
        "default_floor_height": 3.0,
    },
    "wuhan": {
        "name_en": "Wuhan", "name_cn": "武汉",
        "climate_zone": "hscw",
        "place_query": "武昌区, 武汉市, 中国",
        "bbox": None,
        "default_floor_height": 3.0,
    },
    "nanjing": {
        "name_en": "Nanjing", "name_cn": "南京",
        "climate_zone": "hscw",
        "place_query": "鼓楼区, 南京市, 中国",
        "bbox": None,
        "default_floor_height": 3.0,
    },
    "guangzhou": {
        "name_en": "Guangzhou", "name_cn": "广州",
        "climate_zone": "hsww",
        "place_query": "天河区, 广州市, 中国",
        "bbox": None,
        "default_floor_height": 3.0,
    },
    "xiamen": {
        "name_en": "Xiamen", "name_cn": "厦门",
        "climate_zone": "hsww",
        "place_query": "思明区, 厦门市, 中国",
        "bbox": None,
        "default_floor_height": 3.0,
    },
    "guiyang": {
        "name_en": "Guiyang", "name_cn": "贵阳",
        "climate_zone": "mild",
        "place_query": None,
        "bbox": (26.72, 26.52, 106.80, 106.55),  # 云岩区+南明区联合 bbox
        "default_floor_height": 3.0,
    },
    "chengdu": {
        "name_en": "Chengdu", "name_cn": "成都",
        "climate_zone": "mild",
        "place_query": "锦江区, 成都市, 中国",
        "bbox": None,
        "default_floor_height": 3.0,
    },
    # ── 2 non-mainland cities (NC extension) ──
    "hongkong": {
        "name_en": "Hong Kong", "name_cn": "香港",
        "climate_zone": "hsww",
        "place_query": None,
        "bbox": [22.34, 22.30, 114.19, 114.14],  # Kowloon core
        "default_floor_height": 3.0,
    },
    "taipei": {
        "name_en": "Taipei", "name_cn": "台北",
        "climate_zone": "hscw",
        "place_query": None,
        "bbox": [25.06, 25.01, 121.57, 121.51],  # Da'an/Zhongzheng/Xinyi
        "default_floor_height": 3.0,
    },
}

# ── 住宅建筑标签 ──
RESIDENTIAL_BUILDING_TAGS = {
    "residential", "apartments", "house", "detached",
    "semidetached_house", "terrace", "dormitory",
}

# ── 形态分类阈值（层数）──
# 依据中国《民用建筑设计统一标准》GB 50352
TYPOLOGY_THRESHOLDS = {
    "low_rise": (1, 3),      # 低层：1-3层 (≤10m)
    "mid_rise": (4, 6),      # 多层：4-6层 (10-20m)
    "mid_high": (7, 9),      # 中高层：7-9层 (20-27m)
    "high_rise": (10, 999),  # 高层：≥10层 (≥27m)
}


# ============================================================================
# 1. 数据获取与预处理
# ============================================================================

def fetch_buildings(city_key: str, city_info: dict) -> gpd.GeoDataFrame:
    """从 OSM 获取建筑数据。"""
    log.info(f"  从 OSM 获取建筑数据...")
    tags = {"building": True}

    if city_info["bbox"] is not None:
        north, south, east, west = city_info["bbox"]
        gdf = ox.features_from_bbox(bbox=(west, south, east, north), tags=tags)
    else:
        gdf = ox.features_from_place(city_info["place_query"], tags=tags)

    # 只保留面状几何
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    log.info(f"  获取到 {len(gdf):,} 栋建筑")
    return gdf


def load_from_gpkg(city_key: str) -> Optional[gpd.GeoDataFrame]:
    """尝试从已有 GeoPackage 读取。"""
    gpkg_path = AUDIT_DIR / f"{city_key}_buildings.gpkg"
    if gpkg_path.exists():
        log.info(f"  从 GeoPackage 读取: {gpkg_path}")
        return gpd.read_file(gpkg_path)
    return None


def classify_residential(gdf: gpd.GeoDataFrame) -> pd.Series:
    """判断建筑是否为住宅。"""
    is_res = pd.Series(False, index=gdf.index)
    if "building" in gdf.columns:
        is_res |= gdf["building"].isin(RESIDENTIAL_BUILDING_TAGS)
    if "landuse" in gdf.columns:
        is_res |= gdf["landuse"] == "residential"
    return is_res


def infer_height(gdf: gpd.GeoDataFrame, default_floor_h: float = 3.0) -> pd.Series:
    """
    推断建筑高度。优先级：
    1. height 标签 → 直接使用
    2. building:levels 标签 → 乘以层高
    3. 缺失 → 用住宅建筑的中位数填充（分城市）
    """
    height = pd.Series(np.nan, index=gdf.index, dtype=float)

    if "height" in gdf.columns:
        h = pd.to_numeric(gdf["height"], errors="coerce")
        height = height.fillna(h)

    if "building:levels" in gdf.columns:
        levels = pd.to_numeric(gdf["building:levels"], errors="coerce")
        h_from_levels = levels * default_floor_h
        height = height.fillna(h_from_levels)

    # 统计缺失率
    n_total = len(height)
    n_known = height.notna().sum()
    n_missing = n_total - n_known
    log.info(f"  高度属性: {n_known:,} 已知 ({n_known/n_total:.1%}), "
             f"{n_missing:,} 缺失")

    # 用中位数填充缺失值
    if n_missing > 0 and n_known > 0:
        median_h = height.median()
        height = height.fillna(median_h)
        log.info(f"  缺失高度用中位数填充: {median_h:.1f}m")
    elif n_known == 0:
        # 完全没有高度信息，用默认值（6层住宅）
        default_h = 6 * default_floor_h
        height = height.fillna(default_h)
        log.warning(f"  ⚠ 无高度信息，用默认值: {default_h:.1f}m")

    return height


def classify_typology(heights: pd.Series, floor_h: float = 3.0) -> pd.Series:
    """
    根据高度进行形态分类。

    分类依据：中国《民用建筑设计统一标准》GB 50352
    - 低层 (low_rise): 1-3层, ≤10m
    - 多层 (mid_rise): 4-6层, 10-20m
    - 中高层 (mid_high): 7-9层, 20-27m
    - 高层 (high_rise): ≥10层, ≥27m
    """
    levels = (heights / floor_h).round().astype(int).clip(lower=1)

    conditions = [
        levels <= 3,
        (levels >= 4) & (levels <= 6),
        (levels >= 7) & (levels <= 9),
        levels >= 10,
    ]
    choices = ["low_rise", "mid_rise", "mid_high", "high_rise"]

    return pd.Series(
        np.select(conditions, choices, default="unknown"),
        index=heights.index,
    )


# ============================================================================
# 2. D2 城市形态维度指标计算
# ============================================================================

def compute_d2_indicators(
    gdf: gpd.GeoDataFrame,
    study_area_m2: Optional[float] = None,
) -> Dict:
    """
    计算 D2 维度全部子指标。

    子指标：
      d2_1: 平均建筑高度 (m)
      d2_2: 建筑密度 (%) = Σ footprint / study_area
      d2_3: 平均屋顶面积 (m²)
      d2_4: 形态紧凑度 = mean(perimeter² / (4π × area))  (1=圆形, 越大越不规则)
      d2_5: 容积率 FAR = Σ(footprint × floors) / study_area

    Parameters
    ----------
    gdf : GeoDataFrame
        必须包含 'height_m', 'footprint_area_m2', 'n_floors' 列
    study_area_m2 : float, optional
        研究区总面积。如果为 None，用建筑 footprint 的 convex hull 面积估算。
    """
    n = len(gdf)

    # ── 基础统计 ──
    heights = gdf["height_m"]
    areas = gdf["footprint_area_m2"]
    floors = gdf["n_floors"]

    # ── 研究区面积估算 ──
    if study_area_m2 is None:
        # 用所有建筑 footprint 的 convex hull 估算
        try:
            gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
            hull = gdf_proj.geometry.unary_union.convex_hull
            study_area_m2 = hull.area
        except Exception:
            study_area_m2 = areas.sum() / 0.3  # 假设 30% 建筑密度
        log.info(f"  研究区面积估算: {study_area_m2/1e6:.2f} km²")

    # ── d2_1: 平均建筑高度 ──
    d2_1_height_mean = heights.mean()
    d2_1_height_median = heights.median()
    d2_1_height_std = heights.std()

    # ── d2_2: 建筑密度 ──
    total_footprint = areas.sum()
    d2_2_density = total_footprint / study_area_m2

    # ── d2_3: 平均屋顶面积 ──
    d2_3_roof_area_mean = areas.mean()
    d2_3_roof_area_median = areas.median()
    d2_3_roof_area_total = total_footprint

    # ── d2_4: 形态紧凑度 (compactness) ──
    # compactness = perimeter² / (4π × area)
    # = 1 for circle, > 1 for more irregular shapes
    gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
    perimeters = gdf_proj.geometry.length
    compactness = (perimeters ** 2) / (4 * np.pi * areas)
    d2_4_compactness_mean = compactness.mean()
    d2_4_compactness_median = compactness.median()

    # ── d2_5: 容积率 FAR ──
    total_gfa = (areas * floors).sum()  # Gross Floor Area
    d2_5_far = total_gfa / study_area_m2

    return {
        "n_buildings": n,
        "study_area_km2": round(study_area_m2 / 1e6, 3),
        # D2 子指标
        "d2_1_height_mean": round(d2_1_height_mean, 2),
        "d2_1_height_median": round(d2_1_height_median, 2),
        "d2_1_height_std": round(d2_1_height_std, 2),
        "d2_2_building_density": round(d2_2_density, 4),
        "d2_3_roof_area_mean": round(d2_3_roof_area_mean, 1),
        "d2_3_roof_area_median": round(d2_3_roof_area_median, 1),
        "d2_3_roof_area_total_m2": round(d2_3_roof_area_total, 0),
        "d2_4_compactness_mean": round(d2_4_compactness_mean, 3),
        "d2_4_compactness_median": round(d2_4_compactness_median, 3),
        "d2_5_far": round(d2_5_far, 3),
    }


# ============================================================================
# 3. D3 技术部署维度指标计算（Proxy Scoring）
# ============================================================================

def compute_roof_utilization(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    估算每栋建筑的屋顶可利用率。

    考虑因素：
    - 屋顶设备区域（电梯机房、水箱、空调外机等）
    - 安全退缩距离
    - 屋顶形状不规则区域

    规则（基于经验值和文献）：
    - 面积 < 50 m²: 利用率 0.3（太小，难以安装）
    - 面积 50-200 m²: 利用率 0.5
    - 面积 200-500 m²: 利用率 0.6
    - 面积 500-1000 m²: 利用率 0.65
    - 面积 > 1000 m²: 利用率 0.7（大屋顶，规模效应）
    """
    areas = gdf["footprint_area_m2"]

    conditions = [
        areas < 50,
        (areas >= 50) & (areas < 200),
        (areas >= 200) & (areas < 500),
        (areas >= 500) & (areas < 1000),
        areas >= 1000,
    ]
    ratios = [0.30, 0.50, 0.60, 0.65, 0.70]

    return pd.Series(np.select(conditions, ratios, default=0.5), index=gdf.index)


def compute_shading_proxy(gdf: gpd.GeoDataFrame, search_radius: float = 50.0) -> pd.Series:
    """
    Proxy-based 遮挡损失评估。

    方法：基于周边建筑的高度差和距离，估算遮挡对屋顶太阳辐射的影响。
    这是一个简化的 proxy，不是真实的光线追踪，但能捕捉形态对遮挡的主要影响。

    遮挡因子 = 1 - shading_loss (0=完全遮挡, 1=无遮挡)

    算法：
    1. 对每栋建筑，搜索半径内的所有邻居
    2. 计算每个比目标更高的邻居的遮挡角 = atan(Δh / distance)
    3. 遮挡损失 ∝ max(遮挡角) / (π/2)
    4. 最终遮挡因子 = 1 - 遮挡损失系数

    Parameters
    ----------
    gdf : GeoDataFrame (projected CRS)
        必须包含 'height_m' 列
    search_radius : float
        搜索半径（米），默认 50m

    Returns
    -------
    Series of float in [0, 1]: 1=无遮挡, 0=完全遮挡
    """
    gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())

    # 建筑质心坐标
    centroids = np.column_stack([
        gdf_proj.geometry.centroid.x,
        gdf_proj.geometry.centroid.y,
    ])
    heights = gdf["height_m"].values

    # 构建 KD-Tree 加速邻近搜索
    tree = cKDTree(centroids)

    shading_factors = np.ones(len(gdf))

    for i in range(len(gdf)):
        # 搜索半径内的邻居
        neighbor_indices = tree.query_ball_point(centroids[i], search_radius)
        neighbor_indices = [j for j in neighbor_indices if j != i]

        if not neighbor_indices:
            continue

        h_self = heights[i]
        max_shade_angle = 0.0

        for j in neighbor_indices:
            h_neighbor = heights[j]
            delta_h = h_neighbor - h_self  # 邻居比自己高多少

            if delta_h <= 0:
                continue  # 比自己矮的不遮挡

            dist = np.linalg.norm(centroids[i] - centroids[j])
            if dist < 1.0:
                dist = 1.0  # 避免除以零

            shade_angle = np.arctan(delta_h / dist)
            max_shade_angle = max(max_shade_angle, shade_angle)

        # 遮挡损失：最大遮挡角归一化到 [0, 1]
        # 30° 以上的遮挡角认为损失很严重
        critical_angle = np.radians(30)
        shading_loss = min(max_shade_angle / critical_angle, 1.0) * 0.5
        # 最大损失限制在 0.5（即使完全被遮挡也有散射辐射）

        shading_factors[i] = 1.0 - shading_loss

    return pd.Series(shading_factors, index=gdf.index)


def compute_d3_indicators(gdf: gpd.GeoDataFrame) -> Dict:
    """
    计算 D3 维度子指标。

    子指标：
      d3_1: 可部署屋顶面积比 (deployable ratio)
      d3_2: 遮挡损失系数 (shading factor, 1=无遮挡)
      d3_3: 有效部署面积 = footprint × utilization × shading_factor
      d3_4: 总可部署面积 (m²)
    """
    log.info(f"  计算 D3 指标...")

    # ── d3_1: 屋顶利用率 ──
    roof_util = compute_roof_utilization(gdf)
    gdf["roof_utilization"] = roof_util.values

    # ── d3_2: 遮挡因子 ──
    log.info(f"  计算遮挡因子（proxy-based, 可能耗时较长）...")
    n_buildings = len(gdf)
    if n_buildings > 10000:
        log.info(f"  建筑数量 {n_buildings:,} > 10000, 使用抽样估计遮挡")
        # 对大数据集，先对 2000 栋抽样计算，然后按形态类型回填均值
        sample_idx = np.random.choice(gdf.index, size=min(2000, n_buildings), replace=False)
        gdf_sample = gdf.loc[sample_idx]
        shading_sample = compute_shading_proxy(gdf_sample)

        # 按形态类型计算平均遮挡因子
        gdf.loc[sample_idx, "shading_factor_sample"] = shading_sample.values
        gdf["_temp_shading"] = np.nan
        gdf.loc[sample_idx, "_temp_shading"] = shading_sample.values

        if "typology" in gdf.columns:
            type_means = gdf.groupby("typology")["_temp_shading"].mean()
            shading_all = gdf["typology"].map(type_means)
            # 有抽样值的用实际值
            shading_all.update(gdf["_temp_shading"].dropna())
        else:
            shading_all = gdf["_temp_shading"].fillna(shading_sample.mean())

        gdf.drop(columns=["_temp_shading", "shading_factor_sample"],
                 errors="ignore", inplace=True)
        shading_factor = shading_all.fillna(0.85)  # 安全默认值
    else:
        shading_factor = compute_shading_proxy(gdf)

    gdf["shading_factor"] = shading_factor.values

    # ── d3_3: 有效部署面积 ──
    effective_area = gdf["footprint_area_m2"] * gdf["roof_utilization"] * gdf["shading_factor"]
    gdf["effective_pv_area_m2"] = effective_area.values

    # ── 汇总 ──
    d3_1_util_mean = roof_util.mean()
    d3_2_shading_mean = gdf["shading_factor"].mean()
    d3_3_effective_area_mean = effective_area.mean()
    d3_4_total_deployable = effective_area.sum()

    return {
        "d3_1_roof_utilization_mean": round(d3_1_util_mean, 3),
        "d3_2_shading_factor_mean": round(d3_2_shading_mean, 3),
        "d3_3_effective_area_mean_m2": round(d3_3_effective_area_mean, 1),
        "d3_4_total_deployable_m2": round(d3_4_total_deployable, 0),
        "d3_4_total_deployable_mw": round(
            d3_4_total_deployable * 0.20 / 1000, 2  # 假设 200W/m² 功率密度
        ),
    }


# ============================================================================
# 4. 形态类型统计
# ============================================================================

def compute_typology_stats(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """按形态类型分组统计。"""
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

    # 排序
    type_order = ["low_rise", "mid_rise", "mid_high", "high_rise"]
    stats = stats.reindex([t for t in type_order if t in stats.index])

    return stats


# ============================================================================
# 5. 主流程：单城市处理
# ============================================================================

def process_city(
    city_key: str,
    city_info: dict,
    from_gpkg: bool = False,
) -> Dict:
    """处理单个城市的完整形态分析流程。"""

    log.info(f"\n{'='*60}")
    log.info(f"处理: {city_info['name_en']} ({city_info['name_cn']}) — {city_info['climate_zone']}")
    log.info(f"{'='*60}")

    t0 = time.time()

    # ── 1. 获取数据 ──
    gdf = None
    if from_gpkg:
        gdf = load_from_gpkg(city_key)

    if gdf is None:
        gdf = fetch_buildings(city_key, city_info)

    if len(gdf) == 0:
        log.error(f"  没有建筑数据!")
        return {"city": city_key, "status": "NO_DATA"}

    # ── 2. 筛选住宅 ──
    is_res = classify_residential(gdf)
    n_res = is_res.sum()
    n_all = len(gdf)
    log.info(f"  住宅建筑: {n_res:,} / {n_all:,} ({n_res/n_all:.1%})")

    # 如果住宅标签太少（<10%），用 building=yes 的也纳入
    # （很多中国城市的 OSM 数据 building 标签不细分）
    if n_res / n_all < 0.10:
        log.warning(f"  ⚠ 住宅标签比例过低，将 building=yes 也纳入分析")
        if "building" in gdf.columns:
            is_res |= gdf["building"] == "yes"
            n_res = is_res.sum()
            log.info(f"  扩展后: {n_res:,} / {n_all:,} ({n_res/n_all:.1%})")

    # 如果仍然太少，用全部建筑
    if n_res < 100:
        log.warning(f"  ⚠ 住宅建筑不足100栋，使用全部建筑进行分析")
        gdf_res = gdf.copy()
    else:
        gdf_res = gdf[is_res].copy()

    log.info(f"  分析建筑数: {len(gdf_res):,}")

    # ── 3. 计算基础属性 ──
    # 高度
    gdf_res["height_m"] = infer_height(gdf_res, city_info["default_floor_height"]).values
    gdf_res["n_floors"] = (gdf_res["height_m"] / city_info["default_floor_height"]).round().clip(lower=1).astype(int)

    # 面积（投影到 UTM 计算）
    gdf_proj = gdf_res.to_crs(gdf_res.estimate_utm_crs())
    gdf_res["footprint_area_m2"] = gdf_proj.geometry.area.values

    # ── 4. 形态分类 ──
    gdf_res["typology"] = classify_typology(gdf_res["height_m"], city_info["default_floor_height"]).values
    type_counts = gdf_res["typology"].value_counts()
    log.info(f"  形态分类:\n{type_counts.to_string()}")

    # ── 5. 判断城市主导形态 ──
    dominant_type = type_counts.index[0]
    dominant_pct = type_counts.iloc[0] / len(gdf_res)
    if dominant_pct > 0.6:
        city_typology = f"{dominant_type}_dominant"
    else:
        city_typology = "mixed"
    log.info(f"  城市形态: {city_typology} ({dominant_type}: {dominant_pct:.1%})")

    # ── 6. D2 指标 ──
    d2 = compute_d2_indicators(gdf_res)
    log.info(f"  D2 指标计算完成")

    # ── 7. D3 指标 ──
    d3 = compute_d3_indicators(gdf_res)
    log.info(f"  D3 指标计算完成")

    # ── 8. 形态类型统计 ──
    typo_stats = compute_typology_stats(gdf_res)

    # ── 9. 保存结果 ──
    # 分类后的建筑数据
    save_cols = ["geometry", "height_m", "n_floors", "footprint_area_m2",
                 "typology", "roof_utilization", "shading_factor", "effective_pv_area_m2"]
    save_cols = [c for c in save_cols if c in gdf_res.columns]
    gpkg_path = RESULTS_DIR / f"{city_key}_buildings_classified.gpkg"
    try:
        gdf_res[save_cols].to_file(gpkg_path, driver="GPKG")
        log.info(f"  保存: {gpkg_path}")
    except Exception as e:
        log.warning(f"  保存 GPKG 失败: {e}")

    # D2 指标
    d2_df = pd.DataFrame([d2])
    d2_df.insert(0, "city", city_key)
    d2_path = RESULTS_DIR / f"{city_key}_d2_indicators.csv"
    d2_df.to_csv(d2_path, index=False)

    # D3 指标
    d3_df = pd.DataFrame([d3])
    d3_df.insert(0, "city", city_key)
    d3_path = RESULTS_DIR / f"{city_key}_d3_indicators.csv"
    d3_df.to_csv(d3_path, index=False)

    # 形态统计
    typo_path = RESULTS_DIR / f"{city_key}_typology_stats.csv"
    typo_stats.to_csv(typo_path)

    elapsed = time.time() - t0
    log.info(f"  耗时: {elapsed:.1f}s")

    # ── 汇总 ──
    result = {
        "city": city_key,
        "name_en": city_info["name_en"],
        "name_cn": city_info["name_cn"],
        "climate_zone": city_info["climate_zone"],
        "status": "OK",
        "n_buildings_analyzed": len(gdf_res),
        "city_typology": city_typology,
        "dominant_type": dominant_type,
        "dominant_pct": round(dominant_pct, 3),
        "elapsed_s": round(elapsed, 1),
        **d2,
        **d3,
    }

    return result


# ============================================================================
# 6. 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Paper 4 Step 3: 城市形态与技术部署指标计算"
    )
    parser.add_argument("--city", type=str, default=None,
                        help="只处理指定城市 (如 'beijing')")
    parser.add_argument("--from-gpkg", action="store_true",
                        help="优先从已有 GeoPackage 读取数据")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Paper 4 — Step 3: 城市形态与技术部署指标计算")
    log.info("=" * 60)

    # 确定要处理的城市
    cities = CITIES
    if args.city:
        key = args.city.lower()
        if key not in CITIES:
            log.error(f"未知城市: {args.city}. 可选: {list(CITIES.keys())}")
            sys.exit(1)
        cities = {key: CITIES[key]}

    # 逐城市处理
    all_results = []
    for i, (city_key, city_info) in enumerate(cities.items()):
        result = process_city(city_key, city_info, from_gpkg=args.from_gpkg)
        all_results.append(result)

        # API 友好间隔
        if i < len(cities) - 1 and not args.from_gpkg:
            log.info("  等待 15s...")
            time.sleep(15)

    # ── 汇总五城市对比 ──
    ok_results = [r for r in all_results if r["status"] == "OK"]
    if ok_results:
        summary_df = pd.DataFrame(ok_results)
        summary_path = RESULTS_DIR / "cross_city_d2d3_summary.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        log.info(f"\n汇总表保存: {summary_path}")

        # 打印关键对比
        log.info("\n" + "=" * 60)
        log.info("五城市 D2/D3 指标对比")
        log.info("=" * 60)

        display_cols = [
            "name_en", "climate_zone", "n_buildings_analyzed", "city_typology",
            "d2_1_height_mean", "d2_2_building_density", "d2_3_roof_area_mean",
            "d2_4_compactness_mean", "d2_5_far",
            "d3_1_roof_utilization_mean", "d3_2_shading_factor_mean",
            "d3_4_total_deployable_mw",
        ]
        display_cols = [c for c in display_cols if c in summary_df.columns]

        if HAS_TABULATE:
            print("\n" + tabulate(
                summary_df[display_cols],
                headers="keys",
                tablefmt="grid",
                showindex=False,
                floatfmt=".3f",
            ))
        else:
            print(summary_df[display_cols].to_string(index=False))

    failed = [r for r in all_results if r["status"] != "OK"]
    if failed:
        log.warning(f"\n{len(failed)} 城市处理失败:")
        for r in failed:
            log.warning(f"  {r['city']}")

    log.info("\nStep 3 完成。")
    log.info("下一步: python scripts/04_energy_simulation.py (D1, D3补充, D4, D5)")


if __name__ == "__main__":
    main()
