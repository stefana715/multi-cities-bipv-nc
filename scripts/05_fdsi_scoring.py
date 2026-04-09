#!/usr/bin/env python3
"""
============================================================================
Paper 4 - Script 05: FDSI Scoring & Suitability Matrix
============================================================================
Step 5: 整合 D1-D5 全部指标，运行 FDSI 综合评分，生成论文核心产出。

论文核心产出：
  1. 气候×形态适宜性矩阵（Table: Climate-Morphology Suitability Matrix）
  2. 五维雷达图（Figure: Five-City Radar Comparison）
  3. FDSI 综合排名表
  4. 权重敏感性分析图
  5. 维度得分热力图

工作流程：
  1. 汇总 D1-D5 全部子指标
  2. 归一化到 [0,1]
  3. 熵权法 + AHP + 组合赋权
  4. FDSI 综合评分
  5. 按形态类型交叉分析 → 适宜性矩阵
  6. 权重敏感性分析
  7. 生成全部图表

输出：
  results/fdsi/integrated_indicators.csv     — 汇总指标（原始值+归一化）
  results/fdsi/fdsi_scores.csv               — FDSI 综合评分
  results/fdsi/suitability_matrix.csv        — 气候×形态适宜性矩阵
  results/fdsi/weight_sensitivity.csv        — 权重敏感性分析
  results/fdsi/weight_comparison.csv         — 三种赋权结果对比
  figures/fig_radar_five_cities.png           — 五维雷达图
  figures/fig_suitability_matrix.png          — 适宜性矩阵热力图
  figures/fig_weight_sensitivity.png          — 权重敏感性分析图
  figures/fig_dimension_heatmap.png           — 维度得分热力图

用法：
  python scripts/05_fdsi_scoring.py
============================================================================
"""

import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    # 尝试设置中文字体
    for font_name in ["SimHei", "PingFang SC", "Heiti SC", "Microsoft YaHei", "WenQuanYi Micro Hei"]:
        if any(font_name in f.name for f in fm.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

warnings.filterwarnings("ignore")

# ── Setup ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MORPHOLOGY_DIR = PROJECT_DIR / "results" / "morphology"
ENERGY_DIR = PROJECT_DIR / "results" / "energy"
RESULTS_DIR = PROJECT_DIR / "results" / "fdsi"
FIGURES_DIR = PROJECT_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# 添加 src 到 path
sys.path.insert(0, str(PROJECT_DIR))


# ============================================================================
# 城市信息
# ============================================================================

CITY_META = {
    # ── Original 5 cities ──
    "harbin":    {"name_en": "Harbin",    "name_cn": "哈尔滨", "zone": "Severe Cold", "zone_cn": "严寒"},
    "beijing":   {"name_en": "Beijing",   "name_cn": "北京",   "zone": "Cold",        "zone_cn": "寒冷"},
    "changsha":  {"name_en": "Changsha",  "name_cn": "长沙",   "zone": "HSCW",        "zone_cn": "夏热冬冷"},
    "shenzhen":  {"name_en": "Shenzhen",  "name_cn": "深圳",   "zone": "HSWW",        "zone_cn": "夏热冬暖"},
    "kunming":   {"name_en": "Kunming",   "name_cn": "昆明",   "zone": "Mild",        "zone_cn": "温和"},
    # ── 10 new cities (Phase 2) ──
    "changchun": {"name_en": "Changchun", "name_cn": "长春",   "zone": "Severe Cold", "zone_cn": "严寒"},
    "shenyang":  {"name_en": "Shenyang",  "name_cn": "沈阳",   "zone": "Severe Cold", "zone_cn": "严寒"},
    "jinan":     {"name_en": "Jinan",     "name_cn": "济南",   "zone": "Cold",        "zone_cn": "寒冷"},
    "xian":      {"name_en": "Xian",      "name_cn": "西安",   "zone": "Cold",        "zone_cn": "寒冷"},
    "wuhan":     {"name_en": "Wuhan",     "name_cn": "武汉",   "zone": "HSCW",        "zone_cn": "夏热冬冷"},
    "nanjing":   {"name_en": "Nanjing",   "name_cn": "南京",   "zone": "HSCW",        "zone_cn": "夏热冬冷"},
    "guangzhou": {"name_en": "Guangzhou", "name_cn": "广州",   "zone": "HSWW",        "zone_cn": "夏热冬暖"},
    "xiamen":    {"name_en": "Xiamen",    "name_cn": "厦门",   "zone": "HSWW",        "zone_cn": "夏热冬暖"},
    "guiyang":   {"name_en": "Guiyang",   "name_cn": "贵阳",   "zone": "Mild",        "zone_cn": "温和"},
    "chengdu":   {"name_en": "Chengdu",   "name_cn": "成都",   "zone": "HSCW",        "zone_cn": "夏热冬冷"},  # corrected
    # ── 24 new cities (NC expansion) ──
    "dalian":        {"name_en": "Dalian",        "name_cn": "大连",   "zone": "Severe Cold", "zone_cn": "严寒"},
    "hohhot":        {"name_en": "Hohhot",        "name_cn": "呼和浩特","zone": "Severe Cold", "zone_cn": "严寒"},
    "tangshan":      {"name_en": "Tangshan",      "name_cn": "唐山",   "zone": "Severe Cold", "zone_cn": "严寒"},
    "urumqi":        {"name_en": "Urumqi",        "name_cn": "乌鲁木齐","zone": "Severe Cold", "zone_cn": "严寒"},
    "taiyuan":       {"name_en": "Taiyuan",       "name_cn": "太原",   "zone": "Cold",        "zone_cn": "寒冷"},
    "shijiazhuang":  {"name_en": "Shijiazhuang",  "name_cn": "石家庄", "zone": "Cold",        "zone_cn": "寒冷"},
    "lanzhou":       {"name_en": "Lanzhou",       "name_cn": "兰州",   "zone": "Cold",        "zone_cn": "寒冷"},
    "yinchuan":      {"name_en": "Yinchuan",      "name_cn": "银川",   "zone": "Cold",        "zone_cn": "寒冷"},
    "xining":        {"name_en": "Xining",        "name_cn": "西宁",   "zone": "Cold",        "zone_cn": "寒冷"},
    "qingdao":       {"name_en": "Qingdao",       "name_cn": "青岛",   "zone": "Cold",        "zone_cn": "寒冷"},
    "wuxi":          {"name_en": "Wuxi",          "name_cn": "无锡",   "zone": "Cold",        "zone_cn": "寒冷"},
    "suzhou":        {"name_en": "Suzhou",        "name_cn": "苏州",   "zone": "Cold",        "zone_cn": "寒冷"},
    "tianjin":       {"name_en": "Tianjin",       "name_cn": "天津",   "zone": "Cold",        "zone_cn": "寒冷"},
    "zhengzhou":     {"name_en": "Zhengzhou",     "name_cn": "郑州",   "zone": "Cold",        "zone_cn": "寒冷"},
    "hangzhou":      {"name_en": "Hangzhou",      "name_cn": "杭州",   "zone": "HSCW",        "zone_cn": "夏热冬冷"},
    "hefei":         {"name_en": "Hefei",         "name_cn": "合肥",   "zone": "HSCW",        "zone_cn": "夏热冬冷"},
    "nanchang":      {"name_en": "Nanchang",      "name_cn": "南昌",   "zone": "HSCW",        "zone_cn": "夏热冬冷"},
    "ningbo":        {"name_en": "Ningbo",        "name_cn": "宁波",   "zone": "HSCW",        "zone_cn": "夏热冬冷"},
    "shanghai":      {"name_en": "Shanghai",      "name_cn": "上海",   "zone": "HSCW",        "zone_cn": "夏热冬冷"},
    "chongqing":     {"name_en": "Chongqing",     "name_cn": "重庆",   "zone": "HSCW",        "zone_cn": "夏热冬冷"},
    "fuzhou":        {"name_en": "Fuzhou",        "name_cn": "福州",   "zone": "HSWW",        "zone_cn": "夏热冬暖"},
    "nanning":       {"name_en": "Nanning",       "name_cn": "南宁",   "zone": "HSWW",        "zone_cn": "夏热冬暖"},
    "haikou":        {"name_en": "Haikou",        "name_cn": "海口",   "zone": "HSWW",        "zone_cn": "夏热冬暖"},
    "lhasa":         {"name_en": "Lhasa",         "name_cn": "拉萨",   "zone": "Mild",        "zone_cn": "温和"},
    # ── 2 non-mainland cities (NC extension) ──
    "hongkong":      {"name_en": "Hong Kong",     "name_cn": "香港",   "zone": "HSWW",        "zone_cn": "夏热冬暖（参照）"},
    "taipei":        {"name_en": "Taipei",        "name_cn": "台北",   "zone": "HSCW",        "zone_cn": "夏热冬冷（参照）"},
}

CITY_ORDER = [
    # Severe Cold (严寒) — 7 cities
    "harbin", "changchun", "shenyang", "dalian", "hohhot", "tangshan", "urumqi",
    # Cold (寒冷) — 13 cities
    "beijing", "tianjin", "jinan", "xian", "taiyuan", "shijiazhuang", "zhengzhou",
    "qingdao", "lanzhou", "yinchuan", "xining", "wuxi", "suzhou",
    # HSCW (夏热冬冷) — 10 cities + Taipei
    "changsha", "wuhan", "nanjing", "chengdu", "hangzhou", "hefei",
    "nanchang", "ningbo", "shanghai", "chongqing", "taipei",
    # HSWW (夏热冬暖) — 6 cities + Hong Kong
    "shenzhen", "guangzhou", "xiamen", "fuzhou", "nanning", "haikou", "hongkong",
    # Mild (温和) — 3 cities
    "kunming", "guiyang", "lhasa",
]


# ============================================================================
# 1. 汇总全部指标
# ============================================================================

def load_all_indicators() -> pd.DataFrame:
    """从 Step 3 和 Step 4 的输出汇总 D1-D5 全部指标。"""

    # ── 形态数据 (D2, D3) ──
    morph_path = MORPHOLOGY_DIR / "cross_city_d2d3_summary.csv"
    if morph_path.exists():
        morph_df = pd.read_csv(morph_path)
        log.info(f"  D2/D3 数据: {morph_path}")
    else:
        log.error(f"  找不到形态数据: {morph_path}")
        sys.exit(1)

    # ── 能源数据 (D1, D4, D5) ──
    energy_path = ENERGY_DIR / "cross_city_d1d4d5.csv"
    if energy_path.exists():
        energy_df = pd.read_csv(energy_path)
        log.info(f"  D1/D4/D5 数据: {energy_path}")
    else:
        log.error(f"  找不到能源数据: {energy_path}")
        sys.exit(1)

    # ── 合并 ──
    merged = pd.merge(morph_df, energy_df, on="city", how="inner", suffixes=("_morph", "_energy"))

    log.info(f"  合并后城市数: {len(merged)}")
    return merged


def select_dimension_indicators(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    选择每个维度的代表性子指标用于 FDSI 评分。

    每个维度选 1-2 个最有代表性的指标，避免维度内部高度相关的指标同时入选。

    Returns: dict of {dimension: {indicator_name: {col, is_benefit, weight_within}}}
    """
    indicators = {
        "D1_Climate": {
            "GHI_annual": {
                "col": "d1_1_ghi_annual_kwh",
                "is_benefit": True,      # GHI 越高越好
                "weight_within": 0.6,
                "label": "Annual GHI (kWh/m²/yr)",
            },
            "GHI_stability": {
                "col": "d1_3_ghi_cv_seasonal",
                "is_benefit": False,     # CV 越小越稳定越好
                "weight_within": 0.4,
                "label": "GHI Seasonal CV",
            },
        },
        "D2_Morphology": {
            "roof_area": {
                "col": "d2_3_roof_area_mean",
                "is_benefit": True,      # 屋顶面积越大越好
                "weight_within": 0.5,
                "label": "Mean Roof Area (m²)",
            },
            "building_density": {
                "col": "d2_2_building_density",
                "is_benefit": False,     # 密度越高遮挡越严重
                "weight_within": 0.5,
                "label": "Building Density",
            },
        },
        "D3_Technical": {
            "specific_yield": {
                "col": "specific_yield_kwh_kwp",
                "is_benefit": True,      # 发电量越高越好
                "weight_within": 0.5,
                "label": "Specific Yield (kWh/kWp/yr)",
            },
            "shading_factor": {
                "col": "d3_2_shading_factor_mean",
                "is_benefit": True,      # 遮挡越少越好
                "weight_within": 0.5,
                "label": "Shading Factor",
            },
        },
        "D4_Economic": {
            "lcoe": {
                "col": "d4_1_lcoe_cny_kwh",
                "is_benefit": False,     # LCOE 越低越好
                "weight_within": 0.5,
                "label": "LCOE (CNY/kWh)",
            },
            "pbt": {
                "col": "d4_2_pbt_years",
                "is_benefit": False,     # PBT 越短越好
                "weight_within": 0.5,
                "label": "Payback (years)",
            },
        },
        "D5_Uncertainty": {
            "pbt_ci_width": {
                "col": "d5_2_pbt_ci95_width",
                "is_benefit": False,     # CI 越窄 → 越确定 → 越好
                "weight_within": 0.35,   # Phase 3: 0.40 → 0.35
                "label": "PBT 95% CI Width (yr)",
            },
            "lcoe_cv": {
                "col": "mc_lcoe_std",
                "is_benefit": False,     # LCOE 标准差越小 → 越确定 → 越好
                "weight_within": 0.30,   # Phase 3: 0.35 → 0.30
                "label": "LCOE Std Dev",
            },
            "elec_price_sensitivity": {
                "col": "sobol_pbt_S1_elec_price_factor",
                "is_benefit": False,     # 电价敏感性越低 → 受市场波动影响小 → 越好
                "weight_within": 0.35,   # Phase 3: 0.25 → 0.35，引入独立市场风险维度
                "label": "Sobol PBT Sensitivity to Electricity Price",
            },
        },
    }
    return indicators


# ============================================================================
# 2. 归一化 + 维度得分
# ============================================================================

def normalize_indicators(df: pd.DataFrame, indicators: Dict) -> pd.DataFrame:
    """对所有指标做 min-max 归一化到 [0, 1]。"""
    norm_data = {}

    for dim_name, sub_indicators in indicators.items():
        for ind_name, ind_info in sub_indicators.items():
            col = ind_info["col"]
            if col not in df.columns:
                log.warning(f"  ⚠ 列 '{col}' 不存在，跳过 ({dim_name}/{ind_name})")
                continue

            values = pd.to_numeric(df[col], errors="coerce")

            # NaN 安全处理：用中位数填充
            if values.isna().any():
                n_na = values.isna().sum()
                log.warning(f"  ⚠ {col}: {n_na} 个 NaN，用中位数填充")
                values = values.fillna(values.median())

            vmin, vmax = values.min(), values.max()

            if vmax == vmin:
                log.warning(f"  ⚠ {col}: 所有值相同 ({vmin})，归一化为 0.5")
                normalized = pd.Series(0.5, index=df.index)
            elif ind_info["is_benefit"]:
                normalized = (values - vmin) / (vmax - vmin)
            else:
                normalized = (vmax - values) / (vmax - vmin)

            norm_col = f"norm_{col}"
            norm_data[norm_col] = normalized.values
            log.info(f"    {dim_name}/{ind_name}: [{vmin:.4f} → {vmax:.4f}] "
                     f"({'↑benefit' if ind_info['is_benefit'] else '↓cost'})")

    norm_df = pd.DataFrame(norm_data, index=df.index)
    return norm_df


def compute_dimension_scores(norm_df: pd.DataFrame, df: pd.DataFrame,
                              indicators: Dict) -> pd.DataFrame:
    """计算每个维度的加权得分。"""
    dim_scores = {}

    for dim_name, sub_indicators in indicators.items():
        weighted_sum = np.zeros(len(norm_df))
        weight_total = 0

        for ind_name, ind_info in sub_indicators.items():
            norm_col = f"norm_{ind_info['col']}"
            if norm_col in norm_df.columns:
                w = ind_info["weight_within"]
                weighted_sum += norm_df[norm_col].values * w
                weight_total += w

        if weight_total > 0:
            dim_scores[dim_name] = weighted_sum / weight_total
        else:
            dim_scores[dim_name] = np.full(len(norm_df), 0.5)

    return pd.DataFrame(dim_scores, index=norm_df.index)


# ============================================================================
# 3. 赋权 + FDSI 评分
# ============================================================================

def entropy_weight(dim_scores: pd.DataFrame) -> np.ndarray:
    """熵权法（客观权重）。"""
    X = dim_scores.values.copy()
    n, m = X.shape

    # 归一化到 [0.001, 1]
    for j in range(m):
        col = X[:, j]
        cmin, cmax = col.min(), col.max()
        if cmax > cmin:
            X[:, j] = (col - cmin) / (cmax - cmin)
        else:
            X[:, j] = 1.0 / n
    X = np.clip(X, 0.001, None)

    # 比重
    P = X / X.sum(axis=0, keepdims=True)

    # 信息熵
    k = 1.0 / np.log(n)
    E = np.zeros(m)
    for j in range(m):
        pj = P[:, j]
        pj = pj[pj > 0]
        E[j] = -k * np.sum(pj * np.log(pj))

    # 差异系数 → 权重
    D = 1 - E
    w = D / D.sum() if D.sum() > 0 else np.ones(m) / m
    return w


def ahp_weight_d1d5() -> np.ndarray:
    """
    AHP 主观权重（五维度）。

    判断矩阵逻辑（论文中需说明）：
    - D5 确定性是核心贡献，权重最高
    - D1 气候和 D3 技术同等重要，是 BIPV 的物理基础
    - D2 形态和 D4 经济同等重要
    """
    A = np.array([
        [1,     2,    1,    2,    1/2],   # D1
        [1/2,   1,    1/2,  1,    1/3],   # D2
        [1,     2,    1,    2,    1/2],   # D3
        [1/2,   1,    1/2,  1,    1/3],   # D4
        [2,     3,    2,    3,    1  ],   # D5
    ])

    eigenvalues, eigenvectors = np.linalg.eig(A)
    idx = np.argmax(eigenvalues.real)
    w = eigenvectors[:, idx].real
    w = w / w.sum()

    # 一致性检验
    lambda_max = eigenvalues[idx].real
    n = len(A)
    CI = (lambda_max - n) / (n - 1)
    RI = {3: 0.58, 4: 0.90, 5: 1.12}
    CR = CI / RI.get(n, 1.12)
    log.info(f"  AHP: CR = {CR:.4f} ({'通过' if CR < 0.1 else '未通过'})")

    return w


def compute_fdsi(dim_scores: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """计算 FDSI = Σ(w_i × D_i)。"""
    return (dim_scores.values * weights).sum(axis=1)


# ============================================================================
# 4. 适宜性矩阵
# ============================================================================

def build_suitability_matrix(df: pd.DataFrame, fdsi: pd.Series,
                              dim_scores: pd.DataFrame) -> pd.DataFrame:
    """
    构建气候×形态适宜性矩阵。

    每个单元格包含：
    - 适宜性等级 (High/Medium/Low)
    - 不确定性等级 (High/Medium/Low)
    - 主导敏感因子
    """
    # 适宜性等级阈值
    fdsi_arr = fdsi.values
    q33 = np.percentile(fdsi_arr, 33)
    q67 = np.percentile(fdsi_arr, 67)

    def suitability_grade(score):
        if score >= q67:
            return "High"
        elif score >= q33:
            return "Medium"
        else:
            return "Low"

    # 不确定性等级（基于 D5 分数，D5越高=越确定=不确定性越低）
    d5 = dim_scores["D5_Uncertainty"].values
    d5_q33 = np.percentile(d5, 33)
    d5_q67 = np.percentile(d5, 67)

    def uncertainty_grade(d5_score):
        if d5_score >= d5_q67:
            return "Low"      # D5高 = 确定性高 = 不确定性低
        elif d5_score >= d5_q33:
            return "Medium"
        else:
            return "High"

    # 构建矩阵数据
    rows = []
    for i, city_key in enumerate(CITY_ORDER):
        if city_key not in df["city"].values:
            continue
        idx = df[df["city"] == city_key].index[0]
        meta = CITY_META[city_key]

        # 获取主导形态
        dominant = df.loc[idx, "dominant_type"] if "dominant_type" in df.columns else "mixed"

        # 获取 Sobol 主导因子
        dom_factor_yield = df.loc[idx].get("d5_5_dominant_factor_yield", "N/A")
        dom_factor_pbt = df.loc[idx].get("d5_5_dominant_factor_pbt", "N/A")

        fdsi_val = fdsi.loc[city_key]
        rows.append({
            "city": meta["name_en"],
            "city_cn": meta["name_cn"],
            "climate_zone": meta["zone"],
            "climate_zone_cn": meta["zone_cn"],
            "dominant_morphology": dominant,
            "fdsi_score": round(fdsi_val, 4),
            "fdsi_rank": 0,  # 填充在下面
            "suitability": suitability_grade(fdsi_val),
            "uncertainty": uncertainty_grade(dim_scores.loc[city_key, "D5_Uncertainty"]),
            "D1_score": round(dim_scores.loc[city_key, "D1_Climate"], 3),
            "D2_score": round(dim_scores.loc[city_key, "D2_Morphology"], 3),
            "D3_score": round(dim_scores.loc[city_key, "D3_Technical"], 3),
            "D4_score": round(dim_scores.loc[city_key, "D4_Economic"], 3),
            "D5_score": round(dim_scores.loc[city_key, "D5_Uncertainty"], 3),
            "dominant_factor_yield": dom_factor_yield,
            "dominant_factor_pbt": dom_factor_pbt,
        })

    matrix_df = pd.DataFrame(rows)
    matrix_df = matrix_df.sort_values("fdsi_score", ascending=False)
    matrix_df["fdsi_rank"] = range(1, len(matrix_df) + 1)

    return matrix_df


# ============================================================================
# 5. 图表生成
# ============================================================================

def plot_radar(dim_scores: pd.DataFrame, save_path: Path):
    """五维雷达图：五城市对比。"""
    if not HAS_MPL:
        return

    categories = ["D1\nClimate", "D2\nMorphology", "D3\nTechnical",
                   "D4\nEconomic", "D5\nUncertainty"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    colors = [
        "#E53935", "#C62828", "#B71C1C",   # Severe Cold (reds)
        "#1E88E5", "#1565C0", "#0D47A1",   # Cold (blues)
        "#43A047", "#2E7D32", "#1B5E20",   # HSCW (greens)
        "#FB8C00", "#E65100", "#BF360C",   # HSWW (oranges)
        "#8E24AA", "#6A1B9A", "#4A148C",   # Mild (purples)
    ]
    city_labels = [f"{CITY_META[c]['name_en']}\n({CITY_META[c]['zone']})" for c in CITY_ORDER]

    # 生成足够颜色（与 CITY_ORDER 等长）
    import itertools
    _palette = colors + [
        "#F06292", "#AD1457", "#880E4F",   # extra pinks
        "#26C6DA", "#00838F", "#006064",   # extra cyans
        "#DCE775", "#9E9D24", "#827717",   # extra yellows
        "#A5D6A7", "#388E3C", "#1B5E20",   # extra greens
        "#CE93D8", "#7B1FA2", "#4A148C",   # extra purples
        "#FFCC80", "#E65100", "#BF360C",   # extra oranges
        "#90CAF9", "#1565C0", "#0D47A1",   # extra blues
    ]
    _colors_cycle = list(itertools.islice(itertools.cycle(_palette), len(CITY_ORDER)))

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for i, city_key in enumerate(CITY_ORDER):
        if city_key not in dim_scores.index:
            continue
        values = dim_scores.loc[city_key].values.tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2.5, color=_colors_cycle[i],
                label=city_labels[i], markersize=8)
        ax.fill(angles, values, alpha=0.08, color=_colors_cycle[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9, color="grey")
    ax.yaxis.grid(True, color="lightgrey", linestyle="--", alpha=0.7)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=11,
              frameon=True, framealpha=0.9)

    ax.set_title("Five-Dimension BIPV Suitability Radar\nAcross China's Climate Zones",
                 fontsize=15, fontweight="bold", pad=30)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  保存雷达图: {save_path}")


def plot_dimension_heatmap(dim_scores: pd.DataFrame, save_path: Path):
    """维度得分热力图。"""
    if not HAS_MPL or not HAS_SNS:
        return

    labels_y = [f"{CITY_META[c]['name_en']} ({CITY_META[c]['zone']})" for c in CITY_ORDER]
    labels_x = ["D1\nClimate", "D2\nMorphology", "D3\nTechnical",
                "D4\nEconomic", "D5\nUncertainty"]

    fig, ax = plt.subplots(figsize=(10, 5))
    data = dim_scores.values

    sns.heatmap(data, annot=True, fmt=".3f", cmap="RdYlGn",
                xticklabels=labels_x, yticklabels=labels_y,
                vmin=0, vmax=1, linewidths=1, linecolor="white",
                cbar_kws={"label": "Dimension Score"},
                ax=ax)

    ax.set_title("Dimension Scores Across Five Cities", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  保存热力图: {save_path}")


def plot_suitability_matrix_fig(matrix_df: pd.DataFrame, save_path: Path):
    """适宜性矩阵可视化。"""
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(12, 4))

    suit_colors = {"High": "#4CAF50", "Medium": "#FFC107", "Low": "#F44336"}
    uncert_markers = {"Low": "o", "Medium": "s", "High": "^"}

    for i, (_, row) in enumerate(matrix_df.iterrows()):
        color = suit_colors.get(row["suitability"], "grey")
        marker = uncert_markers.get(row["uncertainty"], "o")

        ax.barh(i, row["fdsi_score"], color=color, alpha=0.8, height=0.6,
                edgecolor="white", linewidth=1.5)

        label = f"{row['city']} ({row['climate_zone']})"
        ax.text(row["fdsi_score"] + 0.01, i,
                f"{row['fdsi_score']:.3f}  [{row['suitability']}/{row['uncertainty']} uncert.]",
                va="center", fontsize=10)

    ax.set_yticks(range(len(matrix_df)))
    ax.set_yticklabels([f"#{row['fdsi_rank']} {row['city']}\n({row['climate_zone']})"
                        for _, row in matrix_df.iterrows()], fontsize=11)
    ax.set_xlabel("FDSI Score", fontsize=12)
    ax.set_title("BIPV Suitability Ranking (FDSI)\nwith Suitability & Uncertainty Grades",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.0)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", label="High Suitability"),
        Patch(facecolor="#FFC107", label="Medium Suitability"),
        Patch(facecolor="#F44336", label="Low Suitability"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  保存适宜性矩阵图: {save_path}")


def plot_weight_sensitivity(sensitivity_df: pd.DataFrame, save_path: Path):
    """权重敏感性分析图。"""
    if not HAS_MPL:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    # Dynamic color palette scaled to number of cities
    import matplotlib.cm as _cm
    _zone_cmaps2 = {
        "Severe Cold": _cm.Reds,
        "Cold": _cm.Blues,
        "HSCW": _cm.Greens,
        "HSWW": _cm.Oranges,
        "Mild": _cm.Purples,
    }
    _zc2: dict = {}
    for _c2 in CITY_ORDER:
        _z2 = CITY_META[_c2]["zone"]
        _zc2[_z2] = _zc2.get(_z2, 0) + 1
    _zi2: dict = {_z2: 0 for _z2 in _zc2}
    colors_ws = []
    for _c2 in CITY_ORDER:
        _z2 = CITY_META[_c2]["zone"]
        _frac2 = 0.4 + 0.5 * (_zi2[_z2] / max(_zc2[_z2] - 1, 1))
        colors_ws.append(_zone_cmaps2[_z2](_frac2))
        _zi2[_z2] += 1
    city_labels = [CITY_META[c]["name_en"] for c in CITY_ORDER]

    # Panel 1: FDSI score vs alpha
    for i, city in enumerate(city_labels):
        city_data = sensitivity_df[sensitivity_df["object"] == city]
        if not city_data.empty:
            ax1.plot(city_data["alpha"], city_data["fdsi_score"],
                     "o-", color=colors_ws[i], label=city, linewidth=2, markersize=5)

    ax1.set_xlabel("α (Objective Weight Ratio)", fontsize=12)
    ax1.set_ylabel("FDSI Score", fontsize=12)
    ax1.set_title("(a) FDSI Score Sensitivity to α", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Rank vs alpha
    for i, city in enumerate(city_labels):
        city_data = sensitivity_df[sensitivity_df["object"] == city]
        if not city_data.empty:
            ax2.plot(city_data["alpha"], city_data["rank"],
                     "o-", color=colors_ws[i], label=city, linewidth=2, markersize=5)

    ax2.set_xlabel("α (Objective Weight Ratio)", fontsize=12)
    ax2.set_ylabel("Rank", fontsize=12)
    ax2.set_title("(b) Rank Stability Across α", fontsize=13, fontweight="bold")
    ax2.invert_yaxis()
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  保存敏感性分析图: {save_path}")


# ============================================================================
# 6. 主流程
# ============================================================================

def main():
    log.info("=" * 60)
    log.info("Paper 4 — Step 5: FDSI 综合评分与适宜性矩阵")
    log.info("=" * 60)

    # ── 1. 汇总指标 ──
    log.info("\n[1] 汇总 D1-D5 全部指标...")
    df = load_all_indicators()

    # 确保城市顺序一致
    df["city_order"] = df["city"].map({c: i for i, c in enumerate(CITY_ORDER)})
    df = df.sort_values("city_order").reset_index(drop=True)

    # ── 2. 选择代表性指标 ──
    indicators = select_dimension_indicators(df)

    # 检查列是否存在
    log.info("\n  指标可用性检查:")
    for dim, subs in indicators.items():
        for name, info in subs.items():
            exists = info["col"] in df.columns
            val = df[info["col"]].values if exists else "MISSING"
            log.info(f"    {dim}/{name}: {info['col']} — {'✓' if exists else '✗ MISSING'}")
            if exists:
                log.info(f"      values: {df[info['col']].values}")

    # ── 3. 归一化 ──
    log.info("\n[2] 归一化...")
    norm_df = normalize_indicators(df, indicators)

    # ── 4. 维度得分 ──
    log.info("\n[3] 计算维度得分...")
    dim_scores = compute_dimension_scores(norm_df, df, indicators)
    dim_scores.index = df["city"].values

    log.info("  维度得分:")
    log.info(f"\n{dim_scores.round(3).to_string()}")

    # ── 5. 赋权 ──
    log.info("\n[4] 三种赋权方法...")

    # 熵权法
    w_entropy = entropy_weight(dim_scores)
    log.info(f"  熵权法:  {dict(zip(dim_scores.columns, w_entropy.round(4)))}")

    # AHP
    w_ahp = ahp_weight_d1d5()
    log.info(f"  AHP:     {dict(zip(dim_scores.columns, w_ahp.round(4)))}")

    # 组合赋权
    alpha = 0.5
    w_combined = alpha * w_entropy + (1 - alpha) * w_ahp
    w_combined = w_combined / w_combined.sum()
    log.info(f"  组合(α={alpha}): {dict(zip(dim_scores.columns, w_combined.round(4)))}")

    # 保存权重对比
    weight_df = pd.DataFrame({
        "dimension": dim_scores.columns,
        "w_entropy": w_entropy.round(4),
        "w_ahp": w_ahp.round(4),
        "w_combined": w_combined.round(4),
    })
    weight_df.to_csv(RESULTS_DIR / "weight_comparison.csv", index=False)

    # ── 6. FDSI 评分 ──
    log.info("\n[5] FDSI 综合评分...")
    fdsi = pd.Series(compute_fdsi(dim_scores, w_combined), index=dim_scores.index)

    actual_cities = list(fdsi.index)  # city-name strings in sort order
    fdsi_df = pd.DataFrame({
        "city": actual_cities,
        "name_en": [CITY_META[c]["name_en"] for c in actual_cities],
        "climate_zone": [CITY_META[c]["zone"] for c in actual_cities],
        "fdsi_score": fdsi.values.round(4),
    })
    fdsi_df = fdsi_df.sort_values("fdsi_score", ascending=False)
    fdsi_df["rank"] = range(1, len(fdsi_df) + 1)

    log.info("\n  FDSI 排名:")
    log.info(f"\n{fdsi_df.to_string(index=False)}")

    fdsi_df.to_csv(RESULTS_DIR / "fdsi_scores.csv", index=False)

    # ── 7. 适宜性矩阵 ──
    log.info("\n[6] 构建适宜性矩阵...")
    matrix_df = build_suitability_matrix(df, fdsi, dim_scores)

    log.info("\n  气候×形态适宜性矩阵:")
    display_cols = ["fdsi_rank", "city", "climate_zone", "dominant_morphology",
                    "fdsi_score", "suitability", "uncertainty",
                    "dominant_factor_yield", "dominant_factor_pbt"]
    log.info(f"\n{matrix_df[display_cols].to_string(index=False)}")

    matrix_df.to_csv(RESULTS_DIR / "suitability_matrix.csv", index=False)

    # ── 8. 权重敏感性分析 ──
    log.info("\n[7] 权重敏感性分析...")
    alpha_range = np.arange(0.0, 1.05, 0.1)
    sens_records = []

    for a in alpha_range:
        w = a * w_entropy + (1 - a) * w_ahp
        w = w / w.sum()
        scores = compute_fdsi(dim_scores, w)
        ranks = (-np.array(scores)).argsort().argsort() + 1

        for i, city in enumerate(CITY_ORDER[:len(scores)]):
            sens_records.append({
                "alpha": round(a, 2),
                "object": CITY_META[city]["name_en"],
                "fdsi_score": round(scores[i], 4),
                "rank": int(ranks[i]),
            })

    sens_df = pd.DataFrame(sens_records)
    sens_df.to_csv(RESULTS_DIR / "weight_sensitivity.csv", index=False)

    # 排名稳定性摘要
    log.info("\n  排名稳定性:")
    for city in [CITY_META[c]["name_en"] for c in CITY_ORDER]:
        city_ranks = sens_df[sens_df["object"] == city]["rank"]
        log.info(f"    {city:10s}: rank range [{city_ranks.min()}-{city_ranks.max()}], "
                 f"std={city_ranks.std():.2f}")

    # ── 9. 保存汇总指标表 ──
    integrated = df.copy()
    for col in dim_scores.columns:
        integrated[f"score_{col}"] = dim_scores[col].values
    integrated["fdsi_score"] = fdsi.values
    integrated.to_csv(RESULTS_DIR / "integrated_indicators.csv", index=False, encoding="utf-8-sig")

    # ── 10. 生成图表 ──
    log.info("\n[8] 生成图表...")

    plot_radar(dim_scores, FIGURES_DIR / "fig_radar_five_cities.png")
    plot_dimension_heatmap(dim_scores, FIGURES_DIR / "fig_dimension_heatmap.png")
    plot_suitability_matrix_fig(matrix_df, FIGURES_DIR / "fig_suitability_matrix.png")
    plot_weight_sensitivity(sens_df, FIGURES_DIR / "fig_weight_sensitivity.png")

    # ── 完成 ──
    log.info("\n" + "=" * 60)
    log.info("Step 5 完成！")
    log.info("=" * 60)
    log.info(f"\n输出文件:")
    log.info(f"  {RESULTS_DIR}/")
    for f in sorted(RESULTS_DIR.glob("*.csv")):
        log.info(f"    {f.name}")
    log.info(f"  {FIGURES_DIR}/")
    for f in sorted(FIGURES_DIR.glob("*.png")):
        log.info(f"    {f.name}")

    log.info(f"\n论文核心产出已生成:")
    log.info(f"  1. 适宜性矩阵:     {RESULTS_DIR / 'suitability_matrix.csv'}")
    log.info(f"  2. FDSI 排名:       {RESULTS_DIR / 'fdsi_scores.csv'}")
    log.info(f"  3. 五维雷达图:      {FIGURES_DIR / 'fig_radar_five_cities.png'}")
    log.info(f"  4. 权重敏感性分析:  {FIGURES_DIR / 'fig_weight_sensitivity.png'}")


if __name__ == "__main__":
    main()
