#!/usr/bin/env python3
"""
============================================================================
NC Phase 3: 政策情景分析 — 4 情景 × 39 城市 = 156 FDSI 评分
============================================================================
情景设计：
  baseline      — 2024 现状
  cost_reduction — PV 成本降低 50%（2030 预测）
  carbon_pricing — 碳价 100 CNY/tCO₂
  aggressive     — 组合情景（成本减半 + 碳价 + 补贴 0.10 CNY/kWh）

关键原则：
  • D1（气候）、D2（形态）、D3（技术）在所有情景下不变
  • D4（经济性）和 D5（不确定性）随政策参数变化
  • 归一化跨所有 4×39=156 个 D4/D5 值（保证情景间可比性）
  • D1/D2/D3 使用 baseline 39 城市的归一化范围（固定）

输出：
  results/scenarios/scenario_fdsi_matrix.csv      — 156 行，FDSI+等级
  results/scenarios/suitability_transitions.csv   — 跳变矩阵
  results/scenarios/scenario_d4_detail.csv        — D4 详细数据
  figures/fig_scenario_fdsi_heatmap.png
  figures/fig_scenario_transitions.png
  figures/fig_scenario_rank_bump.png
============================================================================
"""

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_DIR   = Path(__file__).resolve().parent.parent
RESULTS_DIR   = PROJECT_DIR / "results" / "scenarios"
MORPH_DIR     = PROJECT_DIR / "results" / "morphology"
ENERGY_DIR    = PROJECT_DIR / "results" / "energy"
FDSI_DIR      = PROJECT_DIR / "results" / "fdsi"
FIGURES_DIR   = PROJECT_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

# ============================================================================
# 城市元数据：省份 → 电网区域 → 碳排放因子
# ============================================================================

# 2022 年中国区域电网基准线排放因子（tCO₂/MWh）
# 来源：生态环境部《中国区域电网基准线排放因子》(2023)
GRID_EMISSION_FACTORS = {
    "东北":  0.8437,   # 黑龙江、吉林、辽宁
    "华北":  0.8843,   # 北京、天津、河北、山西、山东、内蒙古
    "华东":  0.7035,   # 上海、江苏、浙江、安徽、福建
    "华中":  0.5257,   # 湖北、湖南、河南、江西、四川、重庆
    "西北":  0.6671,   # 陕西、甘肃、宁夏、青海、新疆
    "南方":  0.4267,   # 广东、广西、云南、贵州、海南
    "西藏":  0.1200,   # 可再生能源为主
}

# 城市 → 省份 → 电网区域 映射
CITY_PROVINCE_GRID = {
    "harbin":       ("黑龙江", "东北"),
    "changchun":    ("吉林",   "东北"),
    "shenyang":     ("辽宁",   "东北"),
    "dalian":       ("辽宁",   "东北"),
    "hohhot":       ("内蒙古", "华北"),
    "tangshan":     ("河北",   "华北"),
    "urumqi":       ("新疆",   "西北"),
    "beijing":      ("北京",   "华北"),
    "tianjin":      ("天津",   "华北"),
    "jinan":        ("山东",   "华北"),
    "xian":         ("陕西",   "西北"),
    "taiyuan":      ("山西",   "华北"),
    "shijiazhuang": ("河北",   "华北"),
    "zhengzhou":    ("河南",   "华中"),
    "qingdao":      ("山东",   "华北"),
    "lanzhou":      ("甘肃",   "西北"),
    "yinchuan":     ("宁夏",   "西北"),
    "xining":       ("青海",   "西北"),
    "wuxi":         ("江苏",   "华东"),
    "suzhou":       ("江苏",   "华东"),
    "changsha":     ("湖南",   "华中"),
    "wuhan":        ("湖北",   "华中"),
    "nanjing":      ("江苏",   "华东"),
    "chengdu":      ("四川",   "华中"),
    "hangzhou":     ("浙江",   "华东"),
    "hefei":        ("安徽",   "华东"),
    "nanchang":     ("江西",   "华中"),
    "ningbo":       ("浙江",   "华东"),
    "shanghai":     ("上海",   "华东"),
    "chongqing":    ("重庆",   "华中"),
    "shenzhen":     ("广东",   "南方"),
    "guangzhou":    ("广东",   "南方"),
    "xiamen":       ("福建",   "华东"),
    "fuzhou":       ("福建",   "华东"),
    "nanning":      ("广西",   "南方"),
    "haikou":       ("海南",   "南方"),
    "kunming":      ("云南",   "南方"),
    "guiyang":      ("贵州",   "南方"),
    "lhasa":        ("西藏",   "西藏"),
}

# 城市基础电价（CNY/kWh）— 与 04_energy_simulation.py 一致
CITY_ELEC_PRICE = {
    "harbin": 0.51, "changchun": 0.52, "shenyang": 0.53, "dalian": 0.50,
    "hohhot": 0.496, "tangshan": 0.511, "urumqi": 0.40,
    "beijing": 0.4883, "tianjin": 0.49, "jinan": 0.547, "xian": 0.498,
    "taiyuan": 0.452, "shijiazhuang": 0.511, "zhengzhou": 0.56,
    "qingdao": 0.547, "lanzhou": 0.476, "yinchuan": 0.44, "xining": 0.388,
    "wuxi": 0.558, "suzhou": 0.558,
    "changsha": 0.588, "wuhan": 0.558, "nanjing": 0.558, "chengdu": 0.502,
    "hangzhou": 0.538, "hefei": 0.561, "nanchang": 0.60,
    "ningbo": 0.538, "shanghai": 0.617, "chongqing": 0.522,
    "shenzhen": 0.68, "guangzhou": 0.68, "xiamen": 0.618,
    "fuzhou": 0.618, "nanning": 0.618, "haikou": 0.598,
    "kunming": 0.45, "guiyang": 0.468, "lhasa": 0.368,
}

# PV 系统固定参数
PV_PARAMS = {
    "discount_rate":      0.06,
    "system_lifetime":    25,
    "annual_degradation": 0.005,
    "om_ratio":           0.01,
}

# 情景定义
SCENARIOS = {
    "baseline": {
        "label":          "Baseline (2024)",
        "pv_cost_mode":   3.0,
        "pv_cost_left":   2.5,
        "pv_cost_right":  4.0,
        "carbon_price":   0,
        "subsidy":        0.0,
    },
    "cost_reduction": {
        "label":          "PV Cost −50% (2030)",
        "pv_cost_mode":   1.5,
        "pv_cost_left":   1.2,
        "pv_cost_right":  2.0,
        "carbon_price":   0,
        "subsidy":        0.0,
    },
    "carbon_pricing": {
        "label":          "Carbon Price 100 CNY/tCO₂",
        "pv_cost_mode":   3.0,
        "pv_cost_left":   2.5,
        "pv_cost_right":  4.0,
        "carbon_price":   100,
        "subsidy":        0.0,
    },
    "aggressive": {
        "label":          "Aggressive Policy Package",
        "pv_cost_mode":   1.5,
        "pv_cost_left":   1.2,
        "pv_cost_right":  2.0,
        "carbon_price":   100,
        "subsidy":        0.10,
    },
}

CITY_ORDER = [
    "harbin", "changchun", "shenyang", "dalian", "hohhot", "tangshan", "urumqi",
    "beijing", "tianjin", "jinan", "xian", "taiyuan", "shijiazhuang", "zhengzhou",
    "qingdao", "lanzhou", "yinchuan", "xining", "wuxi", "suzhou",
    "changsha", "wuhan", "nanjing", "chengdu", "hangzhou", "hefei",
    "nanchang", "ningbo", "shanghai", "chongqing",
    "shenzhen", "guangzhou", "xiamen", "fuzhou", "nanning", "haikou",
    "kunming", "guiyang", "lhasa",
]


# ============================================================================
# 1. 加载基础数据
# ============================================================================

def load_baseline_data():
    """加载 baseline D1-D5 数据和形态数据。"""
    morph  = pd.read_csv(MORPH_DIR  / "cross_city_d2d3_summary.csv")
    energy = pd.read_csv(ENERGY_DIR / "cross_city_d1d4d5.csv")
    merged = pd.merge(morph, energy, on="city", how="inner")
    log.info(f"  加载基础数据: {len(merged)} 城市")
    return merged


# ============================================================================
# 2. D4/D5 重新计算（单城市，单情景）
# ============================================================================

def calc_d4(specific_yield: float, elec_price_eff: float,
             pv_cost_mode: float) -> dict:
    """计算 D4 经济性指标（确定性）。"""
    p = PV_PARAMS
    capex    = pv_cost_mode * 1000          # CNY/kWp
    r, n     = p["discount_rate"], p["system_lifetime"]
    deg      = p["annual_degradation"]
    om_annual = capex * p["om_ratio"]

    # 折现发电量
    pv_energy = sum(
        specific_yield * (1 - deg) ** t / (1 + r) ** t
        for t in range(1, n + 1)
    )
    pv_om = sum(om_annual / (1 + r) ** t for t in range(1, n + 1))
    lcoe  = (capex + pv_om) / pv_energy if pv_energy > 0 else 99.0

    annual_revenue = specific_yield * elec_price_eff
    annual_net     = annual_revenue - om_annual
    pbt  = capex / annual_net if annual_net > 0 else 99.0

    npv = -capex + sum(
        (specific_yield * (1 - deg) ** t * elec_price_eff - om_annual) / (1 + r) ** t
        for t in range(1, n + 1)
    )
    return {
        "d4_lcoe":   round(lcoe, 4),
        "d4_pbt":    round(pbt,  2),
        "d4_npv":    round(npv,  0),
        "npv_pos":   npv > 0,
    }


def calc_d5_mc(specific_yield: float, elec_price_eff: float,
               pv_cost_left: float, pv_cost_mode: float,
               pv_cost_right: float, n_samples: int = 5000) -> dict:
    """Monte Carlo 不确定性分析（D5）。使用 LHS 抽样。"""
    rng = np.random.default_rng(42)
    p = PV_PARAMS

    # 抽样参数
    from scipy.stats import qmc, triang, uniform, norm as sp_norm
    sampler = qmc.LatinHypercube(d=5, seed=42)
    u = sampler.random(n=n_samples)

    # 参数分布
    span    = pv_cost_right - pv_cost_left
    c_tri   = (pv_cost_mode - pv_cost_left) / span if span > 0 else 0.5
    pv_cost_s   = triang(c=c_tri, loc=pv_cost_left, scale=span).ppf(u[:, 0])
    mod_eff_s   = triang(c=0.5,  loc=0.18, scale=0.04).ppf(u[:, 1])
    sys_loss_s  = uniform(loc=0.10, scale=0.08).ppf(u[:, 2])
    elec_fac_s  = uniform(loc=0.90, scale=0.20).ppf(u[:, 3])
    deg_s       = triang(c=0.25, loc=0.004, scale=0.003).ppf(u[:, 4])

    pbt_arr  = np.zeros(n_samples)
    lcoe_arr = np.zeros(n_samples)

    for i in range(n_samples):
        eff_price_i = elec_price_eff * elec_fac_s[i]
        # 系统损耗→发电量折减
        yield_i = specific_yield * (1 - sys_loss_s[i]) / (1 - 0.14)
        # 效率调整（不影响面积比）
        capex_i = pv_cost_s[i] * 1000
        om_i    = capex_i * p["om_ratio"]
        net_i   = yield_i * eff_price_i - om_i
        pbt_arr[i]  = capex_i / net_i if net_i > 0 else 99.0

        pv_en   = sum(yield_i * (1 - deg_s[i]) ** t / (1 + p["discount_rate"]) ** t
                      for t in range(1, p["system_lifetime"] + 1))
        pv_om   = sum(om_i / (1 + p["discount_rate"]) ** t
                      for t in range(1, p["system_lifetime"] + 1))
        lcoe_arr[i] = (capex_i + pv_om) / pv_en if pv_en > 0 else 99.0

    pbt_clean = pbt_arr[pbt_arr < 99]
    if len(pbt_clean) > 10:
        ci_lo, ci_hi = np.percentile(pbt_clean, [2.5, 97.5])
        ci_width = ci_hi - ci_lo
    else:
        ci_width = np.nan

    return {
        "d5_pbt_ci_width":  round(ci_width, 2) if not np.isnan(ci_width) else None,
        "d5_lcoe_std":      round(lcoe_arr.std(), 4),
        "d5_prob_pbt_le15": round((pbt_arr <= 15).mean(), 4),
        "pbt_p50":          round(np.median(pbt_clean), 2) if len(pbt_clean) > 10 else None,
    }


# ============================================================================
# 3. 主情景计算循环
# ============================================================================

def run_all_scenarios(base_df: pd.DataFrame) -> pd.DataFrame:
    """对 4 个情景 × 39 个城市计算 D4/D5。"""
    records = []

    for scen_key, scen in SCENARIOS.items():
        log.info(f"\n{'='*60}")
        log.info(f"  情景: {scen['label']}")
        log.info(f"{'='*60}")

        for _, row in base_df.iterrows():
            city = row["city"]

            # 跳过不在 CITY_ORDER 的城市
            if city not in CITY_ORDER:
                continue

            # 电网碳排放因子
            _, grid_region = CITY_PROVINCE_GRID.get(city, ("未知", "华北"))
            ef_tco2_mwh = GRID_EMISSION_FACTORS.get(grid_region, 0.70)
            ef_kgco2_kwh = ef_tco2_mwh  # tCO₂/MWh = kgCO₂/kWh × 1000/1000 → same numerically

            # 碳价等效电价上调（CNY/kWh）
            carbon_uplift = scen["carbon_price"] * ef_kgco2_kwh / 1000.0

            # 有效电价
            base_price  = CITY_ELEC_PRICE.get(city, 0.55)
            elec_eff    = base_price + carbon_uplift + scen["subsidy"]

            # specific_yield from baseline
            sy = float(row["specific_yield_kwh_kwp"])

            # D4
            d4 = calc_d4(sy, elec_eff, scen["pv_cost_mode"])

            # D5 (n=5000 for speed; sufficient for CI)
            d5 = calc_d5_mc(
                sy, elec_eff,
                scen["pv_cost_left"], scen["pv_cost_mode"], scen["pv_cost_right"],
                n_samples=5000,
            )

            # 保留 D1/D2/D3 列（直接传递）
            records.append({
                "scenario":      scen_key,
                "scenario_label": scen["label"],
                "city":          city,
                "name_en":       row.get("name_en", city.title()),
                "climate_zone":  row.get("climate_zone", ""),
                "province":      CITY_PROVINCE_GRID.get(city, ("?", "?"))[0],
                "grid_region":   grid_region,
                "ef_tco2_mwh":   ef_tco2_mwh,
                "carbon_uplift_cny_kwh": round(carbon_uplift, 4),
                "elec_price_base":  base_price,
                "elec_price_eff":   round(elec_eff, 4),
                # D1 (fixed)
                "d1_ghi_annual":    row["d1_1_ghi_annual_kwh"],
                "d1_ghi_cv":        row["d1_3_ghi_cv_seasonal"],
                # D2 (fixed)
                "d2_roof_area":     row["d2_3_roof_area_mean"],
                "d2_density":       row["d2_2_building_density"],
                # D3 (fixed)
                "d3_specific_yield": sy,
                "d3_shading":       row["d3_2_shading_factor_mean"],
                # D4 (scenario-dependent)
                "d4_lcoe":          d4["d4_lcoe"],
                "d4_pbt":           d4["d4_pbt"],
                "d4_npv":           d4["d4_npv"],
                "d4_npv_positive":  d4["npv_pos"],
                # D5 (scenario-dependent)
                "d5_pbt_ci_width":  d5["d5_pbt_ci_width"],
                "d5_lcoe_std":      d5["d5_lcoe_std"],
                "d5_prob_pbt_le15": d5["d5_prob_pbt_le15"],
                # Effective params
                "pv_cost_mode":     scen["pv_cost_mode"],
                "pv_cost_left":     scen["pv_cost_left"],
                "carbon_price":     scen["carbon_price"],
                "subsidy":          scen["subsidy"],
            })
            log.info(f"    {city:15s} elec={elec_eff:.3f} LCOE={d4['d4_lcoe']:.3f} PBT={d4['d4_pbt']:.1f}yr")

    return pd.DataFrame(records)


# ============================================================================
# 4. 跨情景归一化 + FDSI 重算
# ============================================================================

def normalize_col(series: pd.Series, is_benefit: bool,
                  vmin: float = None, vmax: float = None) -> pd.Series:
    """Min-max 归一化，vmin/vmax 可指定（跨情景固定范围）。"""
    if vmin is None: vmin = series.min()
    if vmax is None: vmax = series.max()
    if vmax == vmin:
        return pd.Series(0.5, index=series.index)
    if is_benefit:
        return (series - vmin) / (vmax - vmin)
    else:
        return (vmax - series) / (vmax - vmin)


def compute_fdsi_for_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    跨情景归一化 D4/D5，D1/D2/D3 使用 baseline 范围归一化，
    然后计算 FDSI。

    维度权重（AHP 主观权重，与 05_fdsi_scoring.py 一致）:
      D1: 0.2246, D2: 0.1124, D3: 0.2246, D4: 0.1124, D5: 0.3260
    """
    # AHP 权重（从 05 脚本同步）
    W = {"D1": 0.2246, "D2": 0.1124, "D3": 0.2246, "D4": 0.1124, "D5": 0.3260}

    # ── D1/D2/D3: 用 baseline 子集的范围归一化 ──
    base_mask = df["scenario"] == "baseline"

    def norm_baseline_range(col, is_benefit):
        vmin = df.loc[base_mask, col].min()
        vmax = df.loc[base_mask, col].max()
        return normalize_col(df[col], is_benefit, vmin, vmax)

    n_d1_ghi  = norm_baseline_range("d1_ghi_annual", True)
    n_d1_cv   = norm_baseline_range("d1_ghi_cv",     False)
    n_d2_roof = norm_baseline_range("d2_roof_area",  True)
    n_d2_den  = norm_baseline_range("d2_density",    False)
    n_d3_sy   = norm_baseline_range("d3_specific_yield", True)
    n_d3_sh   = norm_baseline_range("d3_shading",    True)

    # D1/D2/D3 维度得分
    D1 = 0.6 * n_d1_ghi + 0.4 * n_d1_cv
    D2 = 0.5 * n_d2_roof + 0.5 * n_d2_den
    D3 = 0.5 * n_d3_sy  + 0.5 * n_d3_sh

    # ── D4/D5: 跨所有 4×39=156 个值归一化 ──
    # 处理 NaN
    df["d5_pbt_ci_width_fill"] = df["d5_pbt_ci_width"].fillna(df["d5_pbt_ci_width"].median())

    n_d4_lcoe = normalize_col(df["d4_lcoe"],            False)
    n_d4_pbt  = normalize_col(df["d4_pbt"],             False)
    n_d5_ci   = normalize_col(df["d5_pbt_ci_width_fill"], False)
    n_d5_std  = normalize_col(df["d5_lcoe_std"],        False)

    # D4 没有 elec_price_sensitivity (Sobol) —— 此处用 prob_pbt_le15 替代作为额外指标
    # D5 weights: CI_width=0.50, lcoe_std=0.50（简化版）
    D4 = 0.50 * n_d4_lcoe + 0.50 * n_d4_pbt
    D5 = 0.50 * n_d5_ci   + 0.50 * n_d5_std

    # FDSI
    FDSI = W["D1"]*D1 + W["D2"]*D2 + W["D3"]*D3 + W["D4"]*D4 + W["D5"]*D5

    df = df.copy()
    df["D1_score"] = D1.values
    df["D2_score"] = D2.values
    df["D3_score"] = D3.values
    df["D4_score"] = D4.values
    df["D5_score"] = D5.values
    df["fdsi"]     = FDSI.values

    # 排名（within scenario）
    df["rank"] = df.groupby("scenario")["fdsi"].rank(ascending=False, method="min").astype(int)

    return df


def assign_suitability(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于 baseline 的 FDSI 分布固定绝对阈值，对所有情景赋等级。
    High:   FDSI ≥ q67 (baseline)
    Medium: q33 ≤ FDSI < q67
    Low:    FDSI < q33
    """
    base = df[df["scenario"] == "baseline"]["fdsi"]
    q33  = base.quantile(0.33)
    q67  = base.quantile(0.67)
    log.info(f"  适宜性阈值 (baseline): q33={q33:.4f}, q67={q67:.4f}")

    def grade(v):
        if v >= q67: return "High"
        if v >= q33: return "Medium"
        return "Low"

    df = df.copy()
    df["suitability"] = df["fdsi"].apply(grade)
    df["q33"]         = q33
    df["q67"]         = q67
    return df


# ============================================================================
# 5. 跳变矩阵
# ============================================================================

def build_transition_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """构建适宜性等级跳变矩阵（baseline → 其他情景）。"""
    base    = df[df["scenario"] == "baseline"][["city", "suitability", "fdsi", "rank"]]
    base    = base.rename(columns={"suitability": "suit_baseline",
                                   "fdsi": "fdsi_baseline", "rank": "rank_baseline"})

    scen_keys = [s for s in SCENARIOS if s != "baseline"]
    out = base.copy()

    for sk in scen_keys:
        scen_df = df[df["scenario"] == sk][["city", "suitability", "fdsi", "rank"]]
        scen_df = scen_df.rename(columns={
            "suitability": f"suit_{sk}",
            "fdsi":        f"fdsi_{sk}",
            "rank":        f"rank_{sk}",
        })
        out = out.merge(scen_df, on="city", how="left")

    # 判断跳变
    grade_map = {"Low": 0, "Medium": 1, "High": 2}
    for sk in scen_keys:
        out[f"delta_grade_{sk}"] = (
            out[f"suit_{sk}"].map(grade_map) - out["suit_baseline"].map(grade_map)
        )
        out[f"delta_fdsi_{sk}"] = out[f"fdsi_{sk}"] - out["fdsi_baseline"]
        out[f"delta_rank_{sk}"] = out["rank_baseline"] - out[f"rank_{sk}"]  # positive = improved

    # 添加城市元数据
    city_info = df[df["scenario"] == "baseline"][["city", "name_en", "climate_zone",
                                                   "province", "grid_region",
                                                   "ef_tco2_mwh"]].drop_duplicates()
    out = out.merge(city_info, on="city", how="left")
    out = out.sort_values("rank_baseline")

    return out


# ============================================================================
# 6. 汇总统计
# ============================================================================

def print_summary(df: pd.DataFrame, trans: pd.DataFrame):
    """打印关键分析结论。"""
    log.info("\n" + "="*70)
    log.info("  情景分析汇总")
    log.info("="*70)

    # 各情景 High/Medium/Low 城市数
    log.info("\n【各情景适宜性分布】")
    for sk, sv in SCENARIOS.items():
        sub  = df[df["scenario"] == sk]
        h    = (sub["suitability"] == "High").sum()
        m    = (sub["suitability"] == "Medium").sum()
        lo   = (sub["suitability"] == "Low").sum()
        log.info(f"  {sv['label']:40s}: High={h:2d}  Medium={m:2d}  Low={lo:2d}")

    # PV 成本降低 50% 的跳变
    log.info("\n【cost_reduction: 光伏成本下降 50% 的城市跳变】")
    upgrades = trans[trans["delta_grade_cost_reduction"] > 0][
        ["name_en", "suit_baseline", "suit_cost_reduction", "fdsi_baseline",
         "fdsi_cost_reduction", "climate_zone"]]
    if len(upgrades) > 0:
        log.info(upgrades.to_string(index=False))
        pct = len(upgrades) / len(trans) * 100
        log.info(f"  → 共 {len(upgrades)}/{len(trans)} 城市升级 ({pct:.0f}%)")
    else:
        log.info("  → 无城市升级（所有城市已在当前等级内改善 FDSI 但未跨阈值）")

    # 碳价情景受益最大（高碳强度省份）
    log.info("\n【carbon_pricing: 碳价受益最大的前 10 城市（按 ΔFDSI）】")
    top_carbon = trans.nlargest(10, "delta_fdsi_carbon_pricing")[
        ["name_en", "province", "ef_tco2_mwh",
         "fdsi_baseline", "fdsi_carbon_pricing", "delta_fdsi_carbon_pricing",
         "suit_baseline", "suit_carbon_pricing"]]
    log.info(top_carbon.to_string(index=False))

    # FDSI 变化最大的城市（Aggressive）
    log.info("\n【aggressive: FDSI 变化最大的城市（前 10，按绝对值）】")
    top_agg = trans.nlargest(10, "delta_fdsi_aggressive")[
        ["name_en", "climate_zone", "fdsi_baseline", "fdsi_aggressive",
         "delta_fdsi_aggressive", "suit_baseline", "suit_aggressive"]]
    log.info(top_agg.to_string(index=False))

    # Aggressive 情景是否所有城市都达到 Medium+
    log.info("\n【aggressive: 低适宜性城市（Low）情况】")
    still_low = trans[trans["suit_aggressive"] == "Low"][
        ["name_en", "climate_zone", "fdsi_aggressive", "suit_baseline", "suit_aggressive"]]
    if len(still_low) > 0:
        log.info(still_low.to_string(index=False))
        log.info(f"  → 仍有 {len(still_low)} 个城市在 Aggressive 情景下处于 Low")
    else:
        log.info("  → ✓ Aggressive 情景下所有城市均达到 Medium 以上！")

    # 排名变化最大的城市
    log.info("\n【跨情景排名变化最大的城市（Aggressive vs Baseline）】")
    top_rank = trans.reindex(trans["delta_rank_aggressive"].abs().sort_values(ascending=False).index)
    log.info(top_rank[["name_en", "rank_baseline", "rank_aggressive",
                        "delta_rank_aggressive",
                        "suit_baseline", "suit_aggressive"]].head(10).to_string(index=False))


# ============================================================================
# 7. 可视化
# ============================================================================

SCEN_COLORS = {
    "baseline":       "#607D8B",
    "cost_reduction": "#2196F3",
    "carbon_pricing": "#FF9800",
    "aggressive":     "#4CAF50",
}
SCEN_LABELS = {k: v["label"] for k, v in SCENARIOS.items()}

ZONE_PALETTE = {
    "Severe Cold": "#E53935",
    "Cold":        "#1E88E5",
    "HSCW":        "#43A047",
    "HSWW":        "#FB8C00",
    "Mild":        "#8E24AA",
}


def plot_fdsi_heatmap(df: pd.DataFrame, save_path: Path):
    """热力图：城市 × 情景 FDSI。"""
    if not HAS_MPL:
        return

    pivot = df.pivot(index="city", columns="scenario", values="fdsi")
    # 按 baseline 排序
    pivot = pivot.loc[[c for c in CITY_ORDER if c in pivot.index]]
    labels = [df[df["city"] == c]["name_en"].iloc[0] for c in pivot.index]

    # 情景列排序
    col_order = list(SCENARIOS.keys())
    pivot = pivot[[c for c in col_order if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=(9, 14))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([SCEN_LABELS[c] for c in pivot.columns],
                       rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)

    # 注释数值
    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5, color="black" if 0.3 < v < 0.8 else "white")

    plt.colorbar(im, ax=ax, label="FDSI Score", fraction=0.03, pad=0.04)
    ax.set_title("FDSI Score: 39 Cities × 4 Policy Scenarios",
                 fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  保存热力图: {save_path}")


def plot_transitions(trans: pd.DataFrame, save_path: Path):
    """Lollipop 图：baseline → aggressive FDSI 变化。"""
    if not HAS_MPL:
        return

    t = trans.copy()
    t = t.set_index("city")
    t = t.reindex([c for c in CITY_ORDER if c in t.index])

    fig, ax = plt.subplots(figsize=(8, 13))

    grade_colors = {"High": "#4CAF50", "Medium": "#FF9800", "Low": "#F44336"}

    for i, city in enumerate(t.index):
        row = t.loc[city]
        base_f = row["fdsi_baseline"]
        agg_f  = row.get("fdsi_aggressive", base_f)
        suit_b = row["suit_baseline"]
        suit_a = row.get("suit_aggressive", suit_b)

        ax.hlines(i, base_f, agg_f, colors="#BDBDBD", linewidth=1.5, zorder=1)
        ax.plot(base_f, i, "o", color=grade_colors.get(suit_b, "#9E9E9E"),
                ms=6, zorder=3, label="Baseline" if i == 0 else "")
        ax.plot(agg_f,  i, "D", color=grade_colors.get(suit_a, "#9E9E9E"),
                ms=6, zorder=3, label="Aggressive" if i == 0 else "")

    ax.set_yticks(range(len(t)))
    ax.set_yticklabels([trans.loc[trans["city"] == c, "name_en"].iloc[0]
                         for c in t.index], fontsize=8)
    ax.set_xlabel("FDSI Score", fontsize=11)
    ax.set_title("FDSI Shift: Baseline → Aggressive Policy\n(circle=baseline, diamond=aggressive)",
                 fontsize=11, fontweight="bold")

    # 阈值线 (derive from baseline FDSI distribution if not in trans columns)
    if "q33" in trans.columns:
        q33 = trans["q33"].iloc[0]
        q67 = trans["q67"].iloc[0]
    else:
        base_fdsi = trans["fdsi_baseline"]
        q33 = base_fdsi.quantile(0.33)
        q67 = base_fdsi.quantile(0.67)
    ax.axvline(q33, color="#F44336", ls="--", lw=0.8, alpha=0.7, label=f"Low/Medium threshold ({q33:.2f})")
    ax.axvline(q67, color="#4CAF50", ls="--", lw=0.8, alpha=0.7, label=f"Medium/High threshold ({q67:.2f})")

    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  保存跳变图: {save_path}")


def plot_rank_bump(df: pd.DataFrame, trans: pd.DataFrame, save_path: Path):
    """Bump chart：4 情景下的城市排名变化（Top 15 + Bottom 10）。"""
    if not HAS_MPL:
        return

    # 选择排名变化 >5 的城市
    big_movers = trans[trans["delta_rank_aggressive"].abs() >= 3]["city"].tolist()
    # 也加入 baseline Top-5 和 Bottom-5
    base_sorted = trans.sort_values("rank_baseline")
    showcase    = (list(base_sorted.head(8)["city"]) +
                   list(base_sorted.tail(5)["city"]) +
                   big_movers)
    showcase    = list(dict.fromkeys(showcase))[:20]  # 最多 20 个

    scen_keys = list(SCENARIOS.keys())
    pivot = {}
    for sk in scen_keys:
        sub = df[df["scenario"] == sk].set_index("city")
        pivot[sk] = sub["rank"]

    fig, ax = plt.subplots(figsize=(10, 8))
    x_pos = {sk: i for i, sk in enumerate(scen_keys)}

    for city in showcase:
        name = trans.loc[trans["city"] == city, "name_en"].iloc[0] if len(
            trans.loc[trans["city"] == city]) else city
        zone = df.loc[df["city"] == city, "climate_zone"].iloc[0] if len(
            df.loc[df["city"] == city]) else "Cold"
        color = ZONE_PALETTE.get(zone, "#607D8B")

        xs = [x_pos[sk] for sk in scen_keys]
        ys = [pivot[sk].get(city, np.nan) for sk in scen_keys]

        ax.plot(xs, ys, "o-", color=color, alpha=0.8, lw=1.5, ms=4)
        ax.text(xs[-1] + 0.05, ys[-1], name, fontsize=7, va="center",
                color=color)

    ax.set_xticks(list(x_pos.values()))
    ax.set_xticklabels([SCEN_LABELS[sk] for sk in scen_keys],
                       rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Rank (lower = more suitable)", fontsize=10)
    ax.invert_yaxis()
    ax.set_title("City Ranking Across 4 Policy Scenarios (selected cities)",
                 fontsize=11, fontweight="bold")

    # 图例（气候区）
    handles = [mpatches.Patch(color=c, label=z) for z, c in ZONE_PALETTE.items()]
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  保存排名 bump 图: {save_path}")


def plot_scenario_bar(df: pd.DataFrame, save_path: Path):
    """柱状图：各情景下 High/Medium/Low 城市数量。"""
    if not HAS_MPL:
        return

    counts = (df.groupby(["scenario", "suitability"])
                .size().unstack(fill_value=0)
                .reindex(list(SCENARIOS.keys())))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(counts))
    w = 0.25
    grade_colors = {"High": "#4CAF50", "Medium": "#FF9800", "Low": "#F44336"}

    for j, grade in enumerate(["High", "Medium", "Low"]):
        if grade in counts.columns:
            ax.bar(x + j*w, counts[grade], w, label=grade,
                   color=grade_colors[grade], alpha=0.85)
            for i, v in enumerate(counts[grade]):
                ax.text(i + j*w, v + 0.2, str(v), ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + w)
    ax.set_xticklabels([SCEN_LABELS[s] for s in counts.index],
                       rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("Number of Cities", fontsize=11)
    ax.set_title("Suitability Distribution Across Policy Scenarios (39 cities)",
                 fontsize=11, fontweight="bold")
    ax.legend(title="Suitability", fontsize=9)
    ax.set_ylim(0, 25)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  保存分布柱状图: {save_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    log.info("=" * 60)
    log.info("  NC Phase 3: 政策情景分析")
    log.info("=" * 60)

    log.info("\n[1] 加载基础数据...")
    base_df = load_baseline_data()

    log.info("\n[2] 计算 4 情景 × 39 城市 D4/D5...")
    raw_df = run_all_scenarios(base_df)
    raw_df.to_csv(RESULTS_DIR / "scenario_d4_detail.csv", index=False)
    log.info(f"  D4/D5 明细: {RESULTS_DIR / 'scenario_d4_detail.csv'}")

    log.info("\n[3] 跨情景归一化 + FDSI 计算...")
    scored_df = compute_fdsi_for_all(raw_df)

    log.info("\n[4] 适宜性等级分配...")
    scored_df = assign_suitability(scored_df)

    # 保存主矩阵
    out_cols = ["scenario", "scenario_label", "city", "name_en",
                "climate_zone", "province", "grid_region",
                "elec_price_eff", "carbon_uplift_cny_kwh",
                "d4_lcoe", "d4_pbt", "d4_npv_positive",
                "D1_score", "D2_score", "D3_score", "D4_score", "D5_score",
                "fdsi", "rank", "suitability"]
    out_df = scored_df[[c for c in out_cols if c in scored_df.columns]]
    out_df.to_csv(RESULTS_DIR / "scenario_fdsi_matrix.csv", index=False)
    log.info(f"  FDSI 矩阵: {RESULTS_DIR / 'scenario_fdsi_matrix.csv'}")

    log.info("\n[5] 构建跳变矩阵...")
    trans = build_transition_matrix(scored_df)
    trans.to_csv(RESULTS_DIR / "suitability_transitions.csv", index=False)
    log.info(f"  跳变矩阵: {RESULTS_DIR / 'suitability_transitions.csv'}")

    log.info("\n[6] 汇总统计与关键结论...")
    print_summary(scored_df, trans)

    log.info("\n[7] 生成图表...")
    plot_fdsi_heatmap(scored_df, FIGURES_DIR / "fig_scenario_fdsi_heatmap.png")
    plot_transitions(trans,      FIGURES_DIR / "fig_scenario_transitions.png")
    plot_rank_bump(scored_df, trans, FIGURES_DIR / "fig_scenario_rank_bump.png")
    plot_scenario_bar(scored_df, FIGURES_DIR / "fig_scenario_distribution.png")

    log.info("\n" + "="*60)
    log.info("  情景分析完成")
    log.info(f"  输出目录: {RESULTS_DIR}")
    log.info("="*60)


if __name__ == "__main__":
    main()
