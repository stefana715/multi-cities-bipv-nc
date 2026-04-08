#!/usr/bin/env python3
"""
============================================================================
Paper 4 - Script 04: Energy Simulation & Uncertainty Analysis
============================================================================
Step 4: 对五个城市运行 pvlib 确定性模拟 + Monte Carlo 不确定性 + Sobol 敏感性分析。

计算维度：
  D1 气候资源维度 — 从 TMY 数据提取（GHI、气温、日照时数、季节变异）
  D3 技术部署维度（补充）— pvlib 发电量模拟 → 年均单位面积发电量、自给率
  D4 经济可行性维度 — LCOE、简单回收期 PBT
  D5 确定性/稳健性维度 — MC模拟CV、PBT 95%CI、P(PBT≤15yr)、Sobol指数

工作流程：
  1. 解析各城市 PVGIS TMY 数据 → D1 子指标
  2. pvlib ModelChain 确定性模拟 → 年发电量基准
  3. 结合 Step 3 的 D3 数据 → 总发电量、自给率
  4. 经济性计算 → LCOE、PBT
  5. Monte Carlo 模拟 (LHS, N=10000) → 发电量和PBT的分布
  6. Sobol 全局敏感性分析 → 主效应 S1 和总效应 ST
  7. 汇总输出

复用 Paper 3 模块：
  comparison/src/deterministic/pvlib_model.py   → PV系统建模
  comparison/src/deterministic/economics.py     → 经济性计算
  comparison/src/stochastic/lhs_sampler.py      → LHS抽样
  comparison/src/stochastic/monte_carlo.py      → MC引擎
  comparison/src/sensitivity/sobol_analysis.py  → Sobol分析

  如果 Paper 3 模块不可用，本脚本内置了轻量化实现。

输出：
  results/energy/{city}_d1_climate.csv       — D1 气候指标
  results/energy/{city}_deterministic.csv    — 确定性模拟结果
  results/energy/{city}_d4_economics.csv     — D4 经济指标
  results/energy/{city}_mc_summary.csv       — MC统计摘要
  results/energy/{city}_sobol_indices.csv    — Sobol指数
  results/energy/cross_city_d1d4d5.csv       — 五城市汇总

用法：
  python scripts/04_energy_simulation.py                # 全部五城市
  python scripts/04_energy_simulation.py --city beijing  # 单个城市
  python scripts/04_energy_simulation.py --mc-samples 1000  # 减少MC样本（测试用）
  python scripts/04_energy_simulation.py --skip-sobol    # 跳过Sobol（省时间）
============================================================================
"""

import argparse
import logging
import sys
import time
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pvlib
    from pvlib.location import Location
    from pvlib.pvsystem import PVSystem
    from pvlib.modelchain import ModelChain
    from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
    HAS_PVLIB = True
except ImportError:
    HAS_PVLIB = False
    print("WARNING: pvlib not installed. Energy simulation will use simplified model.")
    print("  Install with: pip install pvlib")

try:
    from scipy.stats import qmc, norm, triang, uniform
    from scipy.stats.qmc import LatinHypercube
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from SALib.sample import saltelli as salib_saltelli
    from SALib.analyze import sobol as salib_sobol
    HAS_SALIB = True
except ImportError:
    HAS_SALIB = False
    print("WARNING: SALib not installed. Sobol analysis will be skipped.")
    print("  Install with: pip install SALib")

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from tabulate import tabulate as _tabulate
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
PVGIS_DIR = PROJECT_DIR / "results" / "pvgis"
MORPHOLOGY_DIR = PROJECT_DIR / "results" / "morphology"
RESULTS_DIR = PROJECT_DIR / "results" / "energy"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 城市配置
# ============================================================================

CITIES = {
    "harbin": {
        "name_en": "Harbin", "name_cn": "哈尔滨",
        "climate_zone": "severe_cold",
        "lat": 45.75, "lon": 126.65, "alt": 150,
        "tz": "Asia/Shanghai",
        "electricity_price": 0.51,     # CNY/kWh
        "residential_demand_kwh_m2": 35,  # 年均住宅用电 kWh/m² (建筑面积)
    },
    "beijing": {
        "name_en": "Beijing", "name_cn": "北京",
        "climate_zone": "cold",
        "lat": 39.90, "lon": 116.40, "alt": 44,
        "tz": "Asia/Shanghai",
        "electricity_price": 0.4883,
        "residential_demand_kwh_m2": 30,
    },
    "changsha": {
        "name_en": "Changsha", "name_cn": "长沙",
        "climate_zone": "hscw",
        "lat": 28.23, "lon": 112.94, "alt": 45,
        "tz": "Asia/Shanghai",
        "electricity_price": 0.588,
        "residential_demand_kwh_m2": 28,
    },
    "shenzhen": {
        "name_en": "Shenzhen", "name_cn": "深圳",
        "climate_zone": "hsww",
        "lat": 22.54, "lon": 114.06, "alt": 30,
        "tz": "Asia/Shanghai",
        "electricity_price": 0.68,
        "residential_demand_kwh_m2": 40,  # 空调负荷高
    },
    "kunming": {
        "name_en": "Kunming", "name_cn": "昆明",
        "climate_zone": "mild",
        "lat": 25.04, "lon": 102.68, "alt": 1892,
        "tz": "Asia/Shanghai",
        "electricity_price": 0.45,
        "residential_demand_kwh_m2": 22,
    },
    # ── 10 new cities (Phase 2 expansion) ──
    "changchun": {
        "name_en": "Changchun", "name_cn": "长春",
        "climate_zone": "severe_cold",
        "lat": 43.88, "lon": 125.32, "alt": 237,
        "tz": "Asia/Shanghai",
        "electricity_price": 0.52,
        "residential_demand_kwh_m2": 33,
    },
    "shenyang": {
        "name_en": "Shenyang", "name_cn": "沈阳",
        "climate_zone": "severe_cold",
        "lat": 41.80, "lon": 123.43, "alt": 42,
        "tz": "Asia/Shanghai",
        "electricity_price": 0.53,
        "residential_demand_kwh_m2": 32,
    },
    "jinan": {
        "name_en": "Jinan", "name_cn": "济南",
        "climate_zone": "cold",
        "lat": 36.65, "lon": 116.99, "alt": 50,
        "tz": "Asia/Shanghai",
        "electricity_price": 0.547,
        "residential_demand_kwh_m2": 30,
    },
    "xian": {
        "name_en": "Xian", "name_cn": "西安",
        "climate_zone": "cold",
        "lat": 34.26, "lon": 108.94, "alt": 397,
        "tz": "Asia/Shanghai",
        "electricity_price": 0.498,
        "residential_demand_kwh_m2": 28,
    },
    "wuhan": {
        "name_en": "Wuhan", "name_cn": "武汉",
        "climate_zone": "hscw",
        "lat": 30.58, "lon": 114.30, "alt": 23,
        "tz": "Asia/Shanghai",
        "electricity_price": 0.558,
        "residential_demand_kwh_m2": 35,
    },
    "nanjing": {
        "name_en": "Nanjing", "name_cn": "南京",
        "climate_zone": "hscw",
        "lat": 32.06, "lon": 118.77, "alt": 9,
        "tz": "Asia/Shanghai",
        "electricity_price": 0.558,
        "residential_demand_kwh_m2": 33,
    },
    "guangzhou": {
        "name_en": "Guangzhou", "name_cn": "广州",
        "climate_zone": "hsww",
        "lat": 23.13, "lon": 113.27, "alt": 21,
        "tz": "Asia/Shanghai",
        "electricity_price": 0.68,
        "residential_demand_kwh_m2": 42,
    },
    "xiamen": {
        "name_en": "Xiamen", "name_cn": "厦门",
        "climate_zone": "hsww",
        "lat": 24.48, "lon": 118.09, "alt": 63,
        "tz": "Asia/Shanghai",
        "electricity_price": 0.618,
        "residential_demand_kwh_m2": 38,
    },
    "guiyang": {
        "name_en": "Guiyang", "name_cn": "贵阳",
        "climate_zone": "mild",
        "lat": 26.65, "lon": 106.63, "alt": 1074,
        "tz": "Asia/Shanghai",
        "electricity_price": 0.468,
        "residential_demand_kwh_m2": 20,
    },
    "chengdu": {
        "name_en": "Chengdu", "name_cn": "成都",
        "climate_zone": "mild",
        "lat": 30.67, "lon": 104.07, "alt": 506,
        "tz": "Asia/Shanghai",
        "electricity_price": 0.502,
        "residential_demand_kwh_m2": 25,
    },
}

# PV 系统统一参数（全国一致，方便对比）
PV_SYSTEM_PARAMS = {
    "module_efficiency": 0.20,
    "module_wp_per_m2": 200,        # W/m²
    "temp_coeff": -0.0035,          # %/°C
    "inverter_efficiency": 0.96,
    "system_losses": 0.14,
    "pv_cost_cny_wp": 3.0,
    "annual_degradation": 0.005,
    "system_lifetime": 25,
    "discount_rate": 0.06,
    "om_ratio": 0.01,
}

# Monte Carlo 不确定参数定义
MC_PARAMS = {
    "ghi_factor": {"dist": "normal", "loc": 1.0, "scale": 0.05},
    "module_efficiency": {"dist": "triangular", "left": 0.18, "mode": 0.20, "right": 0.22},
    "system_losses": {"dist": "uniform", "low": 0.10, "high": 0.18},
    "pv_cost": {"dist": "triangular", "left": 2.5, "mode": 3.0, "right": 4.0},
    "elec_price_factor": {"dist": "uniform", "low": 0.9, "high": 1.1},
}


# ============================================================================
# 1. D1 气候资源维度：TMY 数据解析
# ============================================================================

def parse_pvgis_tmy(city_key: str) -> Optional[pd.DataFrame]:
    """
    解析 PVGIS TMY CSV 文件。

    PVGIS TMY CSV 格式：前几行是元数据，数据从某行开始，
    包含 time, G(h), Gb(n), Gd(h), T2m, RH, WS10m, WD10m 等列。
    """
    tmy_path = PVGIS_DIR / f"{city_key}_tmy.csv"
    if not tmy_path.exists():
        log.error(f"  TMY 文件不存在: {tmy_path}")
        return None

    # PVGIS TMY CSV 的前几行是元数据，需要跳过
    lines = tmy_path.read_text(encoding="utf-8").strip().split("\n")

    # 找到数据开始行（包含 "time" 的行）
    header_idx = None
    for i, line in enumerate(lines):
        if "time" in line.lower() and ("g(h)" in line.lower() or "ghi" in line.lower()):
            header_idx = i
            break

    if header_idx is None:
        # 尝试其他格式
        for i, line in enumerate(lines):
            if line.strip().startswith("time") or line.strip().startswith("20"):
                header_idx = max(0, i - (0 if line.strip().startswith("time") else 1))
                break

    if header_idx is None:
        log.warning(f"  无法识别 TMY 文件格式，尝试 skiprows=16")
        header_idx = 16

    try:
        df = pd.read_csv(tmy_path, skiprows=header_idx, skipinitialspace=True)
    except Exception as e:
        log.error(f"  解析 TMY 失败: {e}")
        # 尝试更灵活的解析
        try:
            df = pd.read_csv(tmy_path, skiprows=header_idx,
                             skipinitialspace=True, on_bad_lines="skip")
        except Exception as e2:
            log.error(f"  二次解析也失败: {e2}")
            return None

    # 标准化列名
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        if cl in ("g(h)", "ghi", "global_horizontal"):
            col_map[col] = "ghi"
        elif cl in ("gb(n)", "dni", "beam_normal"):
            col_map[col] = "dni"
        elif cl in ("gd(h)", "dhi", "diffuse_horizontal"):
            col_map[col] = "dhi"
        elif cl in ("t2m", "temp", "temperature", "t_air"):
            col_map[col] = "temp"
        elif cl in ("ws10m", "wind_speed", "ws"):
            col_map[col] = "wind_speed"
        elif cl in ("time", "date", "datetime"):
            col_map[col] = "time"

    df = df.rename(columns=col_map)

    # 解析时间
    if "time" in df.columns:
        try:
            df["time"] = pd.to_datetime(df["time"], format="%Y%m%d:%H%M", errors="coerce")
        except Exception:
            try:
                df["time"] = pd.to_datetime(df["time"], errors="coerce")
            except Exception:
                pass

    # 转为数值
    for col in ["ghi", "dni", "dhi", "temp", "wind_speed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 去掉空行和尾部元数据
    if "ghi" in df.columns:
        df = df.dropna(subset=["ghi"])

    log.info(f"  TMY 解析: {len(df)} 行, 列: {list(df.columns)}")
    return df


def compute_d1_indicators(tmy_df: pd.DataFrame, city_info: dict) -> Dict:
    """
    计算 D1 气候资源维度子指标。

    子指标：
      d1_1: 年均 GHI (kWh/m²/yr)
      d1_2: 年均气温 (°C)
      d1_3: GHI 季节变异系数
      d1_4: 年日照时数 (h) — GHI > 120 W/m² 的小时数
    """
    if tmy_df is None or "ghi" not in tmy_df.columns:
        return {
            "d1_1_ghi_annual": None,
            "d1_2_temp_annual": None,
            "d1_3_ghi_cv_seasonal": None,
            "d1_4_sunshine_hours": None,
        }

    # d1_1: 年均 GHI
    # TMY 数据是逐时 W/m²，求和后 / 1000 转 kWh
    ghi_annual_kwh = tmy_df["ghi"].sum() / 1000.0

    # d1_2: 年均气温
    temp_annual = tmy_df["temp"].mean() if "temp" in tmy_df.columns else None

    # d1_3: GHI 季节变异系数
    # 按月汇总，计算月均 GHI 的 CV
    if "time" in tmy_df.columns and tmy_df["time"].notna().any():
        tmy_df = tmy_df.copy()
        tmy_df["month"] = tmy_df["time"].dt.month
        monthly_ghi = tmy_df.groupby("month")["ghi"].sum() / 1000  # kWh/m²/month
        ghi_cv = monthly_ghi.std() / monthly_ghi.mean() if monthly_ghi.mean() > 0 else None
    else:
        # 无时间戳，假设 8760 行按小时排列
        n_hours = len(tmy_df)
        hours_per_month = n_hours / 12
        monthly_ghi = []
        for m in range(12):
            start = int(m * hours_per_month)
            end = int((m + 1) * hours_per_month)
            monthly_ghi.append(tmy_df["ghi"].iloc[start:end].sum() / 1000)
        monthly_ghi = pd.Series(monthly_ghi)
        ghi_cv = monthly_ghi.std() / monthly_ghi.mean() if monthly_ghi.mean() > 0 else None

    # d1_4: 日照时数（GHI > 120 W/m² 的小时数）
    sunshine_hours = (tmy_df["ghi"] > 120).sum()

    return {
        "d1_1_ghi_annual_kwh": round(ghi_annual_kwh, 1),
        "d1_2_temp_annual_c": round(temp_annual, 1) if temp_annual is not None else None,
        "d1_3_ghi_cv_seasonal": round(ghi_cv, 4) if ghi_cv is not None else None,
        "d1_4_sunshine_hours": int(sunshine_hours),
    }


# ============================================================================
# 2. pvlib 确定性模拟
# ============================================================================

def run_pvlib_simulation(tmy_df: pd.DataFrame, city_info: dict) -> Dict:
    """
    用 pvlib ModelChain 模拟一个参考 PV 系统的年发电量。

    参考系统：1 kWp 标准系统，用于计算单位容量产出。
    """
    if not HAS_PVLIB or tmy_df is None:
        return run_simplified_simulation(tmy_df, city_info)

    lat = city_info["lat"]
    lon = city_info["lon"]
    alt = city_info["alt"]
    tz = city_info["tz"]

    # 创建 Location
    location = Location(lat, lon, tz=tz, altitude=alt)

    # 准备天气数据
    weather = tmy_df.copy()
    if "time" in weather.columns and weather["time"].notna().any():
        weather = weather.set_index("time")
    else:
        # 构造一个标准年的 DatetimeIndex
        weather.index = pd.date_range(
            start="2020-01-01", periods=len(weather), freq="h", tz=tz
        )

    # 确保必要列存在
    required = ["ghi", "dni", "dhi", "temp", "wind_speed"]
    for col in required:
        if col not in weather.columns:
            if col == "wind_speed":
                weather[col] = 1.5  # 默认风速
            elif col == "temp":
                weather[col] = 20.0
            elif col == "dni":
                # 从 GHI 和 DHI 估算
                weather[col] = (weather["ghi"] - weather.get("dhi", weather["ghi"] * 0.3)).clip(lower=0)
            elif col == "dhi":
                weather[col] = weather["ghi"] * 0.3

    # 重命名列以匹配 pvlib 期望
    weather_pvlib = weather.rename(columns={
        "ghi": "ghi", "dni": "dni", "dhi": "dhi",
        "temp": "temp_air", "wind_speed": "wind_speed",
    })

    try:
        # 定义 PV 系统
        # 使用 CEC 数据库中的通用组件
        module_params = {
            "pdc0": 1000,  # 1 kWp 标准系统
            "gamma_pdc": PV_SYSTEM_PARAMS["temp_coeff"],
        }
        inverter_params = {
            "pdc0": 1100,
            "eta_inv_nom": PV_SYSTEM_PARAMS["inverter_efficiency"],
        }

        # 使用 PVWatts 模型（更简洁可靠）
        system = PVSystem(
            surface_tilt=lat,  # 最优倾角近似等于纬度
            surface_azimuth=180,  # 朝南
            module_parameters=module_params,
            inverter_parameters=inverter_params,
        )

        mc = ModelChain.with_pvwatts(
            system, location,
            dc_model="pvwatts",
            ac_model="pvwatts",
            aoi_model="physical",
            spectral_model="no_loss",
            losses_model="pvwatts",
        )

        mc.run_model(weather_pvlib[["ghi", "dni", "dhi", "temp_air", "wind_speed"]])

        # 年发电量（kWh/kWp）
        ac_annual = mc.results.ac.sum() / 1000  # Wh → kWh
        specific_yield = ac_annual  # kWh/kWp/yr

        log.info(f"  pvlib 模拟: {specific_yield:.0f} kWh/kWp/yr")

        return {
            "method": "pvlib_pvwatts",
            "specific_yield_kwh_kwp": round(specific_yield, 1),
            "ac_annual_kwh_kwp": round(ac_annual, 1),
        }

    except Exception as e:
        log.warning(f"  pvlib ModelChain 失败: {e}, 改用简化模型")
        return run_simplified_simulation(tmy_df, city_info)


def run_simplified_simulation(tmy_df: pd.DataFrame, city_info: dict) -> Dict:
    """
    简化的发电量模拟（不依赖 pvlib 完整 ModelChain）。

    公式: E = GHI × η_module × η_inverter × (1 - losses) × temp_correction
    """
    if tmy_df is None or "ghi" not in tmy_df.columns:
        return {"method": "failed", "specific_yield_kwh_kwp": None}

    ghi_annual = tmy_df["ghi"].sum() / 1000  # kWh/m²/yr
    eta = PV_SYSTEM_PARAMS["module_efficiency"]
    inv_eff = PV_SYSTEM_PARAMS["inverter_efficiency"]
    losses = PV_SYSTEM_PARAMS["system_losses"]

    # 温度修正
    if "temp" in tmy_df.columns:
        temp_mean = tmy_df["temp"].mean()
        temp_correction = 1 + PV_SYSTEM_PARAMS["temp_coeff"] * (temp_mean - 25)
    else:
        temp_correction = 1.0

    # 单位面积发电量
    energy_per_m2 = ghi_annual * eta * inv_eff * (1 - losses) * temp_correction

    # 转换为 kWh/kWp (1 kWp ≈ 5 m² @ 200 W/m²)
    specific_yield = energy_per_m2 / (PV_SYSTEM_PARAMS["module_wp_per_m2"] / 1000)

    log.info(f"  简化模拟: {specific_yield:.0f} kWh/kWp/yr")

    return {
        "method": "simplified",
        "specific_yield_kwh_kwp": round(specific_yield, 1),
        "energy_per_m2_kwh": round(energy_per_m2, 1),
    }


# ============================================================================
# 3. D4 经济性计算
# ============================================================================

def compute_d4_economics(
    specific_yield: float,
    city_info: dict,
    morphology_data: Optional[Dict] = None,
) -> Dict:
    """
    计算 D4 经济可行性维度子指标。

    子指标：
      d4_1: 度电成本 LCOE (CNY/kWh)
      d4_2: 简单回收期 PBT (years)
      d4_3: 净现值系数 (NPV > 0 的条件)
    """
    if specific_yield is None or specific_yield <= 0:
        return {"d4_1_lcoe": None, "d4_2_pbt": None, "d4_3_npv_positive": None}

    p = PV_SYSTEM_PARAMS
    elec_price = city_info["electricity_price"]

    # ── LCOE ──
    # LCOE = (CapEx + PV(O&M)) / PV(Energy)
    capex = p["pv_cost_cny_wp"] * 1000  # CNY per kWp
    r = p["discount_rate"]
    n = p["system_lifetime"]
    deg = p["annual_degradation"]
    om_annual = capex * p["om_ratio"]

    # 折现发电量
    pv_energy = sum(
        specific_yield * (1 - deg) ** t / (1 + r) ** t
        for t in range(1, n + 1)
    )
    # 折现运维成本
    pv_om = sum(om_annual / (1 + r) ** t for t in range(1, n + 1))

    lcoe = (capex + pv_om) / pv_energy if pv_energy > 0 else float("inf")

    # ── 简单回收期 PBT ──
    # PBT = CapEx / (年发电收益 - 年运维成本)
    annual_revenue = specific_yield * elec_price
    annual_net = annual_revenue - om_annual
    pbt = capex / annual_net if annual_net > 0 else float("inf")

    # ── NPV > 0? ──
    npv = -capex + sum(
        (specific_yield * (1 - deg) ** t * elec_price - om_annual) / (1 + r) ** t
        for t in range(1, n + 1)
    )
    npv_positive = npv > 0

    return {
        "d4_1_lcoe_cny_kwh": round(lcoe, 4),
        "d4_2_pbt_years": round(pbt, 2),
        "d4_3_npv_positive": npv_positive,
        "d4_npv_cny_kwp": round(npv, 0),
        "d4_annual_revenue_cny_kwp": round(annual_revenue, 1),
    }


# ============================================================================
# 4. D5 Monte Carlo 不确定性分析
# ============================================================================

def lhs_sample(n_samples: int, param_defs: dict, seed: int = 42) -> pd.DataFrame:
    """
    Latin Hypercube Sampling 生成参数样本。
    """
    n_params = len(param_defs)
    param_names = list(param_defs.keys())

    if HAS_SCIPY:
        sampler = LatinHypercube(d=n_params, seed=seed)
        unit_samples = sampler.random(n=n_samples)
    else:
        # 简单的 LHS 替代
        rng = np.random.RandomState(seed)
        unit_samples = np.zeros((n_samples, n_params))
        for j in range(n_params):
            perm = rng.permutation(n_samples)
            unit_samples[:, j] = (perm + rng.random(n_samples)) / n_samples

    # 转换到各参数的实际分布
    samples = {}
    for j, (name, pdef) in enumerate(param_defs.items()):
        u = unit_samples[:, j]
        if pdef["dist"] == "normal":
            samples[name] = norm.ppf(u, loc=pdef["loc"], scale=pdef["scale"])
        elif pdef["dist"] == "uniform":
            samples[name] = uniform.ppf(u, loc=pdef["low"],
                                         scale=pdef["high"] - pdef["low"])
        elif pdef["dist"] == "triangular":
            c = (pdef["mode"] - pdef["left"]) / (pdef["right"] - pdef["left"])
            samples[name] = triang.ppf(u, c=c, loc=pdef["left"],
                                        scale=pdef["right"] - pdef["left"])

    return pd.DataFrame(samples)


def mc_energy_model(params_row: dict, ghi_annual: float, elec_price_base: float) -> Dict:
    """
    单次 MC 模拟：给定参数，计算发电量和 PBT。
    """
    # 调整后的年发电量密度 (kWh/m²)
    ghi_adj = ghi_annual * params_row["ghi_factor"]
    eta = params_row["module_efficiency"]
    losses = params_row["system_losses"]
    inv_eff = PV_SYSTEM_PARAMS["inverter_efficiency"]

    energy_per_m2 = ghi_adj * eta * inv_eff * (1 - losses)
    specific_yield = energy_per_m2 / (PV_SYSTEM_PARAMS["module_wp_per_m2"] / 1000)

    # 经济性
    pv_cost = params_row["pv_cost"]
    elec_price = elec_price_base * params_row["elec_price_factor"]
    capex = pv_cost * 1000  # CNY/kWp
    om = capex * PV_SYSTEM_PARAMS["om_ratio"]
    revenue = specific_yield * elec_price
    net = revenue - om
    pbt = capex / net if net > 0 else 99

    # LCOE
    r = PV_SYSTEM_PARAMS["discount_rate"]
    n = PV_SYSTEM_PARAMS["system_lifetime"]
    deg = PV_SYSTEM_PARAMS["annual_degradation"]
    pv_energy = sum(specific_yield * (1 - deg) ** t / (1 + r) ** t for t in range(1, n + 1))
    pv_om = sum(om / (1 + r) ** t for t in range(1, n + 1))
    lcoe = (capex + pv_om) / pv_energy if pv_energy > 0 else 99

    return {
        "specific_yield": specific_yield,
        "energy_per_m2": energy_per_m2,
        "pbt": pbt,
        "lcoe": lcoe,
    }


def run_monte_carlo(
    ghi_annual: float,
    elec_price: float,
    n_samples: int = 10000,
    seed: int = 42,
) -> Dict:
    """
    Monte Carlo 模拟，输出 D5 子指标。
    """
    log.info(f"  MC 模拟: N={n_samples}, LHS 抽样...")

    t0 = time.time()

    # LHS 抽样
    samples_df = lhs_sample(n_samples, MC_PARAMS, seed=seed)

    # 批量模拟
    results = []
    for i in range(n_samples):
        row = samples_df.iloc[i].to_dict()
        r = mc_energy_model(row, ghi_annual, elec_price)
        results.append(r)

    mc_df = pd.DataFrame(results)
    elapsed = time.time() - t0
    log.info(f"  MC 完成: {elapsed:.1f}s")

    # ── D5 子指标 ──
    sy = mc_df["specific_yield"]
    pbt = mc_df["pbt"]
    lcoe = mc_df["lcoe"]

    # d5_1: 发电量 CV
    d5_1_cv = sy.std() / sy.mean() if sy.mean() > 0 else None

    # d5_2: PBT 95% CI 宽度
    pbt_clean = pbt[pbt < 99]  # 排除极端值
    if len(pbt_clean) > 10:
        pbt_p025 = pbt_clean.quantile(0.025)
        pbt_p975 = pbt_clean.quantile(0.975)
        d5_2_pbt_ci_width = pbt_p975 - pbt_p025
    else:
        pbt_p025 = pbt_p975 = d5_2_pbt_ci_width = None

    # d5_3: P(PBT ≤ 15yr)
    d5_3_prob_pbt15 = (pbt <= 15).mean()

    return {
        "mc_n_samples": n_samples,
        "mc_elapsed_s": round(elapsed, 1),
        # 发电量统计
        "mc_yield_mean": round(sy.mean(), 1),
        "mc_yield_std": round(sy.std(), 1),
        "mc_yield_p05": round(sy.quantile(0.05), 1),
        "mc_yield_p50": round(sy.quantile(0.50), 1),
        "mc_yield_p95": round(sy.quantile(0.95), 1),
        # D5 子指标
        "d5_1_yield_cv": round(d5_1_cv, 4) if d5_1_cv else None,
        "d5_2_pbt_ci95_width": round(d5_2_pbt_ci_width, 2) if d5_2_pbt_ci_width else None,
        "d5_2_pbt_p025": round(pbt_p025, 2) if pbt_p025 is not None else None,
        "d5_2_pbt_p975": round(pbt_p975, 2) if pbt_p975 is not None else None,
        "d5_3_prob_pbt_le_15yr": round(d5_3_prob_pbt15, 4),
        # LCOE 统计
        "mc_lcoe_mean": round(lcoe.mean(), 4),
        "mc_lcoe_std": round(lcoe.std(), 4),
        "mc_lcoe_p05": round(lcoe.quantile(0.05), 4),
        "mc_lcoe_p95": round(lcoe.quantile(0.95), 4),
    }


# ============================================================================
# 5. D5 Sobol 全局敏感性分析
# ============================================================================

def run_sobol_analysis(
    ghi_annual: float,
    elec_price: float,
    n_samples: int = 4096,
) -> Dict:
    """
    Sobol 全局敏感性分析。

    输出每个不确定参数的 S1（一阶主效应）和 ST（总效应指数）。
    """
    if not HAS_SALIB:
        log.warning("  SALib 未安装，跳过 Sobol 分析")
        return {"sobol_status": "skipped_no_salib"}

    log.info(f"  Sobol 分析: N={n_samples} (总模型调用 ≈ {n_samples * (2 * len(MC_PARAMS) + 2):,})")
    t0 = time.time()

    # 定义问题
    param_names = list(MC_PARAMS.keys())
    problem = {
        "num_vars": len(param_names),
        "names": param_names,
        "bounds": [],
    }

    for name, pdef in MC_PARAMS.items():
        if pdef["dist"] == "normal":
            # ±3σ 范围
            lo = pdef["loc"] - 3 * pdef["scale"]
            hi = pdef["loc"] + 3 * pdef["scale"]
        elif pdef["dist"] == "uniform":
            lo, hi = pdef["low"], pdef["high"]
        elif pdef["dist"] == "triangular":
            lo, hi = pdef["left"], pdef["right"]
        else:
            lo, hi = 0, 1
        problem["bounds"].append([lo, hi])

    # Saltelli 抽样
    param_values = salib_saltelli.sample(problem, n_samples, calc_second_order=False)
    log.info(f"  Saltelli 样本: {param_values.shape[0]:,} 组")

    # 模型评估
    Y_yield = np.zeros(param_values.shape[0])
    Y_pbt = np.zeros(param_values.shape[0])

    for i in range(param_values.shape[0]):
        row = dict(zip(param_names, param_values[i]))
        r = mc_energy_model(row, ghi_annual, elec_price)
        Y_yield[i] = r["specific_yield"]
        Y_pbt[i] = min(r["pbt"], 99)  # 截断极端值

    # Sobol 分析
    si_yield = salib_sobol.analyze(problem, Y_yield, calc_second_order=False,
                                    print_to_console=False)
    si_pbt = salib_sobol.analyze(problem, Y_pbt, calc_second_order=False,
                                  print_to_console=False)

    elapsed = time.time() - t0
    log.info(f"  Sobol 完成: {elapsed:.1f}s")

    # 整理结果
    sobol_result = {
        "sobol_n_samples": n_samples,
        "sobol_elapsed_s": round(elapsed, 1),
    }

    # 发电量的 Sobol 指数
    for j, name in enumerate(param_names):
        sobol_result[f"sobol_yield_S1_{name}"] = round(si_yield["S1"][j], 4)
        sobol_result[f"sobol_yield_ST_{name}"] = round(si_yield["ST"][j], 4)

    # PBT 的 Sobol 指数
    for j, name in enumerate(param_names):
        sobol_result[f"sobol_pbt_S1_{name}"] = round(si_pbt["S1"][j], 4)
        sobol_result[f"sobol_pbt_ST_{name}"] = round(si_pbt["ST"][j], 4)

    # 交互效应占比
    s1_sum_yield = sum(si_yield["S1"])
    st_sum_yield = sum(si_yield["ST"])
    sobol_result["d5_4_interaction_ratio_yield"] = round(
        1 - s1_sum_yield / st_sum_yield if st_sum_yield > 0 else 0, 4
    )

    s1_sum_pbt = sum(si_pbt["S1"])
    st_sum_pbt = sum(si_pbt["ST"])
    sobol_result["d5_4_interaction_ratio_pbt"] = round(
        1 - s1_sum_pbt / st_sum_pbt if st_sum_pbt > 0 else 0, 4
    )

    # 主导敏感因子
    s1_yield = dict(zip(param_names, si_yield["S1"]))
    dominant_factor = max(s1_yield, key=s1_yield.get)
    sobol_result["d5_5_dominant_factor_yield"] = dominant_factor
    sobol_result["d5_5_dominant_S1_yield"] = round(s1_yield[dominant_factor], 4)

    s1_pbt = dict(zip(param_names, si_pbt["S1"]))
    dominant_pbt = max(s1_pbt, key=s1_pbt.get)
    sobol_result["d5_5_dominant_factor_pbt"] = dominant_pbt
    sobol_result["d5_5_dominant_S1_pbt"] = round(s1_pbt[dominant_pbt], 4)

    return sobol_result


# ============================================================================
# 6. 读取 Step 3 形态数据
# ============================================================================

def load_morphology_data(city_key: str) -> Optional[Dict]:
    """读取 Step 3 输出的 D2/D3 汇总数据。"""
    summary_path = MORPHOLOGY_DIR / "cross_city_d2d3_summary.csv"
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        row = df[df["city"] == city_key]
        if not row.empty:
            return row.iloc[0].to_dict()

    # 尝试单城市文件
    d3_path = MORPHOLOGY_DIR / f"{city_key}_d3_indicators.csv"
    if d3_path.exists():
        df = pd.read_csv(d3_path)
        return df.iloc[0].to_dict()

    return None


# ============================================================================
# 7. 主流程：单城市处理
# ============================================================================

def process_city(
    city_key: str,
    city_info: dict,
    mc_samples: int = 10000,
    sobol_samples: int = 4096,
    skip_sobol: bool = False,
) -> Dict:
    """处理单个城市的完整能源模拟+不确定性分析流程。"""

    log.info(f"\n{'='*60}")
    log.info(f"处理: {city_info['name_en']} ({city_info['name_cn']}) — {city_info['climate_zone']}")
    log.info(f"{'='*60}")

    t0 = time.time()
    result = {
        "city": city_key,
        "name_en": city_info["name_en"],
        "name_cn": city_info["name_cn"],
        "climate_zone": city_info["climate_zone"],
    }

    # ── 1. D1: 气候指标 ──
    log.info("  [D1] 解析 TMY 数据...")
    tmy_df = parse_pvgis_tmy(city_key)
    d1 = compute_d1_indicators(tmy_df, city_info)
    result.update(d1)
    log.info(f"  D1: GHI={d1['d1_1_ghi_annual_kwh']} kWh/m²/yr, "
             f"T={d1['d1_2_temp_annual_c']}°C, "
             f"日照={d1['d1_4_sunshine_hours']}h")

    # 保存 D1
    pd.DataFrame([d1]).to_csv(RESULTS_DIR / f"{city_key}_d1_climate.csv", index=False)

    # ── 2. 确定性模拟 ──
    log.info("  [Deterministic] pvlib 模拟...")
    det = run_pvlib_simulation(tmy_df, city_info)
    result.update(det)

    specific_yield = det.get("specific_yield_kwh_kwp")
    pd.DataFrame([det]).to_csv(RESULTS_DIR / f"{city_key}_deterministic.csv", index=False)

    # ── 3. D3 补充：结合形态数据计算总发电量和自给率 ──
    morph = load_morphology_data(city_key)
    if morph is not None and specific_yield is not None:
        total_deployable_mw = morph.get("d3_4_total_deployable_mw", 0)
        if total_deployable_mw and total_deployable_mw > 0:
            total_gen_gwh = total_deployable_mw * specific_yield / 1000  # GWh/yr
            result["d3_total_generation_gwh"] = round(total_gen_gwh, 2)

            # 自给率估算
            n_buildings = morph.get("n_buildings_analyzed", 0)
            avg_floors = morph.get("d2_1_height_mean", 18) / 3.0
            avg_area = morph.get("d2_3_roof_area_mean", 200)
            total_gfa = n_buildings * avg_area * avg_floors  # 估算总建筑面积
            total_demand_gwh = total_gfa * city_info["residential_demand_kwh_m2"] / 1e6
            if total_demand_gwh > 0:
                self_sufficiency = total_gen_gwh / total_demand_gwh
                result["d3_self_sufficiency_ratio"] = round(self_sufficiency, 4)
                log.info(f"  D3 补充: 总发电={total_gen_gwh:.1f} GWh/yr, "
                         f"自给率={self_sufficiency:.1%}")

    # ── 4. D4: 经济性 ──
    log.info("  [D4] 经济性计算...")
    d4 = compute_d4_economics(specific_yield, city_info, morph)
    result.update(d4)
    log.info(f"  D4: LCOE={d4['d4_1_lcoe_cny_kwh']} CNY/kWh, "
             f"PBT={d4['d4_2_pbt_years']} yr")

    pd.DataFrame([d4]).to_csv(RESULTS_DIR / f"{city_key}_d4_economics.csv", index=False)

    # ── 5. D5: Monte Carlo ──
    log.info("  [D5] Monte Carlo 不确定性分析...")
    ghi_annual = d1.get("d1_1_ghi_annual_kwh", 1000)
    elec_price = city_info["electricity_price"]

    mc_result = run_monte_carlo(ghi_annual, elec_price, n_samples=mc_samples)
    result.update(mc_result)

    pd.DataFrame([mc_result]).to_csv(RESULTS_DIR / f"{city_key}_mc_summary.csv", index=False)

    # ── 6. D5: Sobol 分析 ──
    if not skip_sobol:
        log.info("  [D5] Sobol 敏感性分析...")
        sobol = run_sobol_analysis(ghi_annual, elec_price, n_samples=sobol_samples)
        result.update(sobol)

        pd.DataFrame([sobol]).to_csv(RESULTS_DIR / f"{city_key}_sobol_indices.csv", index=False)
    else:
        log.info("  [D5] Sobol 分析已跳过 (--skip-sobol)")

    elapsed = time.time() - t0
    result["total_elapsed_s"] = round(elapsed, 1)
    log.info(f"  总耗时: {elapsed:.1f}s")

    return result


# ============================================================================
# 8. 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Paper 4 Step 4: 能源模拟 + MC不确定性 + Sobol敏感性分析"
    )
    parser.add_argument("--city", type=str, default=None)
    parser.add_argument("--mc-samples", type=int, default=10000,
                        help="MC 样本数 (默认 10000, 测试时可用 1000)")
    parser.add_argument("--sobol-samples", type=int, default=4096,
                        help="Sobol 样本数 (默认 4096, 必须是2的幂)")
    parser.add_argument("--skip-sobol", action="store_true",
                        help="跳过 Sobol 分析（节省时间）")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Paper 4 — Step 4: 能源模拟与不确定性分析")
    log.info(f"  MC samples: {args.mc_samples}")
    log.info(f"  Sobol samples: {args.sobol_samples}")
    log.info("=" * 60)

    cities = CITIES
    if args.city:
        key = args.city.lower()
        if key not in CITIES:
            log.error(f"未知城市: {args.city}")
            sys.exit(1)
        cities = {key: CITIES[key]}

    all_results = []
    for city_key, city_info in cities.items():
        result = process_city(
            city_key, city_info,
            mc_samples=args.mc_samples,
            sobol_samples=args.sobol_samples,
            skip_sobol=args.skip_sobol,
        )
        all_results.append(result)

    # ── 汇总 ──
    summary_df = pd.DataFrame(all_results)
    summary_path = RESULTS_DIR / "cross_city_d1d4d5.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    log.info(f"\n汇总表: {summary_path}")

    # 打印关键对比
    log.info("\n" + "=" * 60)
    log.info("五城市 D1/D4/D5 关键指标对比")
    log.info("=" * 60)

    key_cols = [
        "name_en", "climate_zone",
        "d1_1_ghi_annual_kwh", "d1_2_temp_annual_c", "d1_4_sunshine_hours",
        "specific_yield_kwh_kwp",
        "d4_1_lcoe_cny_kwh", "d4_2_pbt_years",
        "d5_1_yield_cv", "d5_3_prob_pbt_le_15yr",
    ]
    key_cols = [c for c in key_cols if c in summary_df.columns]

    if HAS_TABULATE:
        from tabulate import tabulate as tab_fmt
        print("\n" + tab_fmt(summary_df[key_cols], headers="keys",
                             tablefmt="grid", showindex=False, floatfmt=".3f"))
    else:
        print(summary_df[key_cols].to_string(index=False))

    # Sobol 主导因子对比
    sobol_cols = [c for c in summary_df.columns if "dominant_factor" in c]
    if sobol_cols:
        log.info("\nSobol 主导敏感因子:")
        for _, row in summary_df.iterrows():
            log.info(f"  {row['name_en']:10s}: "
                     f"发电量→{row.get('d5_5_dominant_factor_yield', 'N/A')}, "
                     f"PBT→{row.get('d5_5_dominant_factor_pbt', 'N/A')}")

    log.info("\nStep 4 完成。")
    log.info("下一步: python scripts/05_fdsi_scoring.py (FDSI 综合评分)")


if __name__ == "__main__":
    main()
