#!/usr/bin/env python3
"""
============================================================================
Paper 4 - Script 06: Extended Economic & Environmental Analysis
============================================================================
新增分析（Phase 2 扩展）：
  1. NPV（净现值）25年
  2. IRR（内部收益率）
  3. CO₂减排量（年度 + 25年累积）
  4. 月度发电-用电匹配（月度自给率）

电网区域排放因子（2022年中国区域电网 CO₂ 排放因子）:
  东北: 哈尔滨、长春、沈阳
  华北: 北京、济南、西安
  华中: 长沙、武汉、南京
  华南: 深圳、广州、厦门
  云南: 昆明
  贵州: 贵阳
  四川: 成都

输出:
  results/paper4_summary/table_npv_irr_co2.csv
  results/paper4_summary/table_monthly_generation.csv
  results/paper4_summary/table_cashflow_25yr.csv

用法：
  python scripts/06_extended_analysis.py
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
    from scipy.optimize import brentq
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not available. IRR will be approximated.")

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
ENERGY_DIR = PROJECT_DIR / "results" / "energy"
PVGIS_DIR = PROJECT_DIR / "results" / "pvgis"
RESULTS_DIR = PROJECT_DIR / "results" / "paper4_summary"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 城市参数
# ============================================================================

CITY_PARAMS = {
    "harbin":    {"name_en": "Harbin",    "name_cn": "哈尔滨", "zone": "Severe Cold",
                  "grid_region": "东北", "elec_price": 0.51, "demand_kwh_m2yr": 35},
    "changchun": {"name_en": "Changchun", "name_cn": "长春",   "zone": "Severe Cold",
                  "grid_region": "东北", "elec_price": 0.52, "demand_kwh_m2yr": 33},
    "shenyang":  {"name_en": "Shenyang",  "name_cn": "沈阳",   "zone": "Severe Cold",
                  "grid_region": "东北", "elec_price": 0.53, "demand_kwh_m2yr": 32},
    "beijing":   {"name_en": "Beijing",   "name_cn": "北京",   "zone": "Cold",
                  "grid_region": "华北", "elec_price": 0.4883, "demand_kwh_m2yr": 30},
    "jinan":     {"name_en": "Jinan",     "name_cn": "济南",   "zone": "Cold",
                  "grid_region": "华北", "elec_price": 0.547, "demand_kwh_m2yr": 30},
    "xian":      {"name_en": "Xian",      "name_cn": "西安",   "zone": "Cold",
                  "grid_region": "华北", "elec_price": 0.498, "demand_kwh_m2yr": 28},
    "changsha":  {"name_en": "Changsha",  "name_cn": "长沙",   "zone": "HSCW",
                  "grid_region": "华中", "elec_price": 0.588, "demand_kwh_m2yr": 28},
    "wuhan":     {"name_en": "Wuhan",     "name_cn": "武汉",   "zone": "HSCW",
                  "grid_region": "华中", "elec_price": 0.558, "demand_kwh_m2yr": 35},
    "nanjing":   {"name_en": "Nanjing",   "name_cn": "南京",   "zone": "HSCW",
                  "grid_region": "华中", "elec_price": 0.558, "demand_kwh_m2yr": 33},
    "shenzhen":  {"name_en": "Shenzhen",  "name_cn": "深圳",   "zone": "HSWW",
                  "grid_region": "华南", "elec_price": 0.68, "demand_kwh_m2yr": 40},
    "guangzhou": {"name_en": "Guangzhou", "name_cn": "广州",   "zone": "HSWW",
                  "grid_region": "华南", "elec_price": 0.68, "demand_kwh_m2yr": 42},
    "xiamen":    {"name_en": "Xiamen",    "name_cn": "厦门",   "zone": "HSWW",
                  "grid_region": "华南", "elec_price": 0.618, "demand_kwh_m2yr": 38},
    "kunming":   {"name_en": "Kunming",   "name_cn": "昆明",   "zone": "Mild",
                  "grid_region": "云南", "elec_price": 0.45, "demand_kwh_m2yr": 22},
    "guiyang":   {"name_en": "Guiyang",   "name_cn": "贵阳",   "zone": "Mild",
                  "grid_region": "贵州", "elec_price": 0.468, "demand_kwh_m2yr": 20},
    "chengdu":   {"name_en": "Chengdu",   "name_cn": "成都",   "zone": "Mild",
                  "grid_region": "四川", "elec_price": 0.502, "demand_kwh_m2yr": 25},
}

# 2022年各电网区域 CO₂ 排放因子（tCO₂/MWh）
# 来源：中国区域电网基准线排放因子（NDRC 2023）
GRID_EMISSION_FACTORS = {
    "东北": 0.8437,
    "华北": 0.8843,
    "华中": 0.5257,
    "华南": 0.4267,
    "云南": 0.1578,   # 高度水电，低排放
    "贵州": 0.6158,
    "四川": 0.1373,   # 高度水电，最低排放
}

# PV 系统统一参数
PV_PARAMS = {
    "pv_cost_cny_wp": 3.0,
    "system_wp_per_m2": 200,       # W/m² installed
    "annual_degradation": 0.005,   # 0.5%/yr
    "system_lifetime": 25,
    "discount_rate": 0.06,
    "om_ratio": 0.01,              # O&M as fraction of CapEx per year
    "module_efficiency": 0.20,
    "roof_utilization": 0.70,
}


# ============================================================================
# 1. 从现有能源数据加载基础指标
# ============================================================================

def load_energy_data() -> pd.DataFrame:
    """加载 15 城市能源汇总数据。"""
    path = ENERGY_DIR / "cross_city_d1d4d5.csv"
    df = pd.read_csv(path)
    log.info(f"  加载能源数据: {len(df)} 城市")
    return df


# ============================================================================
# 2. NPV 计算
# ============================================================================

def compute_npv(
    annual_yield_kwh_kwp: float,   # kWh/kWp/yr (year 1)
    elec_price: float,             # CNY/kWh
    pv_cost_cny_wp: float,        # CNY/Wp
    discount_rate: float,
    degradation: float,
    lifetime: int,
    om_ratio: float,
) -> float:
    """
    计算 25年 NPV（per kWp installed）。

    NPV = -C_inv + Σ[(annual_revenue_t - om_cost) / (1+r)^t]

    annual_revenue_t = annual_yield_kwh_kwp * (1-degradation)^(t-1) * elec_price
    om_cost = C_inv * om_ratio (constant)
    C_inv = pv_cost_cny_wp * 1000  # per kWp
    """
    C_inv = pv_cost_cny_wp * 1000  # CNY/kWp
    om_annual = C_inv * om_ratio   # CNY/kWp/yr

    npv = -C_inv
    for t in range(1, lifetime + 1):
        yield_t = annual_yield_kwh_kwp * (1 - degradation) ** (t - 1)
        revenue_t = yield_t * elec_price
        net_t = revenue_t - om_annual
        npv += net_t / (1 + discount_rate) ** t

    return npv


def compute_irr(
    annual_yield_kwh_kwp: float,
    elec_price: float,
    pv_cost_cny_wp: float,
    degradation: float,
    lifetime: int,
    om_ratio: float,
) -> float:
    """
    计算 IRR（内部收益率），使 NPV=0 的折现率。
    使用 brentq 求根，搜索区间 [-0.5, 2.0]。
    """
    if not HAS_SCIPY:
        # 简单近似：IRR ≈ 1/PBT
        C_inv = pv_cost_cny_wp * 1000
        annual_revenue = annual_yield_kwh_kwp * elec_price
        om_annual = C_inv * om_ratio
        net_annual = annual_revenue - om_annual
        if net_annual <= 0:
            return -0.1
        return net_annual / C_inv  # rough approximation

    def npv_func(r):
        return compute_npv(
            annual_yield_kwh_kwp, elec_price, pv_cost_cny_wp,
            r, degradation, lifetime, om_ratio
        )

    try:
        irr = brentq(npv_func, -0.5, 2.0, xtol=1e-6)
    except ValueError:
        irr = np.nan  # No solution in range

    return irr


# ============================================================================
# 3. CO₂ 减排计算
# ============================================================================

def compute_co2_reduction(
    annual_yield_kwh_kwp: float,   # kWh/kWp/yr (year 1)
    emission_factor: float,        # tCO₂/MWh
    degradation: float,
    lifetime: int,
) -> Dict[str, float]:
    """
    计算年度和累积 CO₂ 减排量（per kWp installed）。

    annual_co2_t = annual_yield_t * emission_factor / 1000  # MWh → kWh除以1000
    """
    annual_co2_yr1 = annual_yield_kwh_kwp * emission_factor / 1000  # tCO₂/yr/kWp

    lifetime_co2 = 0
    for t in range(1, lifetime + 1):
        yield_t = annual_yield_kwh_kwp * (1 - degradation) ** (t - 1)
        co2_t = yield_t * emission_factor / 1000  # tCO₂
        lifetime_co2 += co2_t

    return {
        "co2_annual_tco2_kwp": round(annual_co2_yr1, 4),
        "co2_lifetime_tco2_kwp": round(lifetime_co2, 2),
        "co2_annual_kgco2_kwh": round(emission_factor, 4),  # kg/kWh = emission factor in tCO2/MWh
    }


# ============================================================================
# 4. 月度发电量分析（从 TMY 数据）
# ============================================================================

def compute_monthly_generation(
    city: str,
    specific_yield_annual: float,  # kWh/kWp/yr (from simulation)
) -> Optional[pd.DataFrame]:
    """
    从 PVGIS TMY 数据提取月度发电量分布。

    使用 GHI 月度比例分配年度发电量
    (比直接用 pvlib 模拟更简单，且与已有数据一致)
    """
    tmy_path = PVGIS_DIR / f"{city}_tmy.csv"
    if not tmy_path.exists():
        log.warning(f"  TMY 文件不存在: {tmy_path}")
        return None

    # 解析 PVGIS TMY
    lines = tmy_path.read_text(encoding="utf-8").strip().split("\n")
    header_idx = None
    for i, line in enumerate(lines):
        if "time" in line.lower() and ("ghi" in line.lower() or "G(h)" in line.lower()):
            header_idx = i
            break

    if header_idx is None:
        for i, line in enumerate(lines):
            parts = line.split(",")
            if len(parts) >= 4:
                try:
                    float(parts[1])
                    header_idx = i - 1 if i > 0 else 0
                    break
                except ValueError:
                    continue

    if header_idx is None:
        return None

    try:
        df_tmy = pd.read_csv(tmy_path, skiprows=header_idx, encoding="utf-8")
        # 找 GHI 列
        ghi_col = None
        for col in df_tmy.columns:
            if "G(h)" in col or "ghi" in col.lower() or col.strip() == "G(h)":
                ghi_col = col
                break
        if ghi_col is None:
            # 第2列通常是 GHI
            ghi_col = df_tmy.columns[1]

        df_tmy = df_tmy[df_tmy[ghi_col].apply(lambda x: str(x).replace('.','').replace('-','').isdigit())]
        df_tmy[ghi_col] = pd.to_numeric(df_tmy[ghi_col], errors='coerce')

        # 8760行 → 按月分配
        months_per_row = 12 / len(df_tmy) * len(df_tmy)
        if len(df_tmy) >= 8760:
            df_tmy = df_tmy.iloc[:8760]
            month_idx = np.repeat(np.arange(1, 13), [744, 672, 744, 720, 744, 720,
                                                       744, 744, 720, 744, 720, 744])[:8760]
            df_tmy["month"] = month_idx

            ghi_monthly = df_tmy.groupby("month")[ghi_col].sum()
            ghi_total = ghi_monthly.sum()
            if ghi_total > 0:
                monthly_fraction = ghi_monthly / ghi_total
                monthly_yield = monthly_fraction * specific_yield_annual
                return pd.DataFrame({
                    "month": range(1, 13),
                    "ghi_monthly_kwh": ghi_monthly.values,
                    "generation_kwh_kwp": monthly_yield.values,
                    "fraction": monthly_fraction.values,
                })
    except Exception as e:
        log.warning(f"  月度解析失败: {e}")

    # 备用：使用典型月度分布（按气候区）
    return None


def compute_monthly_self_sufficiency(
    city: str,
    monthly_gen: pd.DataFrame,
    demand_kwh_m2yr: float,
    deployable_m2_per_kwp: float = 5.0,  # m²/kWp (约 200Wp/m²)
) -> pd.DataFrame:
    """计算月度自给率。"""
    # 年度需求 kWh/kWp 换算
    annual_demand_kwh_kwp = demand_kwh_m2yr * deployable_m2_per_kwp

    # 月度需求假设（简单均匀分配，可改为实测比例）
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    monthly_demand = pd.Series([d / 365 * annual_demand_kwh_kwp for d in month_days])

    result = monthly_gen.copy()
    result["demand_kwh_kwp"] = monthly_demand.values
    result["self_sufficiency"] = (result["generation_kwh_kwp"] / result["demand_kwh_kwp"]).clip(upper=1.0)
    return result


# ============================================================================
# 5. 25 年现金流
# ============================================================================

def compute_cashflow_25yr(
    annual_yield_kwh_kwp: float,
    elec_price: float,
    pv_cost_cny_wp: float,
    degradation: float,
    om_ratio: float,
    discount_rate: float,
) -> pd.DataFrame:
    """生成 25 年逐年现金流数据（per kWp）。"""
    C_inv = pv_cost_cny_wp * 1000
    om_annual = C_inv * om_ratio

    rows = []
    cumulative = -C_inv
    for t in range(1, 26):
        yield_t = annual_yield_kwh_kwp * (1 - degradation) ** (t - 1)
        revenue_t = yield_t * elec_price
        net_t = revenue_t - om_annual
        pv_net_t = net_t / (1 + discount_rate) ** t
        cumulative += pv_net_t
        rows.append({
            "year": t,
            "yield_kwh_kwp": round(yield_t, 1),
            "revenue_cny_kwp": round(revenue_t, 0),
            "om_cost_cny_kwp": round(om_annual, 0),
            "net_cash_cny_kwp": round(net_t, 0),
            "pv_net_cash_cny_kwp": round(pv_net_t, 0),
            "cumulative_npv_cny_kwp": round(cumulative, 0),
        })

    return pd.DataFrame(rows)


# ============================================================================
# 主函数
# ============================================================================

def main():
    log.info("=" * 60)
    log.info("Paper 4 — Step 6: Extended Economic & Environmental Analysis")
    log.info("=" * 60)

    # ── 加载能源数据 ──
    log.info("\n[1] 加载基础数据...")
    energy_df = load_energy_data()

    # 确保 city 列存在
    if "city" not in energy_df.columns:
        log.error("energy df missing 'city' column")
        sys.exit(1)

    # ── NPV / IRR / CO₂ ──
    log.info("\n[2] 计算 NPV、IRR、CO₂ 减排...")

    npv_irr_rows = []
    cashflow_all = []

    for city, params in CITY_PARAMS.items():
        row_e = energy_df[energy_df["city"] == city]
        if len(row_e) == 0:
            log.warning(f"  ⚠ {city}: 能源数据缺失")
            continue

        row_e = row_e.iloc[0]
        specific_yield = float(row_e.get("specific_yield_kwh_kwp", 1200))
        elec_price = params["elec_price"]
        grid_region = params["grid_region"]
        emission_factor = GRID_EMISSION_FACTORS[grid_region]

        # NPV
        npv = compute_npv(
            annual_yield_kwh_kwp=specific_yield,
            elec_price=elec_price,
            pv_cost_cny_wp=PV_PARAMS["pv_cost_cny_wp"],
            discount_rate=PV_PARAMS["discount_rate"],
            degradation=PV_PARAMS["annual_degradation"],
            lifetime=PV_PARAMS["system_lifetime"],
            om_ratio=PV_PARAMS["om_ratio"],
        )

        # IRR
        irr = compute_irr(
            annual_yield_kwh_kwp=specific_yield,
            elec_price=elec_price,
            pv_cost_cny_wp=PV_PARAMS["pv_cost_cny_wp"],
            degradation=PV_PARAMS["annual_degradation"],
            lifetime=PV_PARAMS["system_lifetime"],
            om_ratio=PV_PARAMS["om_ratio"],
        )

        # CO₂
        co2 = compute_co2_reduction(
            annual_yield_kwh_kwp=specific_yield,
            emission_factor=emission_factor,
            degradation=PV_PARAMS["annual_degradation"],
            lifetime=PV_PARAMS["system_lifetime"],
        )

        row_out = {
            "city": city,
            "name_en": params["name_en"],
            "name_cn": params["name_cn"],
            "climate_zone": params["zone"],
            "grid_region": grid_region,
            "specific_yield_kwh_kwp": round(specific_yield, 1),
            "elec_price_cny_kwh": elec_price,
            "npv_cny_kwp": round(npv, 0),
            "irr_pct": round(irr * 100, 2) if not np.isnan(irr) else None,
            "emission_factor_tco2_mwh": emission_factor,
            **co2,
        }
        npv_irr_rows.append(row_out)

        log.info(f"  {params['name_en']:12s}: NPV={npv:8,.0f} CNY/kWp, IRR={irr*100:.1f}%, "
                 f"CO₂={co2['co2_annual_tco2_kwp']:.4f} tCO₂/yr/kWp")

        # 现金流（选代表性城市：每气候区各1）
        representative = {"harbin", "beijing", "changsha", "shenzhen", "kunming"}
        if city in representative:
            cf = compute_cashflow_25yr(
                annual_yield_kwh_kwp=specific_yield,
                elec_price=elec_price,
                pv_cost_cny_wp=PV_PARAMS["pv_cost_cny_wp"],
                degradation=PV_PARAMS["annual_degradation"],
                om_ratio=PV_PARAMS["om_ratio"],
                discount_rate=PV_PARAMS["discount_rate"],
            )
            cf.insert(0, "city", city)
            cf.insert(1, "name_en", params["name_en"])
            cashflow_all.append(cf)

    # 保存 NPV/IRR/CO₂
    df_npv = pd.DataFrame(npv_irr_rows)
    out_path = RESULTS_DIR / "table_npv_irr_co2.csv"
    df_npv.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info(f"\n  已保存: {out_path}")

    # 保存现金流
    if cashflow_all:
        df_cf = pd.concat(cashflow_all, ignore_index=True)
        cf_path = RESULTS_DIR / "table_cashflow_25yr.csv"
        df_cf.to_csv(cf_path, index=False, encoding="utf-8-sig")
        log.info(f"  已保存: {cf_path}")

    # ── 月度发电分析 ──
    log.info("\n[3] 月度发电量分析...")

    monthly_all = []
    for city, params in CITY_PARAMS.items():
        row_e = energy_df[energy_df["city"] == city]
        if len(row_e) == 0:
            continue
        specific_yield = float(row_e.iloc[0].get("specific_yield_kwh_kwp", 1200))

        monthly = compute_monthly_generation(city, specific_yield)
        if monthly is None:
            # 备用：均匀分配
            monthly = pd.DataFrame({
                "month": range(1, 13),
                "ghi_monthly_kwh": [specific_yield / 12] * 12,
                "generation_kwh_kwp": [specific_yield / 12] * 12,
                "fraction": [1/12] * 12,
            })

        monthly.insert(0, "city", city)
        monthly.insert(1, "name_en", params["name_en"])
        monthly.insert(2, "climate_zone", params["zone"])

        # 自给率
        ss = compute_monthly_self_sufficiency(
            city, monthly,
            demand_kwh_m2yr=params["demand_kwh_m2yr"],
        )
        monthly["demand_kwh_kwp"] = ss["demand_kwh_kwp"]
        monthly["self_sufficiency"] = ss["self_sufficiency"]
        monthly_all.append(monthly)
        log.info(f"  {params['name_en']:12s}: 年度发电={specific_yield:.0f} kWh/kWp")

    df_monthly = pd.concat(monthly_all, ignore_index=True)
    monthly_path = RESULTS_DIR / "table_monthly_generation.csv"
    df_monthly.to_csv(monthly_path, index=False, encoding="utf-8-sig")
    log.info(f"\n  已保存: {monthly_path}")

    # ── 汇总打印 ──
    log.info("\n" + "=" * 60)
    log.info("NPV / IRR / CO₂ 汇总")
    log.info("=" * 60)
    display_cols = ["name_en", "climate_zone", "npv_cny_kwp", "irr_pct",
                    "co2_annual_tco2_kwp", "co2_lifetime_tco2_kwp"]
    df_display = df_npv[display_cols].sort_values("npv_cny_kwp", ascending=False)
    log.info("\n" + df_display.to_string(index=False))

    log.info("\nStep 6 完成。")
    log.info("下一步: python scripts/07_paper_figures.py")


if __name__ == "__main__":
    main()
