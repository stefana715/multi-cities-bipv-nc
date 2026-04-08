#!/usr/bin/env python3
"""
============================================================================
Paper 4 Level 2 Output: BIPV Suitability Lookup Tool (CLI)
============================================================================
命令行查询工具：输入气候区或城市名，输出 FDSI 综合评分和部署建议。

用法：
  python tools/bipv_lookup.py                          # 交互模式
  python tools/bipv_lookup.py --city shenzhen          # 查询单个城市
  python tools/bipv_lookup.py --zone cold              # 按气候区查询
  python tools/bipv_lookup.py --compare all            # 五城市对比
  python tools/bipv_lookup.py --export results.csv     # 导出全部数据
============================================================================
"""

import argparse
import sys
import json
from typing import Optional

# ============================================================================
# 数据库（来自 Step 1-5 计算结果）
# ============================================================================

DATABASE = {
    "shenzhen": {
        "name_en": "Shenzhen", "name_cn": "深圳",
        "zone": "hsww", "zone_label": "Hot Summer Warm Winter (夏热冬暖)",
        "province": "Guangdong", "lat": 22.54, "lon": 114.06,
        "fdsi": 0.647, "rank": 1,
        "suitability": "High", "uncertainty": "Low",
        "dominant_morphology": "high_rise",
        "dimensions": {
            "D1_Climate":    {"score": 0.671, "ghi_kwh": 1561, "temp_c": 23.1, "sunshine_h": 3842},
            "D2_Morphology": {"score": 0.117, "height_m": 40.7, "density": 0.667, "roof_area_m2": 329.6, "far": 0.667},
            "D3_Technical":  {"score": 0.761, "yield_kwh_kwp": 1299, "shading_factor": 0.983, "deployable_mw": 329.6},
            "D4_Economic":   {"score": 0.838, "lcoe": 0.214, "pbt_yr": 3.52, "elec_price": 0.68},
            "D5_Uncertainty":{"score": 0.913, "pbt_ci_width": 1.21, "lcoe_std": 0.012, "dominant_factor": "pv_cost"},
        },
        "recommendation": "高适宜性。强太阳能资源、有利的经济性和可靠的评估结果支持大规模实施。"
                          "建议优先在中低层住宅区部署，高层区域考虑立面BIPV。",
    },
    "beijing": {
        "name_en": "Beijing", "name_cn": "北京",
        "zone": "cold", "zone_label": "Cold (寒冷)",
        "province": "Beijing", "lat": 39.90, "lon": 116.40,
        "fdsi": 0.575, "rank": 2,
        "suitability": "High", "uncertainty": "Low",
        "dominant_morphology": "mid_rise",
        "dimensions": {
            "D1_Climate":    {"score": 0.735, "ghi_kwh": 1578, "temp_c": 13.2, "sunshine_h": 4156},
            "D2_Morphology": {"score": 0.712, "height_m": 18.5, "density": 0.052, "roof_area_m2": 475.9, "far": 0.052},
            "D3_Technical":  {"score": 0.648, "yield_kwh_kwp": 1358, "shading_factor": 0.966, "deployable_mw": 475.9},
            "D4_Economic":   {"score": 0.573, "lcoe": 0.205, "pbt_yr": 4.74, "elec_price": 0.4883},
            "D5_Uncertainty":{"score": 0.567, "pbt_ci_width": 1.85, "lcoe_std": 0.015, "dominant_factor": "pv_cost"},
        },
        "recommendation": "高适宜性。多层住宅形态提供良好的屋顶条件，日照充足。"
                          "适合在老城区更新和新建住宅区同步推进BIPV部署。",
    },
    "kunming": {
        "name_en": "Kunming", "name_cn": "昆明",
        "zone": "mild", "zone_label": "Mild (温和)",
        "province": "Yunnan", "lat": 25.04, "lon": 102.68,
        "fdsi": 0.552, "rank": 3,
        "suitability": "Medium", "uncertainty": "Medium",
        "dominant_morphology": "mid_high",
        "dimensions": {
            "D1_Climate":    {"score": 1.000, "ghi_kwh": 1650, "temp_c": 15.8, "sunshine_h": 4520},
            "D2_Morphology": {"score": 0.578, "height_m": 27.4, "density": 0.074, "roof_area_m2": 262.3, "far": 0.074},
            "D3_Technical":  {"score": 1.000, "yield_kwh_kwp": 1405, "shading_factor": 0.971, "deployable_mw": 262.3},
            "D4_Economic":   {"score": 0.340, "lcoe": 0.198, "pbt_yr": 4.98, "elec_price": 0.45},
            "D5_Uncertainty":{"score": 0.421, "pbt_ci_width": 2.31, "lcoe_std": 0.018, "dominant_factor": "pv_cost"},
        },
        "recommendation": "中等适宜性。GHI全国最高（高原强辐射），但电价偏低影响经济性。"
                          "建议结合当地补贴政策推进，优先选择日照条件最好的朝南坡地住宅。",
    },
    "changsha": {
        "name_en": "Changsha", "name_cn": "长沙",
        "zone": "hscw", "zone_label": "Hot Summer Cold Winter (夏热冬冷)",
        "province": "Hunan", "lat": 28.23, "lon": 112.94,
        "fdsi": 0.332, "rank": 4,
        "suitability": "Low", "uncertainty": "High",
        "dominant_morphology": "mid_high",
        "dimensions": {
            "D1_Climate":    {"score": 0.000, "ghi_kwh": 1378, "temp_c": 17.5, "sunshine_h": 3280},
            "D2_Morphology": {"score": 0.458, "height_m": 22.0, "density": 0.046, "roof_area_m2": 361.8, "far": 0.046},
            "D3_Technical":  {"score": 0.000, "yield_kwh_kwp": 1162, "shading_factor": 0.958, "deployable_mw": 361.8},
            "D4_Economic":   {"score": 0.280, "lcoe": 0.239, "pbt_yr": 4.59, "elec_price": 0.588},
            "D5_Uncertainty":{"score": 0.187, "pbt_ci_width": 2.65, "lcoe_std": 0.021, "dominant_factor": "pv_cost"},
        },
        "recommendation": "较低适宜性。梅雨季严重影响太阳能资源稳定性，GHI全国最低。"
                          "BIPV部署应选择性推进，优先遮挡少、朝向好的建筑，并需详细的场地评估。",
    },
    "harbin": {
        "name_en": "Harbin", "name_cn": "哈尔滨",
        "zone": "severe_cold", "zone_label": "Severe Cold (严寒)",
        "province": "Heilongjiang", "lat": 45.75, "lon": 126.65,
        "fdsi": 0.273, "rank": 5,
        "suitability": "Low", "uncertainty": "High",
        "dominant_morphology": "mid_high",
        "dimensions": {
            "D1_Climate":    {"score": 0.169, "ghi_kwh": 1424, "temp_c": 5.2, "sunshine_h": 3650},
            "D2_Morphology": {"score": 0.432, "height_m": 21.3, "density": 0.374, "roof_area_m2": 1362.0, "far": 0.374},
            "D3_Technical":  {"score": 0.240, "yield_kwh_kwp": 1259, "shading_factor": 0.952, "deployable_mw": 1362.0},
            "D4_Economic":   {"score": 0.000, "lcoe": 0.221, "pbt_yr": 4.90, "elec_price": 0.51},
            "D5_Uncertainty":{"score": 0.000, "pbt_ci_width": 2.88, "lcoe_std": 0.024, "dominant_factor": "pv_cost"},
        },
        "recommendation": "较低适宜性。冬季严寒、积雪覆盖影响发电，不确定性最高。"
                          "适合在夏季日照充足的条件下小规模试点，需考虑组件耐寒和除雪维护成本。",
    },
}

# 气候区到城市的映射
ZONE_MAP = {
    "severe_cold": "harbin",
    "cold": "beijing",
    "hscw": "changsha",
    "hsww": "shenzhen",
    "mild": "kunming",
}


def print_city_report(city_key: str, verbose: bool = True):
    """打印单个城市的详细报告。"""
    data = DATABASE.get(city_key)
    if not data:
        print(f"  未找到城市: {city_key}")
        return

    print(f"\n{'═'*60}")
    print(f"  {data['name_en']} ({data['name_cn']}) — {data['zone_label']}")
    print(f"  {data['lat']}°N, {data['lon']}°E · {data['province']}")
    print(f"{'═'*60}")

    print(f"\n  FDSI Score:     {data['fdsi']:.3f}  (Rank #{data['rank']})")
    print(f"  Suitability:    {data['suitability']}")
    print(f"  Uncertainty:    {data['uncertainty']}")
    print(f"  Morphology:     {data['dominant_morphology']}")

    if verbose:
        print(f"\n  {'─'*50}")
        print(f"  Dimension Scores")
        print(f"  {'─'*50}")
        dims = data["dimensions"]
        bar_width = 25
        for dim_key, dim_data in dims.items():
            score = dim_data["score"]
            filled = int(score * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"  {dim_key:20s} {score:.3f}  [{bar}]")

        print(f"\n  {'─'*50}")
        print(f"  Key Indicators")
        print(f"  {'─'*50}")
        d1 = dims["D1_Climate"]
        d2 = dims["D2_Morphology"]
        d3 = dims["D3_Technical"]
        d4 = dims["D4_Economic"]
        d5 = dims["D5_Uncertainty"]

        print(f"  GHI:              {d1['ghi_kwh']:,} kWh/m²/yr")
        print(f"  Annual Temp:      {d1['temp_c']}°C")
        print(f"  Sunshine Hours:   {d1['sunshine_h']:,} h/yr")
        print(f"  Specific Yield:   {d3['yield_kwh_kwp']:,} kWh/kWp/yr")
        print(f"  Mean Height:      {d2['height_m']} m")
        print(f"  Building Density: {d2['density']*100:.1f}%")
        print(f"  LCOE:             ¥{d4['lcoe']:.3f}/kWh")
        print(f"  Payback Period:   {d4['pbt_yr']:.1f} years")
        print(f"  PBT 95% CI:      ±{d5['pbt_ci_width']:.2f} yr")
        print(f"  Key Sensitivity:  {d5['dominant_factor']}")

    print(f"\n  {'─'*50}")
    print(f"  Recommendation")
    print(f"  {'─'*50}")
    print(f"  {data['recommendation']}")
    print()


def print_comparison():
    """打印五城市对比表。"""
    print(f"\n{'═'*75}")
    print(f"  Five-City BIPV Suitability Comparison (FDSI Ranking)")
    print(f"{'═'*75}")

    header = f"{'Rank':>4}  {'City':10}  {'Zone':12}  {'FDSI':>6}  {'Suit.':>6}  {'Uncert.':>7}  {'GHI':>5}  {'PBT':>5}"
    print(f"\n  {header}")
    print(f"  {'─'*len(header)}")

    sorted_cities = sorted(DATABASE.items(), key=lambda x: x[1]["rank"])
    for key, data in sorted_cities:
        d1 = data["dimensions"]["D1_Climate"]
        d4 = data["dimensions"]["D4_Economic"]
        print(f"  {data['rank']:>4}  {data['name_en']:10}  {data['zone']:12}  "
              f"{data['fdsi']:>6.3f}  {data['suitability']:>6}  {data['uncertainty']:>7}  "
              f"{d1['ghi_kwh']:>5}  {d4['pbt_yr']:>5.1f}")

    print()


def interactive_mode():
    """交互模式。"""
    print("\n" + "═" * 60)
    print("  BIPV Suitability Lookup Tool")
    print("  基于 FDSI 五维适宜性评价框架")
    print("═" * 60)
    print("\n  Commands:")
    print("    <city name>    — 查询城市 (e.g., shenzhen, beijing)")
    print("    <zone>         — 按气候区查询 (e.g., cold, hsww)")
    print("    compare        — 五城市对比")
    print("    quit           — 退出")
    print()

    while True:
        try:
            query = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye!")
            break

        if not query:
            continue
        if query in ("quit", "exit", "q"):
            print("  Bye!")
            break
        if query in ("compare", "all", "list"):
            print_comparison()
            continue

        # 尝试匹配城市名
        if query in DATABASE:
            print_city_report(query)
            continue

        # 尝试匹配气候区
        if query in ZONE_MAP:
            print_city_report(ZONE_MAP[query])
            continue

        # 模糊匹配
        matches = [k for k, v in DATABASE.items()
                    if query in k or query in v["name_cn"] or query in v["zone"]]
        if matches:
            for m in matches:
                print_city_report(m)
        else:
            print(f"  未找到 '{query}'。可用: {', '.join(DATABASE.keys())}")
            print(f"  气候区: {', '.join(ZONE_MAP.keys())}")


def main():
    parser = argparse.ArgumentParser(description="BIPV Suitability Lookup Tool")
    parser.add_argument("--city", type=str, help="查询城市")
    parser.add_argument("--zone", type=str, help="按气候区查询")
    parser.add_argument("--compare", type=str, help="对比 (输入 'all')")
    parser.add_argument("--export", type=str, help="导出为 CSV")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    args = parser.parse_args()

    if args.export:
        import csv
        with open(args.export, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "city", "zone", "fdsi", "suitability",
                            "uncertainty", "ghi", "yield", "lcoe", "pbt"])
            for key, data in sorted(DATABASE.items(), key=lambda x: x[1]["rank"]):
                d1 = data["dimensions"]["D1_Climate"]
                d3 = data["dimensions"]["D3_Technical"]
                d4 = data["dimensions"]["D4_Economic"]
                writer.writerow([
                    data["rank"], data["name_en"], data["zone"],
                    data["fdsi"], data["suitability"], data["uncertainty"],
                    d1["ghi_kwh"], d3["yield_kwh_kwp"], d4["lcoe"], d4["pbt_yr"],
                ])
        print(f"  导出完成: {args.export}")
        return

    if args.json:
        print(json.dumps(DATABASE, indent=2, ensure_ascii=False))
        return

    if args.city:
        key = args.city.lower()
        if key in DATABASE:
            print_city_report(key)
        else:
            print(f"  未找到城市: {key}")
        return

    if args.zone:
        zone = args.zone.lower()
        if zone in ZONE_MAP:
            print_city_report(ZONE_MAP[zone])
        else:
            print(f"  未找到气候区: {zone}")
        return

    if args.compare:
        print_comparison()
        return

    # 默认交互模式
    interactive_mode()


if __name__ == "__main__":
    main()
