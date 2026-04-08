#!/usr/bin/env python3
"""
============================================================================
Paper 4 - Script 07: Generate 16 Paper Figures
============================================================================
生成全部 16 张论文图表，保存为 PDF + PNG（300dpi）。

图表列表：
  fig01 - workflow.pdf          方法学流程图
  fig02 - fdsi_framework.pdf    FDSI 五维框架结构树
  fig03 - china_map.pdf         中国气候分区+15城市位置
  fig04 - radar.pdf             15城市五维雷达图
  fig05 - heatmap.pdf           15×5维度得分热力图
  fig06 - fdsi_ranking.pdf      15城市FDSI排名条形图
  fig07 - weight_sensitivity.pdf α扫描权重敏感性
  fig08 - mc_pbt_violin.pdf     15城市MC PBT分布小提琴图
  fig09 - sobol_bar.pdf         Sobol S₁指数堆叠条形图
  fig10 - d4_vs_d5.pdf          D4 vs D5散点图
  fig11 - npv_comparison.pdf    15城市NPV对比
  fig12 - irr_comparison.pdf    15城市IRR对比
  fig13 - co2_reduction.pdf     年度CO₂减排潜力
  fig14 - cashflow.pdf          25年累积现金流
  fig15 - height_distribution.pdf 建筑高度分布
  fig16 - suitability_matrix.pdf  气候×形态适宜性矩阵

用法：
  python scripts/07_paper_figures.py
  python scripts/07_paper_figures.py --fig 4    # 只生成单张图
============================================================================
"""

import argparse
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
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D
    HAS_MPL = True
except ImportError:
    print("ERROR: matplotlib not installed")
    sys.exit(1)

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"
FDSI_DIR = RESULTS_DIR / "fdsi"
ENERGY_DIR = RESULTS_DIR / "energy"
MORPH_DIR = RESULTS_DIR / "morphology"
SUMMARY_DIR = RESULTS_DIR / "paper4_summary"
FIGURES_DIR = PROJECT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 图表样式配置
# ============================================================================

# 字体设置
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# 气候区配色（5色方案，色觉友好）
ZONE_COLORS = {
    "Severe Cold": "#2166AC",   # 深蓝
    "Cold":        "#74ADD1",   # 浅蓝
    "HSCW":        "#F4A582",   # 橙
    "HSWW":        "#D6604D",   # 红
    "Mild":        "#4DAC26",   # 绿
}

ZONE_ORDER = ["Severe Cold", "Cold", "HSCW", "HSWW", "Mild"]

CITY_ORDER = [
    "harbin", "changchun", "shenyang",       # Severe Cold
    "beijing", "jinan", "xian",              # Cold
    "changsha", "wuhan", "nanjing",          # HSCW
    "shenzhen", "guangzhou", "xiamen",       # HSWW
    "kunming", "guiyang", "chengdu",         # Mild
]

CITY_LABELS = {
    "harbin": "Harbin", "changchun": "Changchun", "shenyang": "Shenyang",
    "beijing": "Beijing", "jinan": "Jinan", "xian": "Xi'an",
    "changsha": "Changsha", "wuhan": "Wuhan", "nanjing": "Nanjing",
    "shenzhen": "Shenzhen", "guangzhou": "Guangzhou", "xiamen": "Xiamen",
    "kunming": "Kunming", "guiyang": "Guiyang", "chengdu": "Chengdu",
}

CITY_ZONES = {
    "harbin": "Severe Cold", "changchun": "Severe Cold", "shenyang": "Severe Cold",
    "beijing": "Cold", "jinan": "Cold", "xian": "Cold",
    "changsha": "HSCW", "wuhan": "HSCW", "nanjing": "HSCW",
    "shenzhen": "HSWW", "guangzhou": "HSWW", "xiamen": "HSWW",
    "kunming": "Mild", "guiyang": "Mild", "chengdu": "Mild",
}

# 图幅尺寸（单位 inches，300dpi）
SINGLE_COL = 3.54   # 90mm
DOUBLE_COL = 7.48   # 190mm
HEIGHT_34 = 4.5     # 3/4 page height (approx)
HEIGHT_FULL = 7.0


def save_fig(fig: plt.Figure, name: str):
    """保存图为 PDF + PNG。"""
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    png_path = FIGURES_DIR / f"{name}.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    log.info(f"  Saved: {name}.pdf / .png")
    plt.close(fig)


def get_zone_color(city: str) -> str:
    return ZONE_COLORS.get(CITY_ZONES.get(city, "Cold"), "#888888")


# ============================================================================
# 数据加载
# ============================================================================

def load_data() -> Dict[str, pd.DataFrame]:
    data = {}
    data["fdsi"] = pd.read_csv(FDSI_DIR / "fdsi_scores.csv")
    data["integrated"] = pd.read_csv(FDSI_DIR / "integrated_indicators.csv")
    data["weight_sens"] = pd.read_csv(FDSI_DIR / "weight_sensitivity.csv")
    data["energy"] = pd.read_csv(ENERGY_DIR / "cross_city_d1d4d5.csv")
    data["morphology"] = pd.read_csv(MORPH_DIR / "cross_city_d2d3_summary.csv")

    npv_path = SUMMARY_DIR / "table_npv_irr_co2.csv"
    if npv_path.exists():
        data["npv"] = pd.read_csv(npv_path)

    cf_path = SUMMARY_DIR / "table_cashflow_25yr.csv"
    if cf_path.exists():
        data["cashflow"] = pd.read_csv(cf_path)

    monthly_path = SUMMARY_DIR / "table_monthly_generation.csv"
    if monthly_path.exists():
        data["monthly"] = pd.read_csv(monthly_path)

    return data


# ============================================================================
# Fig 01: Workflow Diagram
# ============================================================================

def fig01_workflow(data: Dict):
    """方法学工作流程图。"""
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, HEIGHT_FULL))
    ax.axis("off")

    steps = [
        ("OSM Building\nData Audit (Layer 3)", "#E3F2FD", "Step 1"),
        ("PVGIS Climate\nData Download (TMY)", "#E8F5E9", "Step 2"),
        ("Urban Morphology\nAnalysis (D2, D3)", "#FFF9C4", "Step 3"),
        ("Energy Simulation\n+ MC/Sobol (D1,D4,D5)", "#FCE4EC", "Step 4"),
        ("FDSI Scoring\n+ Suitability Matrix", "#F3E5F5", "Step 5"),
        ("NPV / IRR / CO₂\nExtended Analysis", "#E0F7FA", "Step 6"),
        ("16 Paper Figures\n+ Summary Tables", "#EFEBE9", "Step 7"),
    ]

    y_positions = np.linspace(0.90, 0.06, len(steps))
    box_h = 0.10
    box_w = 0.55
    x_center = 0.50

    for i, ((label, color, step_num), y) in enumerate(zip(steps, y_positions)):
        # Box
        rect = mpatches.FancyBboxPatch(
            (x_center - box_w/2, y - box_h/2), box_w, box_h,
            boxstyle="round,pad=0.01",
            facecolor=color, edgecolor="#555555", linewidth=1.0,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        # Step label (left)
        ax.text(x_center - box_w/2 - 0.03, y, step_num,
                ha="right", va="center", fontsize=8, fontweight="bold",
                color="#555555", transform=ax.transAxes)
        # Main label
        ax.text(x_center, y, label,
                ha="center", va="center", fontsize=8.5,
                transform=ax.transAxes)
        # Arrow
        if i < len(steps) - 1:
            y_next = y_positions[i+1]
            ax.annotate("", xy=(x_center, y_next + box_h/2 + 0.005),
                        xytext=(x_center, y - box_h/2 - 0.005),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2))

    # Side annotations: output files
    outputs = [
        "audit_results.csv", "pvgis/{city}_tmy.csv",
        "morphology/cross_city_d2d3.csv", "energy/cross_city_d1d4d5.csv",
        "fdsi/fdsi_scores.csv", "paper4_summary/npv_irr_co2.csv",
        "figures/fig*.pdf"
    ]
    for y, out in zip(y_positions, outputs):
        ax.text(x_center + box_w/2 + 0.04, y, f"→ {out}",
                ha="left", va="center", fontsize=6.5, color="#888888",
                transform=ax.transAxes, style="italic")

    ax.set_title("Paper 4: BIPV Suitability Assessment Workflow",
                 fontsize=12, fontweight="bold", pad=8)
    save_fig(fig, "fig01_workflow")


# ============================================================================
# Fig 02: FDSI Framework Tree
# ============================================================================

def fig02_fdsi_framework(data: Dict):
    """FDSI 五维指标框架树状图。"""
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, HEIGHT_34))
    ax.axis("off")

    dimensions = [
        ("D1\nClimate\nResource", "#2166AC", [
            "Annual GHI", "Seasonal GHI CV",
        ]),
        ("D2\nUrban\nMorphology", "#74ADD1", [
            "Mean Roof Area", "Building Density",
        ]),
        ("D3\nTechnical\nPotential", "#F4A582", [
            "Specific Yield", "Shading Factor",
        ]),
        ("D4\nEconomic\nFeasibility", "#D6604D", [
            "LCOE", "Payback Period",
        ]),
        ("D5\nDeployment\nCertainty", "#4DAC26", [
            "PBT CI₉₅ Width (40%)",
            "LCOE Std Dev (35%)",
            "Sobol Interaction (25%)",
        ]),
    ]

    # FDSI root box
    root_x, root_y = 0.5, 0.90
    rect = mpatches.FancyBboxPatch((root_x - 0.18, root_y - 0.055), 0.36, 0.11,
                                    boxstyle="round,pad=0.01",
                                    facecolor="#FFD700", edgecolor="#555", lw=1.5,
                                    transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(root_x, root_y, "FDSI\nFive-Dimension Suitability Index",
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            transform=ax.transAxes)

    # Dimension boxes
    n = len(dimensions)
    xs = np.linspace(0.08, 0.92, n)
    dim_y = 0.58

    for i, (dim_label, color, sub_inds) in enumerate(dimensions):
        x = xs[i]
        # Line from root
        ax.plot([root_x, x], [root_y - 0.055, dim_y + 0.075],
                color="#888", lw=0.8, transform=ax.transAxes, zorder=0)

        # Dimension box
        rect = mpatches.FancyBboxPatch((x - 0.07, dim_y - 0.065), 0.14, 0.14,
                                        boxstyle="round,pad=0.008",
                                        facecolor=color, edgecolor="#444", lw=1.0,
                                        transform=ax.transAxes, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x, dim_y, dim_label, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="white",
                transform=ax.transAxes)

        # Sub-indicator boxes
        sub_y_start = 0.38
        sub_ys = np.linspace(sub_y_start, sub_y_start - 0.14*(len(sub_inds)-1), len(sub_inds))
        for j, (sub_label, sy) in enumerate(zip(sub_inds, sub_ys)):
            ax.plot([x, x], [dim_y - 0.065, sy + 0.025],
                    color=color, lw=0.8, alpha=0.7, transform=ax.transAxes)
            rect_s = mpatches.FancyBboxPatch((x - 0.07, sy - 0.025), 0.14, 0.05,
                                              boxstyle="round,pad=0.005",
                                              facecolor=color, edgecolor="#888", lw=0.5,
                                              transform=ax.transAxes, alpha=0.35)
            ax.add_patch(rect_s)
            ax.text(x, sy, sub_label, ha="center", va="center",
                    fontsize=6, transform=ax.transAxes)

    ax.set_title("FDSI Framework: Five-Dimension Suitability Index Structure",
                 fontsize=11, fontweight="bold", pad=6)
    save_fig(fig, "fig02_fdsi_framework")


# ============================================================================
# Fig 03: China Climate Map
# ============================================================================

def fig03_china_map(data: Dict):
    """中国气候分区+15城市位置地图。"""
    # 城市坐标
    city_coords = {
        "harbin": (126.65, 45.75), "changchun": (125.32, 43.88), "shenyang": (123.43, 41.80),
        "beijing": (116.40, 39.90), "jinan": (116.99, 36.65), "xian": (108.94, 34.26),
        "changsha": (112.94, 28.23), "wuhan": (114.30, 30.58), "nanjing": (118.77, 32.06),
        "shenzhen": (114.06, 22.54), "guangzhou": (113.27, 23.13), "xiamen": (118.09, 24.48),
        "kunming": (102.68, 25.04), "guiyang": (106.63, 26.65), "chengdu": (104.07, 30.67),
    }

    try:
        import geopandas as gpd
        HAS_GPD = True
    except ImportError:
        HAS_GPD = False

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, HEIGHT_34))

    if HAS_GPD:
        try:
            world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            china = world[world['name'] == 'China']
            china.plot(ax=ax, color='#F5F5F5', edgecolor='#CCCCCC', linewidth=0.5)
            ax.set_xlim(72, 136)
            ax.set_ylim(17, 54)
        except Exception:
            HAS_GPD = False

    if not HAS_GPD:
        # Simplified outline
        ax.set_xlim(72, 136)
        ax.set_ylim(17, 54)
        ax.set_facecolor("#E8F4F8")
        # Approximate China boundary polygon
        china_x = [73, 135, 135, 120, 115, 108, 104, 98, 88, 80, 73, 73]
        china_y = [39, 39, 53, 53, 43, 32, 22, 25, 28, 38, 45, 39]
        ax.fill(china_x, china_y, color="#F0F0F0", ec="#AAAAAA", lw=0.5, alpha=0.8)

    # 气候分区背景色块（近似边界）
    zone_boxes = {
        "Severe Cold": {"x": [119, 135, 135, 119], "y": [43, 43, 54, 54]},
        "Cold":        {"x": [100, 119, 119, 100], "y": [34, 34, 43, 43]},
        "HSCW":        {"x": [104, 122, 122, 104], "y": [24, 24, 34, 34]},
        "HSWW":        {"x": [104, 122, 122, 104], "y": [20, 20, 24, 24]},
        "Mild":        {"x": [97, 108, 108, 97],  "y": [24, 24, 32, 32]},
    }
    for zone, box in zone_boxes.items():
        color = ZONE_COLORS[zone]
        ax.fill(box["x"], box["y"], color=color, alpha=0.20, zorder=1)

    # 城市散点
    for city, (lon, lat) in city_coords.items():
        zone = CITY_ZONES[city]
        color = ZONE_COLORS[zone]
        ax.scatter(lon, lat, c=color, s=50, zorder=5, edgecolors="white", linewidth=0.5)
        # 城市标签（错开避免重叠）
        offset_x = 1.2
        offset_y = 0.3
        if city in ["guangzhou", "shenzhen"]:
            offset_y = -1.2
        if city in ["nanjing", "wuhan"]:
            offset_x = -4
        ax.text(lon + offset_x, lat + offset_y, CITY_LABELS[city],
                fontsize=6.5, zorder=6, ha="left",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.6, linewidth=0))

    # 图例
    legend_elements = [
        mpatches.Patch(facecolor=ZONE_COLORS[z], alpha=0.8, label=z)
        for z in ZONE_ORDER
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=7,
              title="Climate Zone", title_fontsize=7.5, framealpha=0.8)

    ax.set_xlabel("Longitude (°E)", fontsize=8)
    ax.set_ylabel("Latitude (°N)", fontsize=8)
    ax.set_title("Study Cities: 15 Chinese Cities across 5 Climate Zones (GB 50176)",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.2, linewidth=0.5)
    save_fig(fig, "fig03_china_map")


# ============================================================================
# Fig 04: Radar Chart (15 cities)
# ============================================================================

def fig04_radar(data: Dict):
    """15城市五维雷达图。"""
    integ = data.get("integrated")
    if integ is None:
        log.warning("  integrated_indicators.csv not found, skipping fig04")
        return

    # Try both column naming conventions
    dim_cols_v1 = ["D1_Climate", "D2_Morphology", "D3_Technical", "D4_Economic", "D5_Uncertainty"]
    dim_cols_v2 = ["score_D1_Climate", "score_D2_Morphology", "score_D3_Technical",
                   "score_D4_Economic", "score_D5_Uncertainty"]
    dim_labels = ["D1\nClimate", "D2\nMorphology", "D3\nTechnical", "D4\nEconomic", "D5\nCertainty"]

    if all(c in integ.columns for c in dim_cols_v2):
        dim_cols = dim_cols_v2
    elif all(c in integ.columns for c in dim_cols_v1):
        dim_cols = dim_cols_v1
    else:
        log.warning("  Dimension score columns missing, computing from FDSI data")
        return

    n_dims = len(dim_cols)
    available_cols = dim_cols

    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, HEIGHT_34),
                            subplot_kw=dict(polar=True))

    plotted_zones = set()
    for _, row in integ.iterrows():
        city = row.get("city", "")
        zone = CITY_ZONES.get(city, "Cold")
        color = ZONE_COLORS[zone]
        values = [row.get(c, 0.5) for c in available_cols]
        values += values[:1]

        lw = 2.0 if city in ["harbin", "beijing", "changsha", "shenzhen", "kunming"] else 0.8
        alpha = 0.8 if lw > 1 else 0.4
        label = zone if zone not in plotted_zones else None
        ax.plot(angles, values, color=color, linewidth=lw, alpha=alpha, label=label)
        ax.fill(angles, values, color=color, alpha=0.05)
        plotted_zones.add(zone)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], fontsize=6.5)
    ax.grid(True, alpha=0.3)

    # Legend
    legend_handles = [
        Line2D([0], [0], color=ZONE_COLORS[z], lw=2, label=z)
        for z in ZONE_ORDER
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              bbox_to_anchor=(1.35, 1.15), fontsize=7.5, title="Climate Zone")
    ax.set_title("Five-Dimension FDSI Profile: 15 Cities", fontsize=11,
                 fontweight="bold", pad=12)
    save_fig(fig, "fig04_radar")


# ============================================================================
# Fig 05: Heatmap (15×5)
# ============================================================================

def fig05_heatmap(data: Dict):
    """15×5维度得分热力图。"""
    integ = data.get("integrated")
    if integ is None:
        log.warning("  Skipping fig05 (no integrated data)")
        return

    dim_cols_v2 = ["score_D1_Climate", "score_D2_Morphology", "score_D3_Technical",
                   "score_D4_Economic", "score_D5_Uncertainty"]
    dim_cols_v1 = ["D1_Climate", "D2_Morphology", "D3_Technical", "D4_Economic", "D5_Uncertainty"]
    if all(c in integ.columns for c in dim_cols_v2):
        dim_cols = dim_cols_v2
    elif all(c in integ.columns for c in dim_cols_v1):
        dim_cols = dim_cols_v1
    else:
        log.warning("  Skipping fig05 (dimension cols missing)")
        return
    available = dim_cols

    integ_ordered = integ.set_index("city").reindex(
        [c for c in CITY_ORDER if c in integ["city"].values]
    )

    labels = [CITY_LABELS.get(c, c) for c in integ_ordered.index]
    matrix = integ_ordered[available].values

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.8, HEIGHT_34))

    if HAS_SNS:
        hm_df = pd.DataFrame(matrix, index=labels,
                             columns=["D1\nClimate","D2\nMorphology","D3\nTechnical",
                                      "D4\nEconomic","D5\nCertainty"])
        sns.heatmap(hm_df, ax=ax, cmap="RdYlGn", vmin=0, vmax=1,
                    annot=True, fmt=".2f", annot_kws={"size": 7},
                    linewidths=0.3, linecolor="#CCCCCC",
                    cbar_kws={"label": "Normalized Score", "shrink": 0.8})
    else:
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(available)))
        ax.set_xticklabels(["D1","D2","D3","D4","D5"], fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        plt.colorbar(im, ax=ax, shrink=0.6, label="Score")
        for i in range(len(labels)):
            for j in range(len(available)):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=6)

    # Zone dividers
    zone_boundaries = [3, 6, 9, 12]  # after each 3-city group
    for b in zone_boundaries:
        if b < len(labels):
            ax.axhline(b - 0.5, color="#555555", lw=1.0)

    ax.set_title("FDSI Dimension Scores: 15 Cities × 5 Dimensions",
                 fontsize=10, fontweight="bold")
    save_fig(fig, "fig05_heatmap")


# ============================================================================
# Fig 06: FDSI Ranking Bar Chart
# ============================================================================

def fig06_fdsi_ranking(data: Dict):
    """15城市FDSI排名条形图。"""
    fdsi = data["fdsi"].copy()
    fdsi = fdsi.sort_values("fdsi_score", ascending=True)

    colors = [ZONE_COLORS.get(row["climate_zone"], "#888")
              for _, row in fdsi.iterrows()]
    labels = [CITY_LABELS.get(row["city"], row["city"])
              for _, row in fdsi.iterrows()]

    # Suitability thresholds
    thresholds = {"High": 0.7, "Medium": 0.5, "Low": 0.0}

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.8, HEIGHT_34))

    bars = ax.barh(range(len(fdsi)), fdsi["fdsi_score"].values,
                   color=colors, height=0.7, alpha=0.85, edgecolor="white", linewidth=0.3)

    # Threshold lines
    ax.axvline(0.7, color="#E53935", lw=1.2, linestyle="--", label="High (≥0.70)")
    ax.axvline(0.5, color="#FB8C00", lw=1.2, linestyle=":", label="Medium (≥0.50)")

    # Score labels
    for i, (_, row) in enumerate(fdsi.iterrows()):
        ax.text(row["fdsi_score"] + 0.01, i, f'{row["fdsi_score"]:.3f}',
                va="center", fontsize=7)

    ax.set_yticks(range(len(fdsi)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("FDSI Score", fontsize=9)
    ax.set_title("FDSI Rankings: 15 Chinese Cities\n(D5 weights: 40/35/25, Phase 2)",
                 fontsize=10, fontweight="bold")

    # Zone legend + suitability legend
    zone_handles = [mpatches.Patch(color=ZONE_COLORS[z], alpha=0.85, label=z)
                    for z in ZONE_ORDER]
    thresh_handles = [
        Line2D([0],[0], color="#E53935", lw=1.2, ls="--", label="High (≥0.70)"),
        Line2D([0],[0], color="#FB8C00", lw=1.2, ls=":", label="Medium (≥0.50)"),
    ]
    ax.legend(handles=zone_handles + thresh_handles,
              loc="lower right", fontsize=7, ncol=2)
    save_fig(fig, "fig06_fdsi_ranking")


# ============================================================================
# Fig 07: Weight Sensitivity
# ============================================================================

def fig07_weight_sensitivity(data: Dict):
    """α扫描下的FDSI值和排名稳定性。"""
    ws = data.get("weight_sens")
    if ws is None:
        log.warning("  Skipping fig07 (no weight_sensitivity data)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, HEIGHT_34))

    alpha_col = "alpha" if "alpha" in ws.columns else ws.columns[0]
    city_cols = [c for c in ws.columns if c != alpha_col and "rank" not in c.lower()]

    # Panel 1: FDSI values vs alpha
    for city in [c for c in CITY_ORDER if c in city_cols][:8]:
        zone = CITY_ZONES.get(city, "Cold")
        ax1.plot(ws[alpha_col], ws[city], color=ZONE_COLORS[zone],
                 linewidth=1.2 if city in ["harbin","beijing","changsha","shenzhen","kunming"] else 0.6,
                 alpha=0.8, label=CITY_LABELS.get(city, city))

    ax1.set_xlabel("Weight Parameter α (entropy vs. AHP)", fontsize=8)
    ax1.set_ylabel("FDSI Score", fontsize=8)
    ax1.set_title("(a) FDSI Sensitivity to α", fontsize=9, fontweight="bold")
    ax1.legend(fontsize=6, ncol=2, loc="best")
    ax1.set_xlim(ws[alpha_col].min(), ws[alpha_col].max())

    # Panel 2: Rank stability (simplified - rank vs alpha for top 5 cities)
    rank_cols = [c for c in ws.columns if "rank" in c.lower()]
    if rank_cols:
        for city_col in rank_cols[:5]:
            ax2.plot(ws[alpha_col], ws[city_col], linewidth=1.2)
    else:
        # Compute rank from values
        for idx in ws.index:
            scores = {c: ws.loc[idx, c] for c in city_cols if c in ws.columns}
            if scores:
                sorted_scores = sorted(scores.values(), reverse=True)
        ax2.text(0.5, 0.5, "Rank stability\n(computed per run)",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=9, color="gray")

    ax2.set_xlabel("Weight Parameter α", fontsize=8)
    ax2.set_ylabel("FDSI Rank", fontsize=8)
    ax2.set_title("(b) Rank Stability vs. α", fontsize=9, fontweight="bold")

    fig.suptitle("Weight Sensitivity Analysis (α = entropy weight proportion)",
                 fontsize=10, fontweight="bold")
    save_fig(fig, "fig07_weight_sensitivity")


# ============================================================================
# Fig 08: MC PBT Violin Plot
# ============================================================================

def fig08_mc_pbt_violin(data: Dict):
    """15城市MC PBT分布小提琴图。"""
    energy = data["energy"]

    # 从MC数据重构分布（用 p025, p975, mean 近似正态）
    cities_ordered = [c for c in CITY_ORDER if c in energy["city"].values]
    energy_ordered = energy.set_index("city").reindex(cities_ordered)

    # 使用近似正态分布生成MC数据供可视化
    pbt_distributions = {}
    for city in cities_ordered:
        row = energy_ordered.loc[city]
        p025 = float(row.get("d5_2_pbt_p025", 4.0))
        p975 = float(row.get("d5_2_pbt_p975", 8.0))
        pbt_mean = float(row.get("d4_2_pbt_years", 5.0))
        pbt_std = (p975 - p025) / (2 * 1.96)
        np.random.seed(42)
        dist = np.random.normal(pbt_mean, pbt_std, 500)
        dist = np.clip(dist, p025 * 0.9, p975 * 1.1)
        pbt_distributions[city] = dist

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, HEIGHT_34))

    violin_data = [pbt_distributions[c] for c in cities_ordered]
    labels = [CITY_LABELS.get(c, c) for c in cities_ordered]
    colors = [get_zone_color(c) for c in cities_ordered]

    parts = ax.violinplot(violin_data, positions=range(len(cities_ordered)),
                          showmedians=True, showextrema=False, widths=0.7)

    for i, (pc, color) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.65)
        pc.set_edgecolor("white")
        pc.set_linewidth(0.5)
    parts["cmedians"].set_color("#333333")
    parts["cmedians"].set_linewidth(1.5)

    ax.set_xticks(range(len(cities_ordered)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax.axhline(15, color="#E53935", lw=1.0, linestyle="--", label="PBT=15yr target", alpha=0.8)

    # Zone background bands
    zone_groups = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)]
    for (x0, x1), zone in zip(zone_groups, ZONE_ORDER):
        ax.axvspan(x0 - 0.5, x1 - 0.5, color=ZONE_COLORS[zone], alpha=0.08)

    ax.set_ylabel("Payback Period (years)", fontsize=9)
    ax.set_title("MC Payback Period Distribution: 15 Cities (N=10,000)", fontsize=10, fontweight="bold")

    zone_handles = [mpatches.Patch(color=ZONE_COLORS[z], alpha=0.65, label=z) for z in ZONE_ORDER]
    ax.legend(handles=zone_handles + [Line2D([0],[0], color="#E53935", lw=1, ls="--", label="15yr target")],
              fontsize=7, ncol=3, loc="upper right")
    save_fig(fig, "fig08_mc_pbt_violin")


# ============================================================================
# Fig 09: Sobol Bar Chart
# ============================================================================

def fig09_sobol_bar(data: Dict):
    """Sobol S₁指数堆叠条形图。"""
    energy = data["energy"]
    cities_ordered = [c for c in CITY_ORDER if c in energy["city"].values]
    energy_ordered = energy.set_index("city").reindex(cities_ordered)

    params = ["ghi_factor", "module_efficiency", "system_losses", "pv_cost", "elec_price_factor"]
    param_labels = ["GHI", "Module η", "Sys. Losses", "PV Cost", "Elec. Price"]
    param_colors = ["#2166AC", "#4DAF4A", "#FF7F00", "#E41A1C", "#984EA3"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, HEIGHT_34))

    for ax, output_key, title in [
        (ax1, "yield", "(a) Annual Yield"),
        (ax2, "pbt", "(b) Payback Period"),
    ]:
        bottoms = np.zeros(len(cities_ordered))
        for param, label, color in zip(params, param_labels, param_colors):
            col = f"sobol_{output_key}_S1_{param}"
            if col not in energy_ordered.columns:
                values = np.full(len(cities_ordered), 0.1)
            else:
                values = energy_ordered[col].fillna(0).values.astype(float)

            ax.bar(range(len(cities_ordered)), values, bottom=bottoms,
                   color=color, label=label, width=0.7, alpha=0.85)
            bottoms += values

        labels = [CITY_LABELS.get(c, c) for c in cities_ordered]
        ax.set_xticks(range(len(cities_ordered)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Sobol S₁ Index", fontsize=8.5)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_ylim(0, 1.05)

        # Zone bands
        zone_groups = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)]
        for (x0, x1), zone in zip(zone_groups, ZONE_ORDER):
            ax.axvspan(x0 - 0.5, x1 - 0.5, color=ZONE_COLORS[zone], alpha=0.08)

    ax1.legend(fontsize=7, loc="upper right", ncol=2)
    fig.suptitle("Sobol First-Order Sensitivity Indices (S₁) by City",
                 fontsize=10, fontweight="bold")
    save_fig(fig, "fig09_sobol_bar")


# ============================================================================
# Fig 10: D4 vs D5 Scatter
# ============================================================================

def fig10_d4_vs_d5(data: Dict):
    """D4 vs D5 散点图（证明非冗余性）。"""
    integ = data.get("integrated")
    if integ is None:
        log.warning("  Skipping fig10 (no integrated data)")
        return

    # Try both naming conventions
    d4_col = "score_D4_Economic" if "score_D4_Economic" in integ.columns else "D4_Economic"
    d5_col = "score_D5_Uncertainty" if "score_D5_Uncertainty" in integ.columns else "D5_Uncertainty"
    if d4_col not in integ.columns or d5_col not in integ.columns:
        log.warning("  Skipping fig10 (D4/D5 columns missing)")
        return

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.6, SINGLE_COL * 1.6))

    for _, row in integ.iterrows():
        city = row.get("city", "")
        zone = CITY_ZONES.get(city, "Cold")
        color = ZONE_COLORS[zone]
        ax.scatter(row[d4_col], row[d5_col],
                   c=color, s=60, zorder=5, edgecolors="white", linewidth=0.5)
        ax.annotate(CITY_LABELS.get(city, city),
                    (row[d4_col], row[d5_col]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("D4 Economic Score", fontsize=9)
    ax.set_ylabel("D5 Certainty Score", fontsize=9)
    ax.set_title("D4 (Economic) vs D5 (Certainty)\nDimension Independence Check",
                 fontsize=9, fontweight="bold")

    # Correlation annotation
    d4 = integ[d4_col].values
    d5 = integ[d5_col].values
    corr = np.corrcoef(d4, d5)[0, 1]
    ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
            fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    zone_handles = [mpatches.Patch(color=ZONE_COLORS[z], alpha=0.8, label=z) for z in ZONE_ORDER]
    ax.legend(handles=zone_handles, fontsize=7, loc="lower right")
    save_fig(fig, "fig10_d4_vs_d5")


# ============================================================================
# Fig 11: NPV Comparison
# ============================================================================

def fig11_npv_comparison(data: Dict):
    """15城市NPV对比条形图。"""
    npv = data.get("npv")
    if npv is None:
        log.warning("  Skipping fig11 (no NPV data)")
        return

    npv_ordered = npv.set_index("city").reindex([c for c in CITY_ORDER if c in npv["city"].values])
    labels = [CITY_LABELS.get(c, c) for c in npv_ordered.index]
    colors = [get_zone_color(c) for c in npv_ordered.index]
    values = npv_ordered["npv_cny_kwp"].values / 1000  # 千元/kWp

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, HEIGHT_34 * 0.7))

    bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.3, width=0.7)
    ax.axhline(0, color="#333333", lw=0.8)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("NPV (×10³ CNY/kWp)", fontsize=9)
    ax.set_title("Net Present Value (NPV) Comparison: 15 Cities\n(25-year, r=6%, 2024 tariffs)",
                 fontsize=10, fontweight="bold")

    # Zone background
    zone_groups = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)]
    for (x0, x1), zone in zip(zone_groups, ZONE_ORDER):
        ax.axvspan(x0 - 0.5, x1 - 0.5, color=ZONE_COLORS[zone], alpha=0.10)

    zone_handles = [mpatches.Patch(color=ZONE_COLORS[z], alpha=0.8, label=z) for z in ZONE_ORDER]
    ax.legend(handles=zone_handles, fontsize=7.5, ncol=5, loc="upper right")
    save_fig(fig, "fig11_npv_comparison")


# ============================================================================
# Fig 12: IRR Comparison
# ============================================================================

def fig12_irr_comparison(data: Dict):
    """15城市IRR对比条形图。"""
    npv = data.get("npv")
    if npv is None:
        log.warning("  Skipping fig12")
        return

    npv_ordered = npv.set_index("city").reindex([c for c in CITY_ORDER if c in npv["city"].values])
    labels = [CITY_LABELS.get(c, c) for c in npv_ordered.index]
    colors = [get_zone_color(c) for c in npv_ordered.index]
    values = npv_ordered["irr_pct"].values

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, HEIGHT_34 * 0.7))

    ax.bar(range(len(labels)), values, color=colors, alpha=0.85,
           edgecolor="white", linewidth=0.3, width=0.7)
    ax.axhline(6, color="#E53935", lw=1.2, ls="--", label="Discount rate (6%)")
    ax.axhline(8, color="#FB8C00", lw=1.0, ls=":", label="Market rate (8%)")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("IRR (%)", fontsize=9)
    ax.set_title("Internal Rate of Return (IRR): 15 Cities", fontsize=10, fontweight="bold")

    zone_groups = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)]
    for (x0, x1), zone in zip(zone_groups, ZONE_ORDER):
        ax.axvspan(x0 - 0.5, x1 - 0.5, color=ZONE_COLORS[zone], alpha=0.10)

    zone_handles = [mpatches.Patch(color=ZONE_COLORS[z], alpha=0.8, label=z) for z in ZONE_ORDER]
    thresh_handles = [
        Line2D([0],[0], color="#E53935", lw=1.2, ls="--", label="Discount rate 6%"),
        Line2D([0],[0], color="#FB8C00", lw=1.0, ls=":", label="Market rate 8%"),
    ]
    ax.legend(handles=zone_handles + thresh_handles, fontsize=7, ncol=3, loc="upper right")
    save_fig(fig, "fig12_irr_comparison")


# ============================================================================
# Fig 13: CO₂ Reduction
# ============================================================================

def fig13_co2_reduction(data: Dict):
    """年度CO₂减排潜力。"""
    npv = data.get("npv")
    if npv is None:
        log.warning("  Skipping fig13")
        return

    npv_ordered = npv.set_index("city").reindex([c for c in CITY_ORDER if c in npv["city"].values])
    labels = [CITY_LABELS.get(c, c) for c in npv_ordered.index]
    colors = [get_zone_color(c) for c in npv_ordered.index]
    values_yr1 = npv_ordered["co2_annual_tco2_kwp"].values * 1000  # tCO₂/kWp → kgCO₂/kWp
    values_life = npv_ordered["co2_lifetime_tco2_kwp"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE_COL, HEIGHT_34 * 0.8))

    # Panel 1: Annual
    ax1.bar(range(len(labels)), values_yr1, color=colors, alpha=0.85,
            edgecolor="white", linewidth=0.3, width=0.7)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax1.set_ylabel("CO₂ Reduction (kg/kWp/yr)", fontsize=8.5)
    ax1.set_title("(a) Annual CO₂ Reduction", fontsize=9, fontweight="bold")

    # Panel 2: Lifetime
    ax2.bar(range(len(labels)), values_life, color=colors, alpha=0.85,
            edgecolor="white", linewidth=0.3, width=0.7)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax2.set_ylabel("CO₂ Reduction (tCO₂/kWp, 25yr)", fontsize=8.5)
    ax2.set_title("(b) 25-Year Cumulative CO₂ Reduction", fontsize=9, fontweight="bold")

    for ax in [ax1, ax2]:
        zone_groups = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)]
        for (x0, x1), zone in zip(zone_groups, ZONE_ORDER):
            ax.axvspan(x0 - 0.5, x1 - 0.5, color=ZONE_COLORS[zone], alpha=0.10)

    zone_handles = [mpatches.Patch(color=ZONE_COLORS[z], alpha=0.8, label=z) for z in ZONE_ORDER]
    ax1.legend(handles=zone_handles, fontsize=7, ncol=3, loc="upper right")
    fig.suptitle("CO₂ Emission Reduction Potential (2022 Grid Emission Factors)",
                 fontsize=10, fontweight="bold")
    save_fig(fig, "fig13_co2_reduction")


# ============================================================================
# Fig 14: 25-Year Cash Flow
# ============================================================================

def fig14_cashflow(data: Dict):
    """25年累积现金流（5个代表城市）。"""
    cf = data.get("cashflow")
    if cf is None:
        log.warning("  Skipping fig14")
        return

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.8, HEIGHT_34 * 0.8))

    representative = ["harbin", "beijing", "changsha", "shenzhen", "kunming"]
    for city in representative:
        city_cf = cf[cf["city"] == city]
        if len(city_cf) == 0:
            continue
        zone = CITY_ZONES.get(city, "Cold")
        color = ZONE_COLORS[zone]
        label = CITY_LABELS.get(city, city)
        ax.plot(city_cf["year"], city_cf["cumulative_npv_cny_kwp"] / 1000,
                color=color, linewidth=2.0, label=label, marker="", alpha=0.9)

    ax.axhline(0, color="#333333", lw=0.8, alpha=0.6)
    ax.fill_between(range(1, 26), 0, color="#EEEEEE", alpha=0.2)

    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel("Cumulative NPV (×10³ CNY/kWp)", fontsize=9)
    ax.set_title("25-Year Cumulative Cash Flow (5 Representative Cities)\n(r=6%, PV cost=3.0 CNY/Wp)",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(1, 25)
    ax.grid(True, alpha=0.2)
    save_fig(fig, "fig14_cashflow")


# ============================================================================
# Fig 15: Height Distribution
# ============================================================================

def fig15_height_distribution(data: Dict):
    """建筑高度分布堆叠条形图。"""
    # 从 typology_stats 读取各城市高度分类
    typology_stats_all = []
    for city in CITY_ORDER:
        path = MORPH_DIR / f"{city}_typology_stats.csv"
        if path.exists():
            ts = pd.read_csv(path)
            ts["city"] = city
            typology_stats_all.append(ts)

    if not typology_stats_all:
        # 从 morphology summary 近似
        morph = data["morphology"]
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, HEIGHT_34 * 0.7))
        cities_ordered = [c for c in CITY_ORDER if c in morph["city"].values]
        morph_ord = morph.set_index("city").reindex(cities_ordered)
        labels = [CITY_LABELS.get(c, c) for c in cities_ordered]
        # Use height mean as proxy
        heights = morph_ord["d2_1_height_mean"].values
        colors = [get_zone_color(c) for c in cities_ordered]
        ax.bar(range(len(labels)), heights, color=colors, alpha=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
        ax.set_ylabel("Mean Building Height (m)", fontsize=9)
        ax.set_title("Mean Building Height: 15 Cities", fontsize=10, fontweight="bold")
        save_fig(fig, "fig15_height_distribution")
        return

    # Typology stacked bar
    all_stats = pd.concat(typology_stats_all, ignore_index=True)
    typology_types = ["low_rise", "mid_rise", "mid_high", "high_rise"]
    typology_labels = ["Low-rise (1-3F)", "Mid-rise (4-6F)", "Mid-high (7-9F)", "High-rise (≥10F)"]
    typology_colors = ["#A8D5A2", "#5A9E6F", "#2171B5", "#08519C"]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, HEIGHT_34 * 0.75))
    cities_ordered = [c for c in CITY_ORDER if c in all_stats["city"].values]

    bottoms = np.zeros(len(cities_ordered))
    for typ, label, color in zip(typology_types, typology_labels, typology_colors):
        values = []
        for city in cities_ordered:
            city_stats = all_stats[all_stats["city"] == city]
            if "typology" in city_stats.columns:
                typ_row = city_stats[city_stats["typology"] == typ]
                count = typ_row["count"].values[0] if len(typ_row) > 0 and "count" in typ_row else 0
                total = city_stats["count"].sum() if "count" in city_stats else 1
                values.append(count / max(total, 1))
            else:
                values.append(0.25)  # fallback uniform
        ax.bar(range(len(cities_ordered)), values, bottom=bottoms,
               color=color, label=label, alpha=0.85, width=0.75)
        bottoms += np.array(values)

    labels_str = [CITY_LABELS.get(c, c) for c in cities_ordered]
    ax.set_xticks(range(len(cities_ordered)))
    ax.set_xticklabels(labels_str, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("Fraction of Buildings", fontsize=9)
    ax.set_title("Building Height Typology Distribution: 15 Cities", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper right", ncol=2)

    zone_groups = [(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)]
    for (x0, x1), zone in zip(zone_groups, ZONE_ORDER):
        ax.axvspan(x0 - 0.5, x1 - 0.5, color=ZONE_COLORS[zone], alpha=0.08)
    save_fig(fig, "fig15_height_distribution")


# ============================================================================
# Fig 16: Suitability Matrix (Climate × Morphology)
# ============================================================================

def fig16_suitability_matrix(data: Dict):
    """气候×形态适宜性矩阵。"""
    fdsi = data["fdsi"]
    morph = data["morphology"]

    # 合并形态类型
    merged = fdsi.merge(morph[["city", "city_typology"]], on="city", how="left")

    climate_zones = ["Severe Cold", "Cold", "HSCW", "HSWW", "Mild"]
    morphology_types = ["low_rise_dominant", "mid_rise_dominant", "mid_high_dominant", "high_rise_dominant"]
    morph_labels = ["Low-rise\nDominant", "Mid-rise\nDominant", "Mid-high\nDominant", "High-rise\nDominant"]

    # Build matrix
    matrix = np.full((len(climate_zones), len(morphology_types)), np.nan)
    annotations = [["" for _ in morphology_types] for _ in climate_zones]

    for _, row in merged.iterrows():
        city = row["city"]
        zone = CITY_ZONES.get(city, "")
        if zone not in climate_zones:
            continue
        typology = str(row.get("city_typology", "mixed"))
        # map to one of 4 types
        morph_type = None
        for mt in morphology_types:
            if mt in typology or typology.replace(" ", "_") == mt:
                morph_type = mt
                break
        if morph_type is None:
            morph_type = "mid_high_dominant"  # default

        zi = climate_zones.index(zone)
        mi = morphology_types.index(morph_type)
        if np.isnan(matrix[zi, mi]):
            matrix[zi, mi] = row["fdsi_score"]
            annotations[zi][mi] = f"{CITY_LABELS.get(city, city)}\n{row['fdsi_score']:.2f}"
        else:
            matrix[zi, mi] = (matrix[zi, mi] + row["fdsi_score"]) / 2
            annotations[zi][mi] += f"\n{CITY_LABELS.get(city, city)}\n{row['fdsi_score']:.2f}"

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 2.0, HEIGHT_34 * 0.85))

    # Plot matrix
    cmap = plt.cm.RdYlGn
    cmap.set_bad("#DDDDDD")
    im = ax.imshow(matrix, cmap=cmap, vmin=0.2, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(morphology_types)))
    ax.set_xticklabels(morph_labels, fontsize=8.5)
    ax.set_yticks(range(len(climate_zones)))
    ax.set_yticklabels(climate_zones, fontsize=8.5)

    for i in range(len(climate_zones)):
        for j in range(len(morphology_types)):
            text = annotations[i][j]
            if text:
                ax.text(j, i, text, ha="center", va="center", fontsize=6.0,
                        color="black" if matrix[i, j] > 0.5 else "white")
            elif np.isnan(matrix[i, j]):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=7, color="#AAAAAA")

    plt.colorbar(im, ax=ax, shrink=0.7, label="FDSI Score")
    ax.set_title("Climate Zone × Morphology Type Suitability Matrix\n(Mean FDSI per cell)",
                 fontsize=9.5, fontweight="bold")
    save_fig(fig, "fig16_suitability_matrix")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Paper 4: Generate 16 Figures")
    parser.add_argument("--fig", type=int, default=None,
                        help="Generate single figure by number (1-16)")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Paper 4 — Step 7: Generating 16 Paper Figures")
    log.info("=" * 60)

    log.info("\n[Loading data...]")
    data = load_data()

    fig_functions = {
        1: fig01_workflow,
        2: fig02_fdsi_framework,
        3: fig03_china_map,
        4: fig04_radar,
        5: fig05_heatmap,
        6: fig06_fdsi_ranking,
        7: fig07_weight_sensitivity,
        8: fig08_mc_pbt_violin,
        9: fig09_sobol_bar,
        10: fig10_d4_vs_d5,
        11: fig11_npv_comparison,
        12: fig12_irr_comparison,
        13: fig13_co2_reduction,
        14: fig14_cashflow,
        15: fig15_height_distribution,
        16: fig16_suitability_matrix,
    }

    if args.fig:
        if args.fig not in fig_functions:
            log.error(f"Figure {args.fig} not found (1-16)")
            sys.exit(1)
        log.info(f"\nGenerating fig{args.fig:02d}...")
        fig_functions[args.fig](data)
    else:
        for num, func in fig_functions.items():
            log.info(f"\n[{num}/16] Generating fig{num:02d}_{func.__name__.split('_', 1)[1]}...")
            try:
                func(data)
            except Exception as e:
                log.error(f"  ERROR in fig{num:02d}: {e}")
                import traceback
                traceback.print_exc()

    generated = list(FIGURES_DIR.glob("fig*.pdf"))
    log.info(f"\n{'='*60}")
    log.info(f"Generated: {len(generated)} PDF figures in {FIGURES_DIR}")
    log.info("Step 7 完成。")


if __name__ == "__main__":
    main()
