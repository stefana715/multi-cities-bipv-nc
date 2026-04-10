"""
nc_figs_main.py
Generate Figs 1–4 for the NC paper.

Fig 1 — Misclassification evidence (§3.1)
  (a) 3×3 GHI-tercile vs FDSI-tercile confusion matrix
  (b) Rank-shift horizontal bar chart, 41 cities, climate-zone colour

Fig 2 — Mechanism pairs (§3.3)
  (a) Changsha vs Chengdu — grouped bar D1–D5 + FDSI
  (b) Shenzhen vs Hong Kong — grouped bar D1–D5 + FDSI

Fig 3 — Policy consequences (§3.4)
  (a) Opportunity-cost of 4 missed cities (capacity / generation / CO₂ / population)
  (b) 4-scenario × 41-city suitability distribution (stacked horizontal bar)

Fig 4 (optional) — China map with FDSI dots
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
FIG_DIR  = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

SM_PATH  = ROOT / "results" / "fdsi" / "suitability_matrix.csv"
II_PATH  = ROOT / "results" / "fdsi" / "integrated_indicators.csv"
CM_PATH  = ROOT / "results_nc" / "misclassification" / "misclass_confusion_matrix.csv"
RS_PATH  = ROOT / "results_nc" / "misclassification" / "rank_shift_analysis.csv"
OC_PATH  = ROOT / "results_nc" / "policy_cost" / "opportunity_cost_physical.csv"
MJ_PATH  = ROOT / "results_nc" / "misclassification" / "misclassification_summary.json"
SC_PATH  = ROOT / "results" / "scenarios" / "scenario_fdsi_matrix.csv"

# ── global NC-style rcParams ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":          7,
    "axes.linewidth":     0.6,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.major.size":   2.5,
    "ytick.major.size":   2.5,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

# ── colour palettes ───────────────────────────────────────────────────────────
ZONE_COLORS = {
    "severe_cold": "#4575B4",   # deep blue
    "cold":        "#74ADD1",   # light blue
    "hscw":        "#F46D43",   # orange
    "hsww":        "#D73027",   # red
    "mild":        "#66BD63",   # green
}
ZONE_LABELS = {
    "severe_cold": "Severe Cold",
    "cold":        "Cold",
    "hscw":        "HSCW",
    "hsww":        "HSWW",
    "mild":        "Mild",
}
SUIT_COLORS = {
    "High":   "#2CA02C",
    "Medium": "#FF7F0E",
    "Low":    "#D62728",
}


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1
# ══════════════════════════════════════════════════════════════════════════════
def make_fig1():
    cm_raw = pd.read_csv(CM_PATH, index_col=0)
    rs     = pd.read_csv(RS_PATH)

    # sort by FDSI rank (ascending = best at bottom for horizontal bar)
    rs = rs.sort_values("fdsi_rank", ascending=False).reset_index(drop=True)

    fig = plt.figure(figsize=(7.0, 4.2))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.35,
                   left=0.06, right=0.97, top=0.90, bottom=0.12)

    # ── Panel (a): confusion matrix ──────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])

    mat  = cm_raw.values.astype(float)
    n    = mat.sum()

    # colour: diagonal=light blue, off-diagonal by magnitude (red scale)
    cmap_off  = plt.cm.Reds
    cmap_diag = plt.cm.Blues

    for i in range(3):
        for j in range(3):
            val = mat[i, j]
            if i == j:
                color = cmap_diag(0.25 + 0.25 * val / mat.diagonal().max())
            else:
                color = cmap_off(0.15 + 0.60 * val / (mat.max() + 1e-9))
            ax_a.add_patch(plt.Rectangle((j, 2 - i), 1, 1, color=color,
                                         lw=0.4, ec="white"))
            pct = val / n * 100
            label = f"{int(val)}\n({pct:.0f}%)" if val > 0 else "0"
            ax_a.text(j + 0.5, 2 - i + 0.5, label,
                      ha="center", va="center", fontsize=6.5,
                      fontweight="bold" if i != j and val > 0 else "normal",
                      color="white" if (i != j and val > 2) else "#333333")

    tick_labs = ["Low", "Medium", "High"]
    ax_a.set_xlim(0, 3); ax_a.set_ylim(0, 3)
    ax_a.set_xticks([0.5, 1.5, 2.5]); ax_a.set_xticklabels(tick_labs, fontsize=6)
    ax_a.set_yticks([0.5, 1.5, 2.5]); ax_a.set_yticklabels(tick_labs, fontsize=6)
    ax_a.set_xlabel("GHI suitability class", fontsize=6.5, labelpad=4)
    ax_a.set_ylabel("FDSI suitability class", fontsize=6.5, labelpad=4)
    ax_a.set_title("(a) Classification consistency", fontsize=7, fontweight="bold",
                   pad=6, loc="left")

    # diagonal label
    ax_a.text(1.5, -0.55, "Correctly classified = 28/41 (68.3%)\nMisclassified = 13/41 (31.7%)",
              ha="center", va="top", fontsize=5.5, color="#555555",
              transform=ax_a.transData)
    ax_a.spines[:].set_visible(False)
    ax_a.tick_params(length=0)

    # ── Panel (b): rank shift bar chart ──────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])

    y   = np.arange(len(rs))
    clr = [ZONE_COLORS.get(z, "#999999") for z in rs["climate_zone"]]

    bars = ax_b.barh(y, rs["rank_shift"], color=clr, height=0.7,
                     linewidth=0.2, edgecolor="white")

    # zero line
    ax_b.axvline(0, color="#444444", linewidth=0.6, zorder=10)

    # annotate extreme cases (|shift| >= 14)
    x_lim_pos = rs["rank_shift"].max() + 2
    x_lim_neg = rs["rank_shift"].min() - 2
    for idx_r, row in rs.iterrows():
        yi = y[idx_r]
        if abs(row.rank_shift) >= 14:
            if row.rank_shift > 0:
                xoff = min(row.rank_shift + 0.8, x_lim_pos)
                ha   = "left"
            else:
                xoff = max(row.rank_shift - 0.8, x_lim_neg)
                ha   = "right"
            ax_b.text(xoff, yi, f"{row.name_en} ({row.rank_shift:+d})",
                      ha=ha, va="center", fontsize=4.8, color="#333333",
                      clip_on=False)

    ax_b.set_yticks(y)
    ax_b.set_yticklabels(rs["name_en"], fontsize=4.5)
    ax_b.set_xlabel("Rank shift  (GHI rank − FDSI rank)", fontsize=6.5, labelpad=4)
    ax_b.set_title("(b) Rank shift: GHI rank vs. FDSI rank", fontsize=7,
                   fontweight="bold", pad=6, loc="left")
    ax_b.xaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax_b.set_xlim(rs["rank_shift"].min() - 9, rs["rank_shift"].max() + 9)

    # annotation: direction hints
    xlim = ax_b.get_xlim()
    ax_b.text(xlim[0] + 0.5, -2.8, "← GHI overestimates\n   (resource-rich trap)",
              ha="left", va="top", fontsize=5.0, color=ZONE_COLORS["cold"],
              style="italic")
    ax_b.text(xlim[1] - 0.5, -2.8, "GHI underestimates →\n(hidden champion)",
              ha="right", va="top", fontsize=5.0, color=ZONE_COLORS["hscw"],
              style="italic")

    # legend — climate zones
    handles = [mpatches.Patch(color=ZONE_COLORS[z], label=ZONE_LABELS[z])
               for z in ZONE_COLORS]
    ax_b.legend(handles=handles, fontsize=5.0, frameon=False,
                loc="lower right", ncol=1, handlelength=0.9,
                bbox_to_anchor=(1.0, 0.0))

    ax_b.yaxis.grid(False)
    ax_b.xaxis.grid(True, linewidth=0.3, color="#E8E8E8", zorder=0)
    ax_b.set_axisbelow(True)

    for fmt in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig1_misclassification.{fmt}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("Fig 1 saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2
# ══════════════════════════════════════════════════════════════════════════════
def _grouped_bar_pair(ax, city_a: dict, city_b: dict,
                      color_a: str, color_b: str,
                      anno_lines: list[str], panel_label: str):
    """Draw a grouped bar chart for two cities on ax."""
    DIMS   = ["D1", "D2", "D3", "D4", "D5", "FDSI"]
    XLABS  = ["D1\n(Climate)", "D2\n(Morphology)", "D3\n(Technical)",
              "D4\n(Economic)", "D5\n(Uncertainty)", "FDSI\n(Composite)"]

    vals_a = [city_a[d] for d in DIMS]
    vals_b = [city_b[d] for d in DIMS]

    x     = np.arange(len(DIMS))
    w     = 0.32
    gap   = 0.04

    ax.bar(x[:-1] - (w/2 + gap/2), vals_a[:-1], w,
           color=color_a, alpha=1.0, linewidth=0.3, edgecolor="white")
    ax.bar(x[:-1] + (w/2 + gap/2), vals_b[:-1], w,
           color=color_b, alpha=1.0, linewidth=0.3, edgecolor="white")

    # FDSI composite (slightly wider)
    ax.bar(x[-1] - (w/2 + gap/2), vals_a[-1], w*1.1,
           color=color_a, alpha=0.85, linewidth=0.4, edgecolor="white")
    ax.bar(x[-1] + (w/2 + gap/2), vals_b[-1], w*1.1,
           color=color_b, alpha=0.85, linewidth=0.4, edgecolor="white")

    # FDSI value labels
    for xi, val, col in [(x[-1] - (w/2+gap/2), vals_a[-1], "white"),
                         (x[-1] + (w/2+gap/2), vals_b[-1], "white")]:
        ax.text(xi, val - 0.025, f"{val:.3f}", ha="center", va="top",
                fontsize=5.2, color=col, fontweight="bold")

    # separator before FDSI
    ax.axvline(x[-1] - 0.62, color="#AAAAAA", linewidth=0.4, linestyle="--")

    # annotation box
    ax.text(0.02, 0.97, "\n".join(anno_lines),
            transform=ax.transAxes, fontsize=5.5, va="top", ha="left",
            linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5",
                      edgecolor="#BBBBBB", linewidth=0.4, alpha=0.92))

    ax.set_xticks(x)
    ax.set_xticklabels(XLABS, fontsize=5.5)
    ax.set_xlim(-0.65, len(DIMS) - 0.35)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=5.5)
    ax.set_ylabel("Normalised score", fontsize=6.0)
    ax.yaxis.grid(True, linewidth=0.3, color="#E8E8E8", zorder=0)
    ax.set_axisbelow(True)

    # legend
    patch_a = mpatches.Patch(color=color_a,
                              label=f"{city_a['name']} (rank #{city_a['rank']})")
    patch_b = mpatches.Patch(color=color_b,
                              label=f"{city_b['name']} (rank #{city_b['rank']})")
    ax.legend(handles=[patch_a, patch_b], fontsize=5.2, frameon=False,
              loc="upper right", bbox_to_anchor=(1.0, 1.02),
              handlelength=0.9, handletextpad=0.4)

    ax.set_title(panel_label, fontsize=7, fontweight="bold", pad=6, loc="left")


def make_fig2():
    sm = pd.read_csv(SM_PATH)
    ii = pd.read_csv(II_PATH)

    def city_data(name_sm, name_ii):
        s = sm[sm["city"] == name_sm].iloc[0]
        i = ii[ii["city"] == name_ii].iloc[0]
        # dominant_factor_pbt is a string label, not a float — skip
        pbt = None
        return {
            "name": name_sm,
            "rank": int(s["fdsi_rank"]),
            "GHI":  float(i["d1_1_ghi_annual_kwh"]),
            "FAR":  float(i.get("d2_5_far", float("nan"))),
            "FDSI": float(s["fdsi_score"]),
            "D1": float(s["D1_score"]), "D2": float(s["D2_score"]),
            "D3": float(s["D3_score"]), "D4": float(s["D4_score"]),
            "D5": float(s["D5_score"]),
        }

    csa  = city_data("Changsha",  "changsha")
    cdu  = city_data("Chengdu",   "chengdu")
    szn  = city_data("Shenzhen",  "shenzhen")
    hk   = city_data("Hong Kong", "hongkong")

    # PBT from scenario baseline
    sc   = pd.read_csv(SC_PATH)
    base = sc[sc["scenario"] == "baseline"]
    def pbt(city_ii):
        row = base[base["city"] == city_ii]
        return float(row["d4_pbt"].iloc[0]) if len(row) else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9),
                             gridspec_kw={"wspace": 0.38})
    fig.subplots_adjust(left=0.07, right=0.97, top=0.82, bottom=0.18)

    # ── panel (a): Changsha vs Chengdu ───────────────────────────────────────
    delta_ghi  = csa["GHI"] - cdu["GHI"]
    delta_rank = cdu["rank"] - csa["rank"]
    anno_a = [
        rf"$\Delta$GHI = {delta_ghi:+.0f} kWh m$^{{-2}}$ yr$^{{-1}}$",
        rf"$\Delta$rank = {delta_rank:+d}  (#{csa['rank']} vs #{cdu['rank']})",
    ]
    _grouped_bar_pair(axes[0], csa, cdu,
                      color_a="#2166AC", color_b="#D6604D",
                      anno_lines=anno_a,
                      panel_label="(a) Changsha vs. Chengdu")
    axes[0].text(0.0, 1.13,
                 "Similar irradiance, strongly different composite suitability",
                 transform=axes[0].transAxes, fontsize=5.5, color="#666666",
                 va="bottom", ha="left")

    # ── panel (b): Shenzhen vs Hong Kong ────────────────────────────────────
    delta_ghi2  = hk["GHI"]  - szn["GHI"]
    delta_rank2 = hk["rank"] - szn["rank"]
    pbt_szn = pbt("shenzhen")
    pbt_hk  = pbt("hongkong")
    anno_b  = [
        rf"$\Delta$GHI = {delta_ghi2:+.0f} kWh m$^{{-2}}$ yr$^{{-1}}$",
        rf"$\Delta$rank = {delta_rank2:+d}  (#{szn['rank']} vs #{hk['rank']})",
        f"PBT: {pbt_szn:.2f} yr vs {pbt_hk:.2f} yr",
        f"FAR: {szn['FAR']:.3f} vs {hk['FAR']:.3f}",
    ]
    _grouped_bar_pair(axes[1], szn, hk,
                      color_a="#1A9641", color_b="#762A83",
                      anno_lines=anno_b,
                      panel_label="(b) Shenzhen vs. Hong Kong")
    axes[1].text(0.0, 1.13,
                 "Institutional contrast: best economics, worst morphology",
                 transform=axes[1].transAxes, fontsize=5.5, color="#666666",
                 va="bottom", ha="left")

    for fmt in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig2_mechanism_pairs.{fmt}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("Fig 2 saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3
# ══════════════════════════════════════════════════════════════════════════════
def make_fig3():
    oc   = pd.read_csv(OC_PATH)
    sc   = pd.read_csv(SC_PATH)
    with open(MJ_PATH) as f:
        mj = json.load(f)

    missed_names = [c.strip() for c in mj["missed_cities"].split(",")]  # English names

    # ── Panel (a): opportunity cost ──────────────────────────────────────────
    # missed city rows
    oc_missed = oc[oc["name_en"].isin(missed_names)].copy()
    total_cap  = oc["capacity_mw"].sum()
    total_gen  = oc["generation_gwh_yr"].sum()
    miss_cap   = oc_missed["capacity_mw"].sum()
    miss_gen   = oc_missed["generation_gwh_yr"].sum()
    miss_co2   = oc_missed["co2_reduction_kt_yr"].sum()
    miss_pop   = oc_missed["population_million"].sum()

    fig = plt.figure(figsize=(7.0, 3.8))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.38,
                   left=0.06, right=0.97, top=0.88, bottom=0.14)

    ax_a = fig.add_subplot(gs[0, 0])

    # Stacked horizontal bars: missed vs rest, for capacity and generation
    categories = ["Deployable\ncapacity (MW)", "Generation\n(GWh/yr)"]
    totals  = [total_cap, total_gen]
    missed  = [miss_cap, miss_gen]

    y = np.arange(len(categories))
    rest = [t - m for t, m in zip(totals, missed)]

    ax_a.barh(y, rest,   0.45, color="#B0C4DE", label="Selected (GHI top-1/3)")
    ax_a.barh(y, missed, 0.45, left=rest, color="#D62728", label="Missed (FDSI high)")

    # value labels on missed segment
    for i, (r, m) in enumerate(zip(rest, missed)):
        ax_a.text(r + m/2, i, f"{m:,.0f}\n({m/(r+m)*100:.1f}%)",
                  ha="center", va="center", fontsize=6.0,
                  color="white", fontweight="bold")

    ax_a.set_yticks(y)
    ax_a.set_yticklabels(categories, fontsize=6.5)
    ax_a.set_xlabel("Magnitude", fontsize=6.5)
    ax_a.set_title("(a) Opportunity cost of GHI-only targeting", fontsize=7,
                   fontweight="bold", pad=6, loc="left")
    ax_a.legend(fontsize=5.5, frameon=False, loc="lower right")

    # summary box
    summ = (f"4 cities missed:\n"
            f"{', '.join(missed_names)}\n"
            f"Population: {miss_pop:.0f} M\n"
            f"CO$_2$ avoided: {miss_co2:,.0f} kt/yr")
    ax_a.text(0.98, 0.98, summ, transform=ax_a.transAxes,
              fontsize=5.2, va="top", ha="right", linespacing=1.5,
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF8F0",
                        edgecolor="#CCAAAA", linewidth=0.4))

    ax_a.xaxis.grid(True, linewidth=0.3, color="#E8E8E8", zorder=0)
    ax_a.set_axisbelow(True)

    # ── Panel (b): scenario distribution ────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])

    scenario_order = [
        ("baseline",       "Baseline\n(2024)"),
        ("carbon_pricing", "Carbon 100\nCNY/tCO$_2$"),
        ("cost_reduction", "PV Cost\n−50%"),
        ("aggressive",     "Aggressive\nPolicy"),
    ]

    y_pos = np.arange(len(scenario_order))
    bar_h = 0.55

    for yi, (sid, slabel) in enumerate(scenario_order):
        grp = sc[sc["scenario"] == sid]
        n_h = (grp["suitability"] == "High").sum()
        n_m = (grp["suitability"] == "Medium").sum()
        n_l = (grp["suitability"] == "Low").sum()
        total_n = len(grp)

        ax_b.barh(yi, n_h,               bar_h, color=SUIT_COLORS["High"],   left=0)
        ax_b.barh(yi, n_m,               bar_h, color=SUIT_COLORS["Medium"], left=n_h)
        ax_b.barh(yi, n_l,               bar_h, color=SUIT_COLORS["Low"],    left=n_h+n_m)

        # count labels
        for x0, cnt, clr in [(0, n_h, "white"), (n_h, n_m, "white"),
                              (n_h+n_m, n_l, "white")]:
            if cnt >= 2:
                ax_b.text(x0 + cnt/2, yi, str(cnt),
                          ha="center", va="center", fontsize=6.0,
                          color=clr, fontweight="bold")

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([s for _, s in scenario_order], fontsize=6.0)
    ax_b.set_xlim(0, 41)
    ax_b.set_xlabel("Number of cities (n = 41)", fontsize=6.5)
    ax_b.set_title("(b) Scenario suitability distribution", fontsize=7,
                   fontweight="bold", pad=6, loc="left")

    handles_s = [mpatches.Patch(color=SUIT_COLORS["High"],   label="High"),
                 mpatches.Patch(color=SUIT_COLORS["Medium"], label="Medium"),
                 mpatches.Patch(color=SUIT_COLORS["Low"],    label="Low")]
    ax_b.legend(handles=handles_s, fontsize=5.5, frameon=False,
                loc="lower right", handlelength=0.9)

    ax_b.xaxis.grid(True, linewidth=0.3, color="#E8E8E8", zorder=0)
    ax_b.set_axisbelow(True)

    for fmt in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig3_policy_consequences.{fmt}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("Fig 3 saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4  (China map — optional, requires cartopy; graceful fallback)
# ══════════════════════════════════════════════════════════════════════════════
def make_fig4():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        HAS_CARTOPY = True
    except ImportError:
        HAS_CARTOPY = False

    sm = pd.read_csv(SM_PATH)
    ii = pd.read_csv(II_PATH)
    # SM has capitalised names, II has lowercase — merge on lowercased key
    ii_ghi = ii[["city", "d1_1_ghi_annual_kwh"]].copy()
    ii_ghi["city_key"] = ii_ghi["city"].str.lower()
    sm2 = sm.copy()
    sm2["city_key"] = sm2["city"].str.lower()
    df = sm2.merge(ii_ghi[["city_key", "d1_1_ghi_annual_kwh"]], on="city_key", how="left")

    # city coordinates — pulled from integrated_indicators if available, else use config
    CITY_COORDS = {
        "Beijing": (116.39, 39.91), "Tianjin": (117.20, 39.12),
        "Harbin": (126.64, 45.75), "Changchun": (125.32, 43.82),
        "Shenyang": (123.46, 41.79), "Dalian": (121.62, 38.91),
        "Hohhot": (111.65, 40.82), "Urumqi": (87.61, 43.79),
        "Tangshan": (118.18, 39.63), "Jinan": (117.00, 36.66),
        "Zhengzhou": (113.63, 34.75), "Xian": (108.94, 34.26),
        "Shijiazhuang": (114.51, 38.05), "Taiyuan": (112.55, 37.87),
        "Lanzhou": (103.73, 36.04), "Yinchuan": (106.27, 38.47),
        "Xining": (101.74, 36.62), "Qingdao": (120.38, 36.07),
        "Wuxi": (120.30, 31.57), "Suzhou": (120.62, 31.32),
        "Shanghai": (121.47, 31.22), "Nanjing": (118.78, 32.04),
        "Hangzhou": (120.15, 30.26), "Hefei": (117.28, 31.86),
        "Wuhan": (114.30, 30.59), "Changsha": (112.98, 28.20),
        "Nanchang": (115.86, 28.68), "Ningbo": (121.55, 29.87),
        "Chengdu": (104.06, 30.67), "Chongqing": (106.55, 29.56),
        "Shenzhen": (114.06, 22.53), "Guangzhou": (113.27, 23.13),
        "Xiamen": (118.09, 24.48), "Fuzhou": (119.30, 26.08),
        "Nanning": (108.37, 22.82), "Haikou": (110.33, 20.02),
        "Kunming": (102.72, 25.04), "Guiyang": (106.71, 26.57),
        "Lhasa": (91.11, 29.65),
        "Hong Kong": (114.17, 22.32), "Taipei": (121.57, 25.03),
    }

    # GHI tercile for dot colour
    q33 = df["d1_1_ghi_annual_kwh"].quantile(1/3)
    q67 = df["d1_1_ghi_annual_kwh"].quantile(2/3)
    def ghi_tercile_color(ghi):
        if ghi >= q67: return "#D62728"
        if ghi >= q33: return "#FF7F0E"
        return "#2CA02C"

    if HAS_CARTOPY:
        proj = ccrs.LambertConformal(central_longitude=105, central_latitude=35,
                                     standard_parallels=(25, 47))
        fig  = plt.figure(figsize=(5.0, 4.5))
        ax   = fig.add_subplot(1, 1, 1, projection=proj)
        ax.set_extent([70, 140, 15, 55], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.LAND,       facecolor="#F5F3EE", edgecolor="none")
        ax.add_feature(cfeature.OCEAN,      facecolor="#D6EAF8", edgecolor="none")
        ax.add_feature(cfeature.BORDERS,    linewidth=0.4, edgecolor="#AAAAAA")
        ax.add_feature(cfeature.COASTLINE,  linewidth=0.4, edgecolor="#888888")
        ax.add_feature(cfeature.RIVERS,     linewidth=0.2, edgecolor="#AADDEE",
                       alpha=0.4)

        for _, row in df.iterrows():
            cname = row["city"]
            if cname not in CITY_COORDS:
                continue
            lon, lat = CITY_COORDS[cname]
            size  = 20 + row["fdsi_score"] * 80
            color = ghi_tercile_color(row["d1_1_ghi_annual_kwh"])
            ax.scatter(lon, lat, s=size, c=color, alpha=0.82,
                       transform=ccrs.PlateCarree(), zorder=5,
                       edgecolors="white", linewidths=0.4)

        # map disclaimer
        ax.text(0.01, 0.01,
                "Map lines delineate study areas and do not necessarily\n"
                "depict accepted national boundaries.",
                transform=ax.transAxes, fontsize=4.5, color="#666666",
                va="bottom", ha="left",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none",
                          pad=2))

        ax.set_title("FDSI suitability score across 41 cities\n"
                     "(dot size = FDSI; dot colour = GHI tercile)",
                     fontsize=6.5, pad=5)

        # legend: GHI tercile
        for col, lab in [("#D62728", "GHI High (top 1/3)"),
                         ("#FF7F0E", "GHI Medium"),
                         ("#2CA02C", "GHI Low (bottom 1/3)")]:
            ax.scatter([], [], c=col, s=30, label=lab,
                       transform=ccrs.PlateCarree())
        ax.scatter([], [], c="grey", s=20, label="FDSI ≈ 0.2", edgecolors="white",
                   linewidths=0.4, transform=ccrs.PlateCarree())
        ax.scatter([], [], c="grey", s=80, label="FDSI ≈ 0.7", edgecolors="white",
                   linewidths=0.4, transform=ccrs.PlateCarree())
        ax.legend(fontsize=5.0, frameon=True, framealpha=0.8,
                  loc="lower left", handletextpad=0.3)

    else:
        # fallback: simple scatter on lat/lon axes
        fig, ax = plt.subplots(figsize=(5.0, 4.5))
        for _, row in df.iterrows():
            cname = row["city"]
            if cname not in CITY_COORDS:
                continue
            lon, lat = CITY_COORDS[cname]
            size  = 30 + row["fdsi_score"] * 100
            color = ghi_tercile_color(row["d1_1_ghi_annual_kwh"])
            ax.scatter(lon, lat, s=size, c=color, alpha=0.8, zorder=5,
                       edgecolors="white", linewidths=0.4)
        ax.set_xlabel("Longitude (°E)", fontsize=6.5)
        ax.set_ylabel("Latitude (°N)", fontsize=6.5)
        ax.set_title("FDSI suitability score across 41 cities\n"
                     "(dot size = FDSI; dot colour = GHI tercile; "
                     "install cartopy for full map)",
                     fontsize=6.0)
        ax.text(0.01, 0.01,
                "Map lines delineate study areas and do not necessarily\n"
                "depict accepted national boundaries.",
                transform=ax.transAxes, fontsize=4.5, color="#666666",
                va="bottom", ha="left")
        for col, lab in [("#D62728", "GHI High"), ("#FF7F0E", "GHI Medium"),
                         ("#2CA02C", "GHI Low")]:
            ax.scatter([], [], c=col, s=40, label=lab)
        ax.legend(fontsize=5.5, frameon=False)
        ax.grid(linewidth=0.3, color="#E8E8E8")

    for fmt in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig4_spatial_distribution.{fmt}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("Fig 4 saved.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures...")
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    print("\nAll figures written to", FIG_DIR)
