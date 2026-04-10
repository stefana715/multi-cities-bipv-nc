"""
nc_fig2a_changsha_chengdu.py
Fig. 2a — Changsha vs Chengdu dimension comparison (grouped bar chart)
Nature Communications style
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SM_PATH  = ROOT / "results" / "fdsi" / "suitability_matrix.csv"
II_PATH  = ROOT / "results" / "fdsi" / "integrated_indicators.csv"
OUT_DIR  = ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)

# ── load data ────────────────────────────────────────────────────────────────
sm = pd.read_csv(SM_PATH)
ii = pd.read_csv(II_PATH)

def get_city(city_sm: str, city_ii: str) -> dict:
    s = sm[sm["city"] == city_sm].iloc[0]
    i = ii[ii["city"] == city_ii].iloc[0]
    return {
        "name":   city_sm,
        "GHI":    float(i["d1_1_ghi_annual_kwh"]),
        "rank":   int(s["fdsi_rank"]),
        "FDSI":   float(s["fdsi_score"]),
        "D1":     float(s["D1_score"]),
        "D2":     float(s["D2_score"]),
        "D3":     float(s["D3_score"]),
        "D4":     float(s["D4_score"]),
        "D5":     float(s["D5_score"]),
    }

csa = get_city("Changsha", "changsha")
cdu = get_city("Chengdu",  "chengdu")

delta_ghi  = csa["GHI"]  - cdu["GHI"]
delta_rank = cdu["rank"] - csa["rank"]   # positive = Chengdu ranks lower (worse)

# ── figure constants ──────────────────────────────────────────────────────────
DIMS   = ["D1", "D2", "D3", "D4", "D5", "FDSI"]
LABELS = ["D1\n(Climate)", "D2\n(Morphology)", "D3\n(Technical)",
          "D4\n(Economic)", "D5\n(Uncertainty)", "FDSI\n(Composite)"]

C_CSA = "#2166AC"   # Changsha — blue
C_CDU = "#D6604D"   # Chengdu  — red-orange
C_FDSI_BAR = 0.85   # alpha for FDSI bars to differentiate from D-score bars

vals_csa  = [csa[d] for d in DIMS]
vals_cdu  = [cdu[d] for d in DIMS]

# ── NC-style rcParams ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         7,
    "axes.linewidth":    0.6,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size":  2.5,
    "ytick.major.size":  2.5,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "figure.dpi":        300,
})

# ── layout ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.5, 2.6))

x      = np.arange(len(DIMS))
width  = 0.32
gap    = 0.04

# dim-score bars (D1–D5)
alpha_dim  = 1.0
alpha_fdsi = 0.85

bars_csa = ax.bar(x[:-1] - (width/2 + gap/2), vals_csa[:-1],
                  width, color=C_CSA, alpha=alpha_dim, linewidth=0.4,
                  edgecolor="white", label="Changsha")
bars_cdu = ax.bar(x[:-1] + (width/2 + gap/2), vals_cdu[:-1],
                  width, color=C_CDU, alpha=alpha_dim, linewidth=0.4,
                  edgecolor="white")

# FDSI bars — slightly wider + hatched to distinguish composite
fdsi_idx = len(DIMS) - 1
ax.bar(x[fdsi_idx] - (width/2 + gap/2), vals_csa[-1],
       width * 1.1, color=C_CSA, alpha=alpha_fdsi,
       linewidth=0.5, edgecolor="white", label="_nolegend_")
ax.bar(x[fdsi_idx] + (width/2 + gap/2), vals_cdu[-1],
       width * 1.1, color=C_CDU, alpha=alpha_fdsi,
       linewidth=0.5, edgecolor="white")

# value labels on bars (only FDSI)
for xi, val, color in [
    (x[fdsi_idx] - (width/2 + gap/2), vals_csa[-1], "white"),
    (x[fdsi_idx] + (width/2 + gap/2), vals_cdu[-1], "white"),
]:
    ax.text(xi, val - 0.02, f"{val:.3f}", ha="center", va="top",
            fontsize=5.5, color=color, fontweight="bold")

# vertical separator before FDSI
ax.axvline(x[fdsi_idx] - 0.60, color="#999999", linewidth=0.4, linestyle="--")

# ── annotation box ────────────────────────────────────────────────────────────
anno_txt = (
    rf"$\Delta$GHI = {delta_ghi:+.0f} kWh m$^{{-2}}$ yr$^{{-1}}$"
    "\n"
    rf"$\Delta$rank = {delta_rank:+d}  (#{csa['rank']} vs #{cdu['rank']})"
)
ax.text(0.02, 0.97, anno_txt,
        transform=ax.transAxes,
        fontsize=5.5, va="top", ha="left",
        linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F0F0",
                  edgecolor="#AAAAAA", linewidth=0.5, alpha=0.9))

# ── axes decoration ───────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=6.0)
ax.set_xlim(-0.65, len(DIMS) - 0.35)
ax.set_ylim(0, 1.05)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=6.0)
ax.set_ylabel("Normalised score (0–1)", fontsize=6.5)

# title + subtitle
ax.set_title("Changsha versus Chengdu",
             fontsize=7.5, fontweight="bold", pad=14, loc="left")
ax.text(0.0, 1.065,
        "Similar irradiance, strongly different composite suitability",
        transform=ax.transAxes, fontsize=6.0, color="#555555",
        va="bottom", ha="left")

# legend
patch_csa = mpatches.Patch(color=C_CSA, label=f"Changsha (rank #{csa['rank']})")
patch_cdu = mpatches.Patch(color=C_CDU, label=f"Chengdu  (rank #{cdu['rank']})")
ax.legend(handles=[patch_csa, patch_cdu],
          fontsize=5.5, frameon=False,
          loc="upper right", bbox_to_anchor=(1.0, 1.02),
          handlelength=1.0, handletextpad=0.4)

# light horizontal grid
ax.yaxis.grid(True, linewidth=0.3, color="#E0E0E0", zorder=0)
ax.set_axisbelow(True)

# ── save ──────────────────────────────────────────────────────────────────────
fig.tight_layout()
for fmt in ("png", "pdf"):
    out = OUT_DIR / f"fig2a_changsha_chengdu.{fmt}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
plt.close(fig)

# ── console summary ───────────────────────────────────────────────────────────
print("\n── City summary ──────────────────────────────────────────────")
for c in [csa, cdu]:
    print(f"{c['name']:10s}  GHI={c['GHI']:.0f}  rank=#{c['rank']}  FDSI={c['FDSI']:.4f}")
    print(f"           D1={c['D1']:.4f}  D2={c['D2']:.4f}  D3={c['D3']:.4f}  "
          f"D4={c['D4']:.4f}  D5={c['D5']:.4f}")
print(f"\nΔGHI  = {delta_ghi:+.0f} kWh m⁻² yr⁻¹")
print(f"Δrank = {delta_rank:+d}  (Chengdu ranks {abs(delta_rank)} places lower)")
