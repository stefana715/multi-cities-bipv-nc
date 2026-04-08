#!/usr/bin/env python3
"""
Paper 4 — Additional figures for dissertation final chapter.
Run from project root: python3 scripts/06_additional_figures.py

Generates 8 new figures into figures/ directory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# ── Setup ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

FIGDIR = Path('figures')
FIGDIR.mkdir(exist_ok=True)

# City display order (by FDSI rank) and colors
CITY_ORDER = ['shenzhen', 'beijing', 'kunming', 'changsha', 'harbin']
CITY_LABELS = {
    'shenzhen': 'Shenzhen', 'beijing': 'Beijing', 'kunming': 'Kunming',
    'changsha': 'Changsha', 'harbin': 'Harbin'
}
CLIMATE_LABELS = {
    'shenzhen': 'HSWW', 'beijing': 'Cold', 'kunming': 'Mild',
    'changsha': 'HSCW', 'harbin': 'Severe Cold'
}
# Color palette: warm→cool by climate
CITY_COLORS = {
    'shenzhen': '#E63946',   # red — HSWW
    'beijing':  '#457B9D',   # steel blue — Cold
    'kunming':  '#2A9D8F',   # teal — Mild
    'changsha': '#E9C46A',   # gold — HSCW
    'harbin':   '#264653',   # dark teal — Severe Cold
}

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
energy = pd.read_csv('results/energy/cross_city_d1d4d5.csv')
morph  = pd.read_csv('results/morphology/cross_city_d2d3_summary.csv')
matrix = pd.read_csv('results/fdsi/suitability_matrix.csv')
scores = pd.read_csv('results/fdsi/fdsi_scores.csv')

# Per-city MC and Sobol
mc_data = {}
sobol_data = {}
for c in CITY_ORDER:
    mc_data[c] = pd.read_csv(f'results/energy/{c}_mc_summary.csv')
    sobol_data[c] = pd.read_csv(f'results/energy/{c}_sobol_indices.csv')

# Set city as index for easy lookup
energy_idx = energy.set_index('city')
morph_idx  = morph.set_index('city')

def city_label(c):
    return f"{CITY_LABELS[c]}\n({CLIMATE_LABELS[c]})"

def city_label_short(c):
    return CITY_LABELS[c]

# ══════════════════════════════════════════════════════════════════════════
# FIG 1: Monthly GHI distribution — boxplot from hourly data
# ══════════════════════════════════════════════════════════════════════════
print("[1/8] Monthly GHI boxplot...")

fig, axes = plt.subplots(1, 5, figsize=(16, 4), sharey=True)
for i, c in enumerate(CITY_ORDER):
    ax = axes[i]
    # PVGIS hourly files have metadata header rows; detect and skip
    fpath = f'results/pvgis/{c}_hourly.csv'
    # Find the header row (contains 'time')
    with open(fpath, 'r') as f:
        for skip_n, line in enumerate(f):
            if 'time' in line.lower():
                break
    try:
        df_h = pd.read_csv(fpath, skiprows=skip_n)
        # PVGIS hourly typically has columns: time, G(i), Gb(i), Gd(i), T2m, ...
        # Find GHI column (usually G(h) or Gb(n) — look for 'G(h)' or 'Gb' or 'G(i)')
        ghi_col = [col for col in df_h.columns if col.strip().startswith('G(') or col.strip() == 'Gb(i)']
        if not ghi_col:
            # fallback: use first numeric column after 'time'
            ghi_col = [df_h.columns[1]]
        ghi_col = ghi_col[0]
        df_h['month'] = pd.to_datetime(df_h.iloc[:, 0].astype(str).str[:6], format='%Y%m', errors='coerce').dt.month
        if df_h['month'].isna().all():
            # Try different time parsing
            df_h['month'] = pd.to_datetime(df_h.iloc[:, 0], errors='coerce').dt.month
        monthly_ghi = df_h.groupby('month')[ghi_col].sum() / 1000  # Wh→kWh per m²
        # If we have multi-year data, group by year+month then boxplot
        df_h['year'] = pd.to_datetime(df_h.iloc[:, 0].astype(str).str[:6], format='%Y%m', errors='coerce').dt.year
        if df_h['year'].isna().all():
            df_h['year'] = pd.to_datetime(df_h.iloc[:, 0], errors='coerce').dt.year
        ym = df_h.groupby(['year', 'month'])[ghi_col].sum() / 1000
        ym = ym.reset_index()
        ym.columns = ['year', 'month', 'ghi_monthly']
        ym.boxplot(column='ghi_monthly', by='month', ax=ax,
                   boxprops=dict(color=CITY_COLORS[c]),
                   medianprops=dict(color=CITY_COLORS[c], linewidth=2),
                   whiskerprops=dict(color=CITY_COLORS[c]),
                   capprops=dict(color=CITY_COLORS[c]),
                   flierprops=dict(markeredgecolor=CITY_COLORS[c], markersize=3),
                   patch_artist=False, widths=0.6)
        ax.set_title(city_label_short(c), fontweight='bold', color=CITY_COLORS[c])
        ax.set_xlabel('Month')
        if i == 0:
            ax.set_ylabel('Monthly GHI (kWh/m²)')
        else:
            ax.set_ylabel('')
        ax.get_figure().suptitle('')  # remove pandas auto-title
    except Exception as e:
        ax.text(0.5, 0.5, f'Parse error:\n{str(e)[:60]}', transform=ax.transAxes,
                ha='center', va='center', fontsize=8)
        ax.set_title(city_label_short(c))

plt.suptitle('Monthly GHI Distribution by City (2005–2020)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGDIR / 'fig_monthly_ghi_boxplot.png')
plt.close()
print("  → fig_monthly_ghi_boxplot.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 2: Building height distribution — violin plot
# ══════════════════════════════════════════════════════════════════════════
print("[2/8] Building height violin plot...")

fig, ax = plt.subplots(figsize=(10, 5))
height_data = []
for c in CITY_ORDER:
    try:
        d2 = pd.read_csv(f'results/morphology/{c}_d2_indicators.csv')
        # Find height column
        h_col = [col for col in d2.columns if 'height' in col.lower() and 'mean' not in col.lower() and 'std' not in col.lower()]
        if not h_col:
            h_col = [col for col in d2.columns if 'height' in col.lower()]
        if h_col:
            heights = d2[h_col[0]].dropna()
            heights = heights[heights > 0]
            # Cap at 150m for visualization
            heights = heights.clip(upper=150)
            for h in heights:
                height_data.append({'city': city_label_short(c), 'height': h, 'city_key': c})
    except Exception as e:
        print(f"  Warning: {c} d2_indicators — {e}")

if height_data:
    df_h = pd.DataFrame(height_data)
    order = [city_label_short(c) for c in CITY_ORDER if city_label_short(c) in df_h['city'].unique()]
    palette = {city_label_short(c): CITY_COLORS[c] for c in CITY_ORDER}

    parts = ax.violinplot(
        [df_h[df_h['city'] == lab]['height'].values for lab in order],
        positions=range(len(order)), showmeans=True, showmedians=True, showextrema=False
    )
    for i, pc in enumerate(parts['bodies']):
        c_key = CITY_ORDER[i] if i < len(CITY_ORDER) else CITY_ORDER[0]
        pc.set_facecolor(CITY_COLORS[c_key])
        pc.set_alpha(0.7)
    parts['cmeans'].set_color('black')
    parts['cmedians'].set_color('white')

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontweight='bold')
    # Add building count annotation
    for i, lab in enumerate(order):
        n = len(df_h[df_h['city'] == lab])
        ax.annotate(f'n={n:,}', (i, ax.get_ylim()[1]*0.95), ha='center', fontsize=8, color='gray')

    ax.set_ylabel('Building Height (m)')
    ax.set_title('Residential Building Height Distribution by City', fontweight='bold')
    ax.axhline(y=27, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(len(order)-0.5, 28, 'High-rise threshold (27m)', fontsize=7, color='gray', ha='right')
else:
    ax.text(0.5, 0.5, 'No height data found in d2_indicators files',
            transform=ax.transAxes, ha='center')

plt.tight_layout()
plt.savefig(FIGDIR / 'fig_height_violin.png')
plt.close()
print("  → fig_height_violin.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 3: Morphology typology stacked bar
# ══════════════════════════════════════════════════════════════════════════
print("[3/8] Typology stacked bar chart...")

fig, ax = plt.subplots(figsize=(10, 5))
typo_colors = {
    'low_rise': '#A8DADC', 'mid_rise': '#457B9D',
    'mid_high': '#1D3557', 'high_rise': '#E63946'
}
typo_labels = {
    'low_rise': 'Low-rise (<12m)', 'mid_rise': 'Mid-rise (12–24m)',
    'mid_high': 'Mid-high (24–60m)', 'high_rise': 'High-rise (>60m)'
}
typo_order = ['low_rise', 'mid_rise', 'mid_high', 'high_rise']

bar_data = {}
for c in CITY_ORDER:
    try:
        ts = pd.read_csv(f'results/morphology/{c}_typology_stats.csv')
        # Expect columns like: typology/type, count/percentage
        type_col = [col for col in ts.columns if 'typo' in col.lower() or 'type' in col.lower() or 'class' in col.lower()]
        count_col = [col for col in ts.columns if 'count' in col.lower() or 'n_build' in col.lower() or 'pct' in col.lower() or 'percent' in col.lower()]
        if type_col and count_col:
            ts_dict = dict(zip(ts[type_col[0]].str.lower().str.strip(), ts[count_col[0]]))
            total = sum(ts_dict.values())
            bar_data[c] = {k: ts_dict.get(k, 0) / total * 100 if total > 0 else 0 for k in typo_order}
        else:
            # Try using city_typology column
            bar_data[c] = {}
    except Exception as e:
        print(f"  Warning: {c} typology_stats — {e}")

if bar_data:
    x = np.arange(len(CITY_ORDER))
    bottom = np.zeros(len(CITY_ORDER))
    for typo in typo_order:
        vals = [bar_data.get(c, {}).get(typo, 0) for c in CITY_ORDER]
        ax.bar(x, vals, bottom=bottom, color=typo_colors[typo],
               label=typo_labels[typo], width=0.6, edgecolor='white', linewidth=0.5)
        # Add percentage labels for segments > 5%
        for j, v in enumerate(vals):
            if v > 5:
                ax.text(j, bottom[j] + v/2, f'{v:.0f}%', ha='center', va='center',
                        fontsize=8, color='white' if typo in ['mid_high', 'high_rise', 'mid_rise'] else 'black')
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([city_label(c) for c in CITY_ORDER])
    ax.set_ylabel('Proportion (%)')
    ax.set_title('Residential Building Typology Distribution', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 105)

plt.tight_layout()
plt.savefig(FIGDIR / 'fig_typology_stacked_bar.png')
plt.close()
print("  → fig_typology_stacked_bar.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 4: LCOE vs PBT scatter (bubble = capacity, color = climate)
# ══════════════════════════════════════════════════════════════════════════
print("[4/8] LCOE vs PBT bubble scatter...")

fig, ax = plt.subplots(figsize=(8, 6))
for c in CITY_ORDER:
    row_e = energy_idx.loc[c]
    row_m = morph_idx.loc[c]
    lcoe = row_e['d4_1_lcoe_cny_kwh']
    pbt = row_e['d4_2_pbt_years']
    capacity = row_m['d3_4_total_deployable_mw']
    ax.scatter(lcoe, pbt, s=capacity * 0.4, c=CITY_COLORS[c],
               edgecolors='white', linewidth=1.5, zorder=5, alpha=0.85)
    # Label
    offset = {'shenzhen': (8, -12), 'beijing': (8, 8), 'kunming': (-10, 10),
              'changsha': (8, 8), 'harbin': (8, -12)}
    ax.annotate(f"{CITY_LABELS[c]}\n({capacity:.0f} MW)",
                (lcoe, pbt), textcoords='offset points',
                xytext=offset.get(c, (10, 5)), fontsize=9,
                arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

ax.set_xlabel('LCOE (CNY/kWh)')
ax.set_ylabel('Payback Time (years)')
ax.set_title('Economic Feasibility: LCOE vs Payback Time\n(bubble size = deployable capacity)',
             fontweight='bold')
ax.grid(True, alpha=0.3)

# Add reference lines
ax.axhline(y=5, color='green', linestyle='--', alpha=0.4, linewidth=0.8)
ax.text(ax.get_xlim()[0] + 0.001, 5.1, 'PBT = 5 yr', fontsize=7, color='green', alpha=0.6)

plt.tight_layout()
plt.savefig(FIGDIR / 'fig_lcoe_pbt_scatter.png')
plt.close()
print("  → fig_lcoe_pbt_scatter.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 5: MC PBT distribution — ridgeline / violin
# ══════════════════════════════════════════════════════════════════════════
print("[5/8] MC PBT uncertainty violin...")

fig, ax = plt.subplots(figsize=(10, 5))
pbt_stats = []
for c in CITY_ORDER:
    mc = mc_data[c]
    # Extract PBT CI bounds
    p025 = mc['d5_2_pbt_p025'].values[0]
    p975 = mc['d5_2_pbt_p975'].values[0]
    p05  = mc['mc_yield_p05'].values[0]  # yield percentiles for context
    p95  = mc['mc_yield_p95'].values[0]
    mean_y = mc['mc_yield_mean'].values[0]
    std_y  = mc['mc_yield_std'].values[0]
    ci_width = mc['d5_2_pbt_ci95_width'].values[0]
    pbt_stats.append({
        'city': c, 'pbt_low': p025, 'pbt_high': p975,
        'ci_width': ci_width
    })

# Horizontal bar chart showing CI range
y_pos = np.arange(len(CITY_ORDER))
for i, ps in enumerate(pbt_stats):
    c = ps['city']
    ax.barh(i, ps['pbt_high'] - ps['pbt_low'], left=ps['pbt_low'],
            height=0.5, color=CITY_COLORS[c], alpha=0.7, edgecolor='white')
    # Mark midpoint
    mid = (ps['pbt_low'] + ps['pbt_high']) / 2
    ax.plot(mid, i, 'o', color='white', markersize=6, zorder=5)
    # CI width annotation
    ax.text(ps['pbt_high'] + 0.1, i, f'CI₉₅ = {ps["ci_width"]:.2f} yr',
            va='center', fontsize=9, color=CITY_COLORS[c], fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels([city_label(c) for c in CITY_ORDER])
ax.set_xlabel('Payback Time (years)')
ax.set_title('Monte Carlo PBT Uncertainty: 95% Confidence Intervals (N=10,000)',
             fontweight='bold')
ax.grid(True, axis='x', alpha=0.3)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(FIGDIR / 'fig_mc_pbt_ci.png')
plt.close()
print("  → fig_mc_pbt_ci.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 6: Sobol indices — grouped bar (S1 vs interaction)
# ══════════════════════════════════════════════════════════════════════════
print("[6/8] Sobol sensitivity grouped bar...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Panel A: Yield Sobol S1 ---
ax = axes[0]
factors_yield = ['ghi_factor', 'module_efficiency', 'system_losses', 'pv_cost', 'elec_price_factor']
factor_labels = ['GHI', 'Module eff.', 'System losses', 'PV cost', 'Elec. price']
x = np.arange(len(factors_yield))
width = 0.15

for i, c in enumerate(CITY_ORDER):
    sb = sobol_data[c]
    s1_vals = []
    for f in factors_yield:
        col = f'sobol_yield_S1_{f}'
        if col in sb.columns:
            s1_vals.append(sb[col].values[0])
        else:
            s1_vals.append(0)
    ax.bar(x + i*width, s1_vals, width, color=CITY_COLORS[c],
           label=city_label_short(c), edgecolor='white', linewidth=0.5)

ax.set_xticks(x + width*2)
ax.set_xticklabels(factor_labels, rotation=15)
ax.set_ylabel('First-order Sobol index (S₁)')
ax.set_title('(a) Yield sensitivity', fontweight='bold')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, axis='y', alpha=0.3)

# --- Panel B: PBT Sobol S1 ---
ax = axes[1]
factors_pbt = ['ghi_factor', 'module_efficiency', 'system_losses', 'pv_cost', 'elec_price_factor']

for i, c in enumerate(CITY_ORDER):
    sb = sobol_data[c]
    s1_vals = []
    for f in factors_pbt:
        col = f'sobol_pbt_S1_{f}'
        if col in sb.columns:
            s1_vals.append(sb[col].values[0])
        else:
            s1_vals.append(0)
    ax.bar(x + i*width, s1_vals, width, color=CITY_COLORS[c],
           label=city_label_short(c), edgecolor='white', linewidth=0.5)

ax.set_xticks(x + width*2)
ax.set_xticklabels(factor_labels, rotation=15)
ax.set_ylabel('First-order Sobol index (S₁)')
ax.set_title('(b) Payback time sensitivity', fontweight='bold')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, axis='y', alpha=0.3)

plt.suptitle('Sobol First-Order Sensitivity Indices (N=4,096)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGDIR / 'fig_sobol_grouped_bar.png')
plt.close()
print("  → fig_sobol_grouped_bar.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 7: Suitability matrix heatmap (3×3 grid)
# ══════════════════════════════════════════════════════════════════════════
print("[7/8] Suitability matrix 3x3 heatmap...")

fig, ax = plt.subplots(figsize=(8, 6))

suit_levels = ['High', 'Medium', 'Low']
uncert_levels = ['Low', 'Medium', 'High']

# Create 3x3 grid
grid = np.zeros((3, 3))
city_positions = {}
for _, row in matrix.iterrows():
    s_idx = suit_levels.index(row['suitability']) if row['suitability'] in suit_levels else -1
    u_idx = uncert_levels.index(row['uncertainty']) if row['uncertainty'] in uncert_levels else -1
    if s_idx >= 0 and u_idx >= 0:
        grid[s_idx, u_idx] += 1
        key = (s_idx, u_idx)
        if key not in city_positions:
            city_positions[key] = []
        city_positions[key].append(row['city'])

# Background colors: green(good) → yellow → red(bad)
bg_colors = np.array([
    ['#2D6A4F', '#52B788', '#B7E4C7'],  # High suit: low/med/high uncert
    ['#74C69D', '#D4A373', '#E76F51'],   # Medium suit
    ['#D4A373', '#E76F51', '#9B2226'],   # Low suit
])

for i in range(3):
    for j in range(3):
        rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, facecolor=bg_colors[i][j], alpha=0.6)
        ax.add_patch(rect)
        # Add city names
        key = (i, j)
        if key in city_positions:
            cities = city_positions[key]
            text = '\n'.join([f"{CITY_LABELS.get(c, c)}" for c in cities])
            ax.text(j, i, text, ha='center', va='center', fontsize=11,
                    fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.4))

ax.set_xticks(range(3))
ax.set_xticklabels(uncert_levels, fontsize=11)
ax.set_yticks(range(3))
ax.set_yticklabels(suit_levels, fontsize=11)
ax.set_xlabel('Uncertainty Level (D5)', fontsize=12, fontweight='bold')
ax.set_ylabel('Suitability Level (FDSI)', fontsize=12, fontweight='bold')
ax.set_title('BIPV Deployment Suitability Matrix', fontsize=13, fontweight='bold')
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.5, 2.5)
ax.invert_yaxis()
ax.set_aspect('equal')

# Add corner annotations
ax.text(-0.4, -0.4, '★ PRIORITY', fontsize=8, color='white', fontweight='bold', va='top')
ax.text(2.4, 2.4, '✗ AVOID', fontsize=8, color='white', fontweight='bold', ha='right')

plt.tight_layout()
plt.savefig(FIGDIR / 'fig_suitability_matrix_3x3.png')
plt.close()
print("  → fig_suitability_matrix_3x3.png")


# ══════════════════════════════════════════════════════════════════════════
# FIG 8: D1-D5 dimension score comparison — grouped bar with FDSI line
# ══════════════════════════════════════════════════════════════════════════
print("[8/8] Dimension score grouped bar + FDSI line...")

fig, ax1 = plt.subplots(figsize=(12, 6))

dims = ['D1_score', 'D2_score', 'D3_score', 'D4_score', 'D5_score']
dim_labels = ['D1\nClimate', 'D2\nMorphology', 'D3\nTechnical', 'D4\nEconomic', 'D5\nCertainty']
x = np.arange(len(dims))
width = 0.15

mat = matrix.set_index('city')
for i, c in enumerate(CITY_ORDER):
    if c in mat.index:
        vals = [mat.loc[c, d] for d in dims]
        bars = ax1.bar(x + i*width, vals, width, color=CITY_COLORS[c],
                       label=city_label_short(c), edgecolor='white', linewidth=0.5)

ax1.set_xticks(x + width*2)
ax1.set_xticklabels(dim_labels)
ax1.set_ylabel('Normalised Dimension Score')
ax1.set_ylim(0, 1.05)
ax1.grid(True, axis='y', alpha=0.3)
ax1.legend(loc='upper left', ncol=5, framealpha=0.9)

# Overlay FDSI as line on secondary axis
ax2 = ax1.twinx()
fdsi_vals = [mat.loc[c, 'fdsi_score'] if c in mat.index else 0 for c in CITY_ORDER]
ax2.plot(range(len(CITY_ORDER)), fdsi_vals, 'ko-', markersize=8, linewidth=2, label='FDSI')
for i, (c, v) in enumerate(zip(CITY_ORDER, fdsi_vals)):
    ax2.annotate(f'{v:.3f}', (i, v), textcoords='offset points',
                 xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')
ax2.set_ylabel('FDSI Score')
ax2.set_ylim(0, 0.85)
ax2.set_xticks(range(len(CITY_ORDER)))
ax2.set_xticklabels([city_label_short(c) for c in CITY_ORDER])

# Title
ax1.set_title('Dimension Scores and FDSI Ranking by City', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGDIR / 'fig_dimension_scores_fdsi.png')
plt.close()
print("  → fig_dimension_scores_fdsi.png")


# ══════════════════════════════════════════════════════════════════════════
print("\n✅ All 8 figures saved to figures/")
print("Files:")
for f in sorted(FIGDIR.glob('fig_*.png')):
    print(f"  {f}")
