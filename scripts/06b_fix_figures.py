#!/usr/bin/env python3
"""
Fix 3 broken figures. Run from project root:
    python3 scripts/06b_fix_figures.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

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
CITY_ORDER = ['shenzhen', 'beijing', 'kunming', 'changsha', 'harbin']
CITY_LABELS = {
    'shenzhen': 'Shenzhen', 'beijing': 'Beijing', 'kunming': 'Kunming',
    'changsha': 'Changsha', 'harbin': 'Harbin'
}
CLIMATE_LABELS = {
    'shenzhen': 'HSWW', 'beijing': 'Cold', 'kunming': 'Mild',
    'changsha': 'HSCW', 'harbin': 'Severe Cold'
}
CITY_COLORS = {
    'shenzhen': '#E63946', 'beijing': '#457B9D', 'kunming': '#2A9D8F',
    'changsha': '#E9C46A', 'harbin': '#264653',
}

def city_label_short(c):
    return CITY_LABELS[c]


# ══════════════════════════════════════════════════════════════════════════
# FIX 1: Monthly GHI — robust PVGIS hourly parsing
# ══════════════════════════════════════════════════════════════════════════
print("[FIX 1/3] Monthly GHI boxplot...")

fig, axes = plt.subplots(1, 5, figsize=(16, 4.5), sharey=True)

for i, c in enumerate(CITY_ORDER):
    ax = axes[i]
    fpath = f'results/pvgis/{c}_hourly.csv'

    try:
        # PVGIS hourly CSV has metadata lines at top. Find the actual header.
        with open(fpath, 'r') as f:
            lines = f.readlines()

        # Find header line: it contains 'time' (case-insensitive)
        header_idx = None
        for idx, line in enumerate(lines):
            if 'time' in line.lower() and ',' in line:
                header_idx = idx
                break

        if header_idx is None:
            raise ValueError("Could not find 'time' header row")

        # Also find where data ends (PVGIS sometimes has footer lines)
        # Data lines start with a digit (year)
        data_end = len(lines)
        for idx in range(header_idx + 1, len(lines)):
            stripped = lines[idx].strip()
            if stripped == '' or (not stripped[0].isdigit()):
                data_end = idx
                break

        df = pd.read_csv(fpath, skiprows=header_idx, nrows=data_end - header_idx - 1)

        # Parse time column (first column)
        time_col = df.columns[0]
        # PVGIS formats: "20050101:0010" or "2005-01-01 00:10" or "200501010010"
        time_str = df[time_col].astype(str)

        # Try removing colon in middle: "20050101:0010" -> "200501010010"
        time_clean = time_str.str.replace(':', '', regex=False)

        # Extract year and month
        df['year'] = time_clean.str[:4].astype(int)
        df['month'] = time_clean.str[4:6].astype(int)

        # Find GHI column: look for G(h), Gb(n), G(i), or just take column index 1
        ghi_col = None
        for col in df.columns:
            col_clean = col.strip()
            if col_clean in ['G(h)', 'Gb(n)', 'G(i)', 'Gd(h)']:
                if col_clean == 'G(h)':  # prefer G(h) = global horizontal
                    ghi_col = col
                    break
            if col_clean.startswith('G(') and ghi_col is None:
                ghi_col = col
        if ghi_col is None:
            # Fallback: second column (first after time)
            ghi_col = df.columns[1]

        df[ghi_col] = pd.to_numeric(df[ghi_col], errors='coerce')

        # Aggregate: monthly total GHI per year (W/m² hourly -> kWh/m²/month)
        monthly = df.groupby(['year', 'month'])[ghi_col].sum() / 1000.0
        monthly = monthly.reset_index()
        monthly.columns = ['year', 'month', 'ghi']

        # Remove incomplete years (first and last might be partial)
        year_counts = monthly.groupby('year').size()
        complete_years = year_counts[year_counts == 12].index
        monthly = monthly[monthly['year'].isin(complete_years)]

        if len(monthly) == 0:
            raise ValueError("No complete years found")

        # Boxplot by month
        bp_data = [monthly[monthly['month'] == m]['ghi'].values for m in range(1, 13)]
        bp = ax.boxplot(bp_data, positions=range(1, 13),
                        boxprops=dict(color=CITY_COLORS[c], linewidth=1.2),
                        medianprops=dict(color=CITY_COLORS[c], linewidth=2),
                        whiskerprops=dict(color=CITY_COLORS[c]),
                        capprops=dict(color=CITY_COLORS[c]),
                        flierprops=dict(markeredgecolor=CITY_COLORS[c], markersize=3),
                        patch_artist=True, widths=0.6)
        for patch in bp['boxes']:
            patch.set_facecolor(CITY_COLORS[c])
            patch.set_alpha(0.3)

        ax.set_title(city_label_short(c), fontweight='bold', color=CITY_COLORS[c])
        ax.set_xlabel('Month')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels([str(m) for m in range(1, 13)], fontsize=7)
        if i == 0:
            ax.set_ylabel('Monthly GHI (kWh/m²)')

        n_years = len(complete_years)
        ax.text(0.95, 0.95, f'{n_years} yrs', transform=ax.transAxes,
                ha='right', va='top', fontsize=8, color='gray')

    except Exception as e:
        ax.text(0.5, 0.5, f'Error:\n{str(e)[:80]}', transform=ax.transAxes,
                ha='center', va='center', fontsize=7, wrap=True)
        ax.set_title(city_label_short(c))

plt.suptitle('Monthly GHI Distribution (Inter-annual Variability)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGDIR / 'fig_monthly_ghi_boxplot.png')
plt.close()
print("  -> fig_monthly_ghi_boxplot.png")


# ══════════════════════════════════════════════════════════════════════════
# FIX 2: Building height violin — read from GeoPackage
# ══════════════════════════════════════════════════════════════════════════
print("[FIX 2/3] Building height violin (from .gpkg)...")

try:
    import geopandas as gpd
    HAS_GPD = True
except ImportError:
    HAS_GPD = False
    print("  WARNING: geopandas not available, skipping gpkg-based violin")

if HAS_GPD:
    fig, ax = plt.subplots(figsize=(10, 5))

    all_heights = {}
    for c in CITY_ORDER:
        gpkg_path = f'results/morphology/{c}_buildings_classified.gpkg'
        try:
            gdf = gpd.read_file(gpkg_path)
            h = gdf['height_m'].dropna()
            h = h[h > 0].clip(upper=150)
            all_heights[c] = h.values
            print(f"    {c}: {len(h)} buildings loaded")
        except Exception as e:
            print(f"    {c}: error — {e}")
            all_heights[c] = np.array([])

    valid_cities = [c for c in CITY_ORDER if len(all_heights[c]) > 10]

    if valid_cities:
        data_list = [all_heights[c] for c in valid_cities]
        parts = ax.violinplot(data_list, positions=range(len(valid_cities)),
                              showmeans=True, showmedians=True, showextrema=False,
                              widths=0.7)

        for j, pc in enumerate(parts['bodies']):
            pc.set_facecolor(CITY_COLORS[valid_cities[j]])
            pc.set_alpha(0.65)
            pc.set_edgecolor(CITY_COLORS[valid_cities[j]])

        parts['cmeans'].set_color('black')
        parts['cmeans'].set_linewidth(1.5)
        parts['cmedians'].set_color('white')
        parts['cmedians'].set_linewidth(2)

        ax.set_xticks(range(len(valid_cities)))
        labels = []
        for c in valid_cities:
            n = len(all_heights[c])
            labels.append(f"{city_label_short(c)}\n({CLIMATE_LABELS[c]})\nn={n:,}")
        ax.set_xticklabels(labels, fontsize=9)

        ax.set_ylabel('Building Height (m)')
        ax.set_title('Residential Building Height Distribution by City', fontweight='bold')

        # Reference lines
        for h_ref, label, ls in [(12, '12m (mid-rise)', ':'),
                                  (24, '24m (mid-high)', '--'),
                                  (60, '60m (high-rise)', '-.')]:
            if h_ref < ax.get_ylim()[1]:
                ax.axhline(y=h_ref, color='gray', linestyle=ls, linewidth=0.7, alpha=0.5)
                ax.text(len(valid_cities)-0.5, h_ref+0.5, label, fontsize=7, color='gray', ha='right')

        ax.grid(True, axis='y', alpha=0.2)

    plt.tight_layout()
    plt.savefig(FIGDIR / 'fig_height_violin.png')
    plt.close()
    print("  -> fig_height_violin.png")


# ══════════════════════════════════════════════════════════════════════════
# FIX 3: Dimension scores + FDSI — robust city matching
# ══════════════════════════════════════════════════════════════════════════
print("[FIX 3/3] Dimension scores + FDSI line...")

matrix = pd.read_csv('results/fdsi/suitability_matrix.csv')

# Debug: print city column values to understand the mismatch
print(f"  suitability_matrix.csv 'city' column values: {matrix['city'].tolist()}")

# Try matching by multiple possible columns
# The city column might be full names like "Shenzhen" or keys like "shenzhen"
# or there might be a separate name column

# Build a lookup: try city column first, then name_en, then city_cn
mat = matrix.copy()

# Normalise city key: lowercase, strip
mat['city_key'] = mat['city'].str.lower().str.strip()

# If city column has Chinese names, use city_cn or name_en to match
if 'name_en' in mat.columns:
    mat['city_key'] = mat['name_en'].str.lower().str.strip()
elif mat['city_key'].iloc[0] not in CITY_ORDER:
    # Try using first column values as-is
    print(f"  Trying alternate matching...")
    # Maybe city column has values like 'Shenzhen', 'Beijing' etc
    name_to_key = {v.lower(): k for k, v in CITY_LABELS.items()}
    mat['city_key'] = mat['city'].str.lower().str.strip().map(
        lambda x: name_to_key.get(x, x)
    )

print(f"  Resolved city keys: {mat['city_key'].tolist()}")

dims = ['D1_score', 'D2_score', 'D3_score', 'D4_score', 'D5_score']
dim_labels = ['D1\nClimate', 'D2\nMorphology', 'D3\nTechnical', 'D4\nEconomic', 'D5\nCertainty']

# Check which dim columns actually exist
available_dims = [d for d in dims if d in mat.columns]
if not available_dims:
    print(f"  Available columns: {list(mat.columns)}")
    print("  ERROR: No D1_score...D5_score columns found. Skipping Fig 8.")
else:
    mat_idx = mat.set_index('city_key')
    valid_cities = [c for c in CITY_ORDER if c in mat_idx.index]

    if not valid_cities:
        print(f"  ERROR: No matching cities found. city_key index: {mat_idx.index.tolist()}")
    else:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        x = np.arange(len(available_dims))
        width = 0.14
        n_cities = len(valid_cities)

        for i, c in enumerate(valid_cities):
            vals = [mat_idx.loc[c, d] if d in mat_idx.columns else 0 for d in available_dims]
            offset = (i - n_cities/2 + 0.5) * width
            ax1.bar(x + offset, vals, width, color=CITY_COLORS[c],
                    label=city_label_short(c), edgecolor='white', linewidth=0.5)

        ax1.set_xticks(x)
        ax1.set_xticklabels([dim_labels[dims.index(d)] for d in available_dims])
        ax1.set_ylabel('Normalised Dimension Score')
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, axis='y', alpha=0.3)
        ax1.legend(loc='upper right', ncol=5, framealpha=0.9)

        # FDSI line on twin axis
        ax2 = ax1.twinx()
        fdsi_vals = [mat_idx.loc[c, 'fdsi_score'] for c in valid_cities]
        x_fdsi = np.arange(len(valid_cities))
        for j, c in enumerate(valid_cities):
            ax2.scatter(j, fdsi_vals[j], s=120, c=CITY_COLORS[c],
                       edgecolors='black', linewidth=1.5, zorder=10)
            ax2.annotate(f'{fdsi_vals[j]:.3f}', (j, fdsi_vals[j]),
                        textcoords='offset points', xytext=(0, 12),
                        ha='center', fontsize=10, fontweight='bold')
        ax2.plot(x_fdsi, fdsi_vals, 'k--', linewidth=1, alpha=0.4, zorder=5)
        ax2.set_ylabel('FDSI Score', fontsize=11)
        ax2.set_ylim(0, 0.85)

        # City labels on top x-axis
        ax_top = ax1.twiny()
        ax_top.set_xlim(ax1.get_xlim())
        ax_top.set_xticks(x_fdsi)
        ax_top.set_xticklabels([f"{city_label_short(c)}\n(#{i+1})" for i, c in enumerate(valid_cities)],
                                fontsize=9)

        ax1.set_title('Five-Dimension Scores with FDSI Ranking', fontweight='bold', pad=40)
        plt.tight_layout()
        plt.savefig(FIGDIR / 'fig_dimension_scores_fdsi.png')
        plt.close()
        print("  -> fig_dimension_scores_fdsi.png")


# ══════════════════════════════════════════════════════════════════════════
# Also fix the CI subscript encoding issue in Fig 5
# ══════════════════════════════════════════════════════════════════════════
print("[BONUS] Fixing MC PBT CI label encoding...")

mc_data = {}
for c in CITY_ORDER:
    mc_data[c] = pd.read_csv(f'results/energy/{c}_mc_summary.csv')

fig, ax = plt.subplots(figsize=(10, 5))
pbt_stats = []
for c in CITY_ORDER:
    mc = mc_data[c]
    p025 = mc['d5_2_pbt_p025'].values[0]
    p975 = mc['d5_2_pbt_p975'].values[0]
    ci_width = mc['d5_2_pbt_ci95_width'].values[0]
    pbt_stats.append({'city': c, 'pbt_low': p025, 'pbt_high': p975, 'ci_width': ci_width})

y_pos = np.arange(len(CITY_ORDER))
for i, ps in enumerate(pbt_stats):
    c = ps['city']
    ax.barh(i, ps['pbt_high'] - ps['pbt_low'], left=ps['pbt_low'],
            height=0.5, color=CITY_COLORS[c], alpha=0.7, edgecolor='white')
    mid = (ps['pbt_low'] + ps['pbt_high']) / 2
    ax.plot(mid, i, 'o', color='white', markersize=6, zorder=5)
    # Use plain ASCII for the label to avoid encoding issues
    ax.text(ps['pbt_high'] + 0.1, i, f'CI95 = {ps["ci_width"]:.2f} yr',
            va='center', fontsize=9, color=CITY_COLORS[c], fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels([f"{CITY_LABELS[c]}\n({CLIMATE_LABELS[c]})" for c in CITY_ORDER])
ax.set_xlabel('Payback Time (years)')
ax.set_title('Monte Carlo PBT Uncertainty: 95% Confidence Intervals (N=10,000)', fontweight='bold')
ax.grid(True, axis='x', alpha=0.3)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(FIGDIR / 'fig_mc_pbt_ci.png')
plt.close()
print("  -> fig_mc_pbt_ci.png (fixed labels)")

print("\nDone! Fixed figures saved to figures/")
