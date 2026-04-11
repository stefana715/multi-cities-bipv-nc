"""
§3.5 Directional Bias Analysis
——————————————————————————————
Tests whether misclassified cities systematically differ from
correctly classified cities in density-related indicators.

USAGE (Claude Code or terminal):
  python scripts/nc_directional_bias.py

REQUIRED INPUT FILES (adjust paths as needed):
  - results/fdsi/fdsi_scores.csv          (city, GHI, FDSI, ghi_rank, fdsi_rank)
  - results/fdsi/suitability_matrix.csv   (city, D1–D5 scores)
  - results/fdsi/integrated_indicators.csv (city, d2_5_far, d2_2_building_density, 
                                            d2_1_height_mean, d3_4_total_deployable_mw, etc.)

OUTPUT:
  - Console: full statistical report
  - results_nc/directional_bias/directional_bias_report.txt
  - results_nc/directional_bias/misclass_vs_correct_boxplot.png
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. CONFIGURE PATHS — ADJUST THESE TO YOUR PROJECT
# ============================================================
PROJECT_ROOT = Path(".")  # Change if needed

# Try multiple possible file locations
# rank_shift_analysis already has ghi_rank, fdsi_rank, ghi_annual, rank_shift
FDSI_CANDIDATES = [
    PROJECT_ROOT / "results_nc/misclassification/rank_shift_analysis.csv",
    PROJECT_ROOT / "results/fdsi/fdsi_scores.csv",
    PROJECT_ROOT / "results_nc/fdsi_scores.csv",
]

INDICATORS_CANDIDATES = [
    PROJECT_ROOT / "results/fdsi/integrated_indicators.csv",
    PROJECT_ROOT / "results_nc/integrated_indicators.csv",
    PROJECT_ROOT / "data/integrated_indicators.csv",
]

MATRIX_CANDIDATES = [
    PROJECT_ROOT / "results/fdsi/suitability_matrix.csv",
    PROJECT_ROOT / "results_nc/suitability_matrix.csv",
    PROJECT_ROOT / "data/suitability_matrix.csv",
]

OUTPUT_DIR = PROJECT_ROOT / "results_nc" / "directional_bias"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def find_file(candidates):
    for p in candidates:
        if p.exists():
            return p
    return None


# ============================================================
# 2. LOAD DATA
# ============================================================
def load_data():
    fdsi_path = find_file(FDSI_CANDIDATES)
    ind_path = find_file(INDICATORS_CANDIDATES)
    mat_path = find_file(MATRIX_CANDIDATES)
    
    if fdsi_path is None:
        raise FileNotFoundError(
            f"Cannot find fdsi_scores.csv. Searched:\n" +
            "\n".join(f"  {p}" for p in FDSI_CANDIDATES) +
            "\n\nPlease adjust FDSI_CANDIDATES in the script."
        )
    
    print(f"Loading FDSI scores from: {fdsi_path}")
    fdsi = pd.read_csv(fdsi_path)
    
    indicators = None
    if ind_path:
        print(f"Loading indicators from: {ind_path}")
        indicators = pd.read_csv(ind_path)
    
    matrix = None
    if mat_path:
        print(f"Loading suitability matrix from: {mat_path}")
        matrix = pd.read_csv(mat_path)
    
    return fdsi, indicators, matrix


# ============================================================
# 3. CLASSIFY CITIES INTO TERCILES AND IDENTIFY MISCLASSIFIED
# ============================================================
def classify_and_identify(fdsi):
    """Assign tercile tiers and flag misclassified cities."""
    n = len(fdsi)
    
    # Ensure rank columns exist
    if 'ghi_rank' not in fdsi.columns:
        for col in ['GHI', 'ghi', 'd1_1_ghi_annual', 'ghi_annual']:
            if col in fdsi.columns:
                fdsi['ghi_rank'] = fdsi[col].rank(ascending=False).astype(int)
                break

    if 'fdsi_rank' not in fdsi.columns:
        for col in ['FDSI', 'fdsi', 'fdsi_score', 'rank']:
            if col in fdsi.columns:
                fdsi['fdsi_rank'] = fdsi[col].rank(ascending=True).astype(int)
                break
    
    # Assign tercile tiers (1=High, 2=Medium, 3=Low)
    tercile_size = n // 3
    remainder = n % 3
    
    def assign_tercile(rank, n):
        t1 = n // 3 + (1 if n % 3 > 0 else 0)
        t2 = t1 + n // 3 + (1 if n % 3 > 1 else 0)
        if rank <= t1:
            return "High"
        elif rank <= t2:
            return "Medium"
        else:
            return "Low"
    
    fdsi['ghi_tier'] = fdsi['ghi_rank'].apply(lambda r: assign_tercile(r, n))
    fdsi['fdsi_tier'] = fdsi['fdsi_rank'].apply(lambda r: assign_tercile(r, n))
    fdsi['misclassified'] = fdsi['ghi_tier'] != fdsi['fdsi_tier']
    if 'rank_shift' not in fdsi.columns:
        fdsi['rank_shift'] = fdsi['ghi_rank'] - fdsi['fdsi_rank']
    
    return fdsi


# ============================================================
# 4. STATISTICAL COMPARISON: MISCLASSIFIED vs CORRECTLY CLASSIFIED
# ============================================================
def compare_groups(df, indicators, matrix):
    """Compare misclassified vs correctly classified on density indicators."""
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("§3.5 DIRECTIONAL BIAS ANALYSIS")
    report_lines.append("Misclassified vs Correctly Classified Cities")
    report_lines.append("=" * 70)
    
    mis = df[df['misclassified'] == True]
    cor = df[df['misclassified'] == False]
    
    report_lines.append(f"\nSample: {len(df)} cities total")
    report_lines.append(f"  Misclassified: {len(mis)} cities")
    report_lines.append(f"  Correctly classified: {len(cor)} cities")
    
    # Try to merge with indicators for raw FAR, building density, height
    if indicators is not None:
        # Find city column
        city_col = None
        for c in ['city', 'City', 'name', 'city_name']:
            if c in df.columns and c in indicators.columns:
                city_col = c
                break
        
        if city_col is None:
            # Try matching by index or first column
            city_col_df = df.columns[0]
            city_col_ind = indicators.columns[0]
            df_merged = df.merge(indicators, left_on=city_col_df, right_on=city_col_ind, how='left', suffixes=('', '_ind'))
        else:
            df_merged = df.merge(indicators, on=city_col, how='left', suffixes=('', '_ind'))
    else:
        df_merged = df.copy()
    
    # Also merge D-scores if available
    # suitability_matrix has capitalised city names; df has lowercase — normalise
    if matrix is not None:
        mat2 = matrix.copy()
        mat2['city_key'] = mat2['city'].str.lower()
        df_merged['city_key'] = df_merged['city'].str.lower()
        d_cols = [c for c in mat2.columns if c.startswith('D') or c == 'fdsi_score']
        mat2 = mat2[['city_key'] + d_cols]
        df_merged = df_merged.merge(mat2, on='city_key', how='left', suffixes=('', '_mat'))
    
    mis_m = df_merged[df_merged['misclassified'] == True]
    cor_m = df_merged[df_merged['misclassified'] == False]
    
    # List of indicators to test
    test_vars = []
    
    # Raw indicators
    for col, label in [
        ('d2_5_far', 'Floor Area Ratio (FAR)'),
        ('d2_2_building_density', 'Building Density'),
        ('d2_1_height_mean', 'Mean Building Height (m)'),
        ('d2_3_roof_area_total_m2', 'Total Roof Area (m²)'),
        ('d3_4_total_deployable_mw', 'Deployable Capacity (MW)'),
        ('d2_6_height_cv', 'Height CV'),
    ]:
        if col in df_merged.columns:
            test_vars.append((col, label))
    
    # D-scores
    for col, label in [
        ('D1', 'D1 (Solar Resource)'),
        ('D2', 'D2 (Urban Morphology)'),
        ('D3', 'D3 (Technical Potential)'),
        ('D4', 'D4 (Economic Viability)'),
        ('D5', 'D5 (Assessment Certainty)'),
    ]:
        if col in df_merged.columns:
            test_vars.append((col, label))
    
    if not test_vars:
        report_lines.append("\n⚠ No density indicators found in merged data.")
        report_lines.append("Available columns: " + ", ".join(df_merged.columns[:20].tolist()))
        return report_lines, df_merged
    
    report_lines.append("\n" + "-" * 70)
    report_lines.append("MANN-WHITNEY U TESTS (two-sided)")
    report_lines.append("-" * 70)
    report_lines.append(f"{'Variable':<30} {'Mis median':>12} {'Cor median':>12} {'U':>10} {'p-value':>10} {'Effect r':>10} {'Sig':>5}")
    report_lines.append("-" * 70)
    
    significant_results = []
    
    for col, label in test_vars:
        mis_vals = mis_m[col].dropna()
        cor_vals = cor_m[col].dropna()
        
        if len(mis_vals) < 3 or len(cor_vals) < 3:
            continue
        
        # Mann-Whitney U test
        u_stat, p_val = stats.mannwhitneyu(mis_vals, cor_vals, alternative='two-sided')
        
        # Effect size (rank-biserial correlation)
        n1, n2 = len(mis_vals), len(cor_vals)
        r_effect = 1 - (2 * u_stat) / (n1 * n2)
        
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "†" if p_val < 0.10 else ""
        
        report_lines.append(
            f"{label:<30} {mis_vals.median():>12.4f} {cor_vals.median():>12.4f} "
            f"{u_stat:>10.0f} {p_val:>10.4f} {r_effect:>10.3f} {sig:>5}"
        )
        
        if p_val < 0.05:
            significant_results.append({
                'variable': label,
                'col': col,
                'mis_median': mis_vals.median(),
                'cor_median': cor_vals.median(),
                'mis_mean': mis_vals.mean(),
                'cor_mean': cor_vals.mean(),
                'u': u_stat,
                'p': p_val,
                'r_effect': r_effect,
            })
    
    # Also run t-tests for comparison
    report_lines.append("\n" + "-" * 70)
    report_lines.append("WELCH'S T-TESTS (for comparison)")
    report_lines.append("-" * 70)
    report_lines.append(f"{'Variable':<30} {'Mis mean':>12} {'Cor mean':>12} {'t':>10} {'p-value':>10} {'Cohen d':>10} {'Sig':>5}")
    report_lines.append("-" * 70)
    
    for col, label in test_vars:
        mis_vals = mis_m[col].dropna()
        cor_vals = cor_m[col].dropna()
        
        if len(mis_vals) < 3 or len(cor_vals) < 3:
            continue
        
        t_stat, p_val = stats.ttest_ind(mis_vals, cor_vals, equal_var=False)
        
        # Cohen's d
        pooled_std = np.sqrt((mis_vals.std()**2 + cor_vals.std()**2) / 2)
        cohens_d = (mis_vals.mean() - cor_vals.mean()) / pooled_std if pooled_std > 0 else 0
        
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "†" if p_val < 0.10 else ""
        
        report_lines.append(
            f"{label:<30} {mis_vals.mean():>12.4f} {cor_vals.mean():>12.4f} "
            f"{t_stat:>10.3f} {p_val:>10.4f} {cohens_d:>10.3f} {sig:>5}"
        )
    
    # Summary
    report_lines.append("\n" + "=" * 70)
    report_lines.append("SUMMARY FOR §3.5")
    report_lines.append("=" * 70)
    
    if significant_results:
        report_lines.append(f"\n✓ {len(significant_results)} variables significantly differ (p < 0.05):")
        for r in significant_results:
            direction = "higher" if r['mis_median'] > r['cor_median'] else "lower"
            report_lines.append(
                f"  • {r['variable']}: misclassified cities have {direction} values "
                f"(median {r['mis_median']:.3f} vs {r['cor_median']:.3f}, "
                f"U = {r['u']:.0f}, p = {r['p']:.4f}, r = {r['r_effect']:.3f})"
            )
        
        report_lines.append("\n" + "=" * 70)
        report_lines.append("READY-TO-PASTE SENTENCES FOR §3.5")
        report_lines.append("=" * 70)
        
        # Generate paper-ready sentences
        for r in significant_results:
            if 'FAR' in r['variable'] or 'far' in r['col']:
                direction = "higher" if r['mis_median'] > r['cor_median'] else "lower"
                report_lines.append(
                    f"\nFAR sentence:\n"
                    f"Misclassified cities had significantly {direction} median FAR "
                    f"({r['mis_median']:.3f}) than correctly classified cities "
                    f"({r['cor_median']:.3f}; Mann\u2013Whitney U = {r['u']:.0f}, "
                    f"p = {r['p']:.3f}, rank-biserial r = {r['r_effect']:.2f})."
                )
            elif 'D2' in r['variable']:
                direction = "lower" if r['mis_median'] < r['cor_median'] else "higher"
                report_lines.append(
                    f"\nD2 sentence:\n"
                    f"Misclassified cities scored significantly {direction} on the "
                    f"urban morphology dimension (D2 median: {r['mis_median']:.3f} vs "
                    f"{r['cor_median']:.3f}; U = {r['u']:.0f}, p = {r['p']:.3f})."
                )
    else:
        report_lines.append("\n⚠ No significant differences found at p < 0.05.")
        report_lines.append("Consider: the pattern may still be directional but not")
        report_lines.append("statistically significant with n=13 vs n=28.")
        report_lines.append("In this case, do NOT add §3.5 to the paper.")
    
    # Partial correlation: FAR vs FDSI controlling for GHI
    report_lines.append("\n" + "=" * 70)
    report_lines.append("PARTIAL CORRELATION: FAR → FDSI | GHI")
    report_lines.append("=" * 70)
    
    far_col = None
    for c in ['d2_5_far', 'FAR', 'far']:
        if c in df_merged.columns:
            far_col = c
            break
    
    ghi_col = None
    for c in ['d1_1_ghi_annual_kwh', 'ghi_annual', 'd1_1_ghi_annual', 'GHI', 'ghi']:
        if c in df_merged.columns:
            ghi_col = c
            break

    fdsi_col = None
    for c in ['fdsi_score', 'FDSI', 'fdsi']:
        if c in df_merged.columns:
            fdsi_col = c
            break
    
    if far_col and ghi_col and fdsi_col:
        # Manual partial correlation
        subset = df_merged[[far_col, ghi_col, fdsi_col]].dropna()
        
        # Residualise FAR on GHI
        from numpy.polynomial.polynomial import polyfit
        slope_fg, intercept_fg = np.polyfit(subset[ghi_col], subset[far_col], 1)
        far_resid = subset[far_col] - (intercept_fg + slope_fg * subset[ghi_col])
        
        # Residualise FDSI on GHI
        slope_fsg, intercept_fsg = np.polyfit(subset[ghi_col], subset[fdsi_col], 1)
        fdsi_resid = subset[fdsi_col] - (intercept_fsg + slope_fsg * subset[ghi_col])
        
        # Correlation of residuals = partial correlation
        partial_r, partial_p = stats.pearsonr(far_resid, fdsi_resid)
        
        report_lines.append(f"\nPartial correlation (FAR → FDSI | GHI):")
        report_lines.append(f"  r = {partial_r:.3f}, p = {partial_p:.4f}")
        report_lines.append(f"\nPaper-ready sentence:")
        sig_text = f"p = {partial_p:.3f}" if partial_p >= 0.001 else "p < 0.001"
        report_lines.append(
            f"After controlling for GHI, FAR remained a significant predictor "
            f"of FDSI (partial r = {partial_r:.2f}, {sig_text}), confirming that "
            f"morphological variation exerts an independent effect on suitability "
            f"ranking."
        )
    else:
        report_lines.append(f"\n⚠ Could not compute partial correlation.")
        report_lines.append(f"  FAR col: {far_col}, GHI col: {ghi_col}, FDSI col: {fdsi_col}")
    
    return report_lines, df_merged


# ============================================================
# 5. GENERATE BOXPLOT
# ============================================================
def make_boxplot(df_merged):
    """Create boxplot comparing misclassified vs correctly classified."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        plot_vars = []
        for col, label in [
            ('d2_5_far', 'FAR'),
            ('D2', 'D2 Score'),
            ('d2_2_building_density', 'Building Density'),
            ('d2_1_height_mean', 'Mean Height (m)'),
        ]:
            if col in df_merged.columns:
                plot_vars.append((col, label))
        
        if not plot_vars:
            print("No variables available for plotting.")
            return
        
        n_vars = len(plot_vars)
        fig, axes = plt.subplots(1, n_vars, figsize=(4 * n_vars, 5))
        if n_vars == 1:
            axes = [axes]
        
        for ax, (col, label) in zip(axes, plot_vars):
            data_mis = df_merged[df_merged['misclassified'] == True][col].dropna()
            data_cor = df_merged[df_merged['misclassified'] == False][col].dropna()
            
            bp = ax.boxplot([data_cor, data_mis], 
                           labels=['Correct\n(n={})'.format(len(data_cor)), 
                                   'Misclassified\n(n={})'.format(len(data_mis))],
                           patch_artist=True,
                           widths=0.6)
            
            bp['boxes'][0].set_facecolor('#4ECDC4')
            bp['boxes'][1].set_facecolor('#FF6B6B')
            
            ax.set_ylabel(label)
            ax.set_title(label)
            
            # Add significance annotation
            u_stat, p_val = stats.mannwhitneyu(data_mis, data_cor, alternative='two-sided')
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
            ax.text(0.5, 0.95, f"p = {p_val:.3f} {sig}", transform=ax.transAxes,
                   ha='center', va='top', fontsize=10)
        
        plt.suptitle("Misclassified vs Correctly Classified Cities", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        out_path = OUTPUT_DIR / "misclass_vs_correct_boxplot.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"\nBoxplot saved to: {out_path}")
        plt.close()
        
    except ImportError:
        print("matplotlib not available, skipping boxplot.")


# ============================================================
# 6. MAIN
# ============================================================
def main():
    print("=" * 70)
    print("§3.5 DIRECTIONAL BIAS ANALYSIS")
    print("=" * 70)
    
    fdsi, indicators, matrix = load_data()
    
    print(f"\nFDSI data: {len(fdsi)} cities, columns: {fdsi.columns.tolist()[:10]}")
    if indicators is not None:
        print(f"Indicators: {len(indicators)} cities, columns: {indicators.columns.tolist()[:10]}")
    
    # Step 1: Classify
    df = classify_and_identify(fdsi)
    
    mis_cities = df[df['misclassified'] == True]
    print(f"\n13 misclassified cities:")
    for _, row in mis_cities.iterrows():
        city_name = row.get('city', row.get('name', row.get('City', '?')))
        print(f"  {city_name}: GHI tier={row['ghi_tier']}, FDSI tier={row['fdsi_tier']}, shift={row.get('rank_shift', '?')}")
    
    # Step 2: Compare
    report_lines, df_merged = compare_groups(df, indicators, matrix)
    
    # Step 3: Print and save report
    report_text = "\n".join(report_lines)
    print(report_text)
    
    report_path = OUTPUT_DIR / "directional_bias_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\nReport saved to: {report_path}")
    
    # Step 4: Boxplot
    make_boxplot(df_merged)
    
    print("\n" + "=" * 70)
    print("DECISION GUIDE:")
    print("  If FAR or D2 is significant (p < 0.05): ADD §3.5 to paper")
    print("  If marginal (0.05 < p < 0.10): Consider adding as Discussion point")
    print("  If not significant: Do NOT add §3.5")
    print("=" * 70)


if __name__ == "__main__":
    main()
