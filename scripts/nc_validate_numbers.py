"""
nc_validate_numbers.py
Cross-check every key number cited in the paper against CSV/JSON source data.
Prints PASS / FAIL for each claim.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent

# ── load data ─────────────────────────────────────────────────────────────────
sm   = pd.read_csv(ROOT / "results/fdsi/suitability_matrix.csv")
ii   = pd.read_csv(ROOT / "results/fdsi/integrated_indicators.csv")
rs   = pd.read_csv(ROOT / "results_nc/misclassification/rank_shift_analysis.csv")
oc   = pd.read_csv(ROOT / "results_nc/policy_cost/opportunity_cost_physical.csv")
sc   = pd.read_csv(ROOT / "results/scenarios/scenario_fdsi_matrix.csv")
rob  = pd.read_csv(ROOT / "results_nc/robustness/robustness_summary.csv")
sens = pd.read_csv(ROOT / "results_nc/sensitivity/classification_sensitivity.csv")

with open(ROOT / "results_nc/misclassification/misclassification_summary.json") as f:
    mj = json.load(f)
with open(ROOT / "results_nc/policy_cost/opportunity_cost_summary.json") as f:
    oj = json.load(f)

# ── helpers ───────────────────────────────────────────────────────────────────
_results = []

def check(label: str, expected, actual, tol=None):
    """tol: relative tolerance (0.001 = 0.1%); None for exact equality."""
    if tol is None:
        ok = expected == actual
    else:
        ok = abs(actual - expected) / (abs(expected) + 1e-12) <= tol
    mark = "PASS" if ok else "FAIL"
    _results.append((mark, label, expected, actual))
    status = f"[{mark}]"
    if not ok:
        print(f"  {status}  {label}")
        print(f"          expected={expected!r}  actual={actual!r}")
    else:
        print(f"  {status}  {label}")

def _spearman(x, y):
    r, p = stats.spearmanr(x, y)
    return round(r, 4), round(p, 4)

# ══════════════════════════════════════════════════════════════════════════════
print("\n══ §1  Sample size ══════════════════════════════════════════════")
check("n_cities = 41", 41, len(rs))
check("n_cities in opp_cost_summary", 41, oj["n_cities"])
check("n_cities in misclass_summary", 41, mj["n_cities"])

print("\n══ §1  Spearman r_s (GHI rank vs FDSI rank) ════════════════════")
r_s, p_s = _spearman(rs["ghi_rank"], rs["fdsi_rank"])
check("r_s = 0.7601", 0.7601, r_s, tol=0.001)
check("p < 0.001",    True,   p_s < 0.001)

print("\n══ §1  Unexplained rank variance ════════════════════════════════")
unexplained = round((1 - r_s**2) * 100, 1)
check("42.2% unexplained", 42.2, unexplained, tol=0.005)

print("\n══ §1  Tercile misclassification ════════════════════════════════")
# re-derive from raw data
rs["ghi_t"] = pd.qcut(rs["ghi_rank"],  q=3, labels=False, duplicates="drop")
rs["fdsi_t"] = pd.qcut(rs["fdsi_rank"], q=3, labels=False, duplicates="drop")
n_correct = (rs["ghi_t"] == rs["fdsi_t"]).sum()
n_misclass = len(rs) - n_correct
pct_mis = round(n_misclass / len(rs) * 100, 1)
check("n_misclassified = 13", 13, int(n_misclass))
check("pct_misclass = 31.7%", 31.7, pct_mis, tol=0.005)
check("n_correct = 28",       28, int(n_correct))
# cross-check against JSON
check("mj n_misclass_tercile = 13", 13, mj["n_misclass_tercile"])
check("mj pct_misclass_tercile = 31.7", 31.7, mj["pct_misclass_tercile"], tol=0.005)

print("\n══ §1  Rank shift statistics ════════════════════════════════════")
abs_shift  = rs["abs_rank_shift"]
check("median |shift| = 5.0",  5.0, float(abs_shift.median()))
check("max |shift| = 21",      21,  int(abs_shift.max()))
check("max-shift city = Hong Kong", "Hong Kong",
      rs.loc[abs_shift.idxmax(), "name_en"])
# Urumqi: confirm actual value
urumqi_shift = int(rs.loc[rs.city=="urumqi", "rank_shift"].values[0])
check("Urumqi actual shift = -17 (NOT -22)", -17, urumqi_shift)
n_ge5 = (abs_shift >= 5).sum()
check("n |shift|>=5 = 21",     21, int(n_ge5))
check("51.2% |shift|>=5",      51.2, round(n_ge5/len(rs)*100, 1), tol=0.005)

print("\n══ §2  Robustness — r_s vs GHI ══════════════════════════════════")
rob_orig = rob[rob.scheme == "original"].iloc[0]
check("original rs_vs_ghi = 0.7601", 0.7601,
      round(float(rob_orig["rs_vs_ghi"]), 4), tol=0.001)
for scheme, expected_rs in [("equal_weight", 0.2010),
                              ("no_D5",        0.1683),
                              ("no_D4",        0.2226),
                              ("pca_composite",0.0652)]:
    row = rob[rob.scheme == scheme]
    if len(row):
        val = round(float(row.iloc[0]["rs_vs_ghi"]), 4)
        check(f"{scheme} rs_vs_ghi = {expected_rs}", expected_rs, val, tol=0.005)

print("\n══ §2  Persistent misclassification cities ═══════════════════════")
# From robustness summary — check field exists
if "persistent_misclass_cities" in rob_orig:
    check("6 persistent misclass cities", 6,
          len(rob_orig["persistent_misclass_cities"].split(",")))
else:
    print("  [INFO]  persistent_misclass_cities not in robustness_summary.csv — check robustness_report.json")

print("\n══ §3  Cross-pair: Changsha vs Chengdu ══════════════════════════")
csa = rs[rs.city == "changsha"].iloc[0]
cdu = rs[rs.city == "chengdu"].iloc[0]
csa_ghi = float(ii[ii.city=="changsha"]["d1_1_ghi_annual_kwh"].values[0])
cdu_ghi = float(ii[ii.city=="chengdu"]["d1_1_ghi_annual_kwh"].values[0])
delta_ghi  = round(csa_ghi - cdu_ghi, 0)
delta_rank = int(cdu["fdsi_rank"] - csa["fdsi_rank"])
check("Changsha GHI = 1378", 1378, int(round(csa_ghi)))
check("Chengdu GHI  = 1292", 1292, int(round(cdu_ghi)))
check("ΔGHI = 86 kWh/m²/yr",  86, int(delta_ghi))
check("Δrank = 22",            22, delta_rank)

print("\n══ §3  Cross-pair: Shenzhen vs Hong Kong ════════════════════════")
szn_ghi = float(ii[ii.city=="shenzhen"]["d1_1_ghi_annual_kwh"].values[0])
hk_ghi  = float(ii[ii.city=="hongkong"]["d1_1_ghi_annual_kwh"].values[0])
szn_sm  = sm[sm.city=="Shenzhen"].iloc[0]
hk_sm   = sm[sm.city=="Hong Kong"].iloc[0]
check("Shenzhen FDSI rank = 14", 14, int(szn_sm["fdsi_rank"]))
check("HK FDSI rank = 32",       32, int(hk_sm["fdsi_rank"]))
check("HK D2 < 0.07",  True, float(hk_sm["D2_score"]) < 0.07)
check("HK D4 > 0.80",  True, float(hk_sm["D4_score"]) > 0.80)
hk_far = float(ii[ii.city=="hongkong"]["d2_5_far"].values[0])
check("HK FAR > 1.5",  True, hk_far > 1.5)

print("\n══ §4  Opportunity cost — physical quantities ════════════════════")
missed_names = {"Beijing", "Xian", "Wuhan", "Guangzhou"}
oc_m = oc[oc["name_en"].isin(missed_names)]
check("n missed cities = 4", 4, len(oc_m))
check("missed capacity ≈ 1552 MW",     1552, int(round(oc_m["capacity_mw"].sum())), )
miss_gen = oc_m["generation_gwh_yr"].sum()
check("missed generation ≈ 2044 GWh/yr", 2044, int(round(miss_gen)), )
miss_co2 = oc_m["co2_reduction_kt_yr"].sum()
check("missed CO₂ ≈ 1124 ktCO₂/yr",   1124, int(round(miss_co2)), )
miss_pop = oc_m["population_million"].sum()
check("missed population = 67 M",       67,  int(round(miss_pop)))
# from JSON
check("oj missed_capacity_mw = 1552",  1552, oj["targeting_top_third"]["missed_capacity_mw"])
check("oj missed_generation_gwh = 2043.5", 2043.5,
      oj["targeting_top_third"]["missed_generation_gwh"], tol=0.005)
check("oj missed_co2_kt = 1123.9",    1123.9,
      oj["targeting_top_third"]["missed_co2_kt"], tol=0.005)
check("oj missed_population_million = 66.9", 66.9,
      oj["targeting_top_third"]["missed_population_million"], tol=0.02)

print("\n══ §4  Sensitivity analysis ══════════════════════════════════════")
t = sens[sens.scheme == "tercile"].iloc[0]
q = sens[sens.scheme == "quartile"].iloc[0]
p = sens[sens.scheme == "quintile"].iloc[0]
check("tercile misclass = 31.7%",   31.7, float(t["pct_misclassified"]), tol=0.005)
check("quartile misclass = 46.3%",  46.3, float(q["pct_misclassified"]), tol=0.005)
check("quintile misclass = 51.2%",  51.2, float(p["pct_misclassified"]), tol=0.005)

print("\n══ §4  Scenario distribution (41 cities) ════════════════════════")
for sid, slabel, exp_h, exp_m, exp_l in [
    ("baseline",       "Baseline",       14, 13, 14),
    ("carbon_pricing", "Carbon pricing", 16, 13, 12),
    ("cost_reduction", "PV cost -50%",   37,  3,  1),
    ("aggressive",     "Aggressive",     39,  1,  1),
]:
    grp = sc[sc["scenario"] == sid]
    n_h = (grp["suitability"] == "High").sum()
    n_m = (grp["suitability"] == "Medium").sum()
    n_l = (grp["suitability"] == "Low").sum()
    check(f"{slabel}: High={exp_h}",   exp_h, int(n_h))
    check(f"{slabel}: Medium={exp_m}", exp_m, int(n_m))
    check(f"{slabel}: Low={exp_l}",    exp_l, int(n_l))

# ── summary ───────────────────────────────────────────────────────────────────
print("\n" + "═"*60)
n_pass = sum(1 for r in _results if r[0] == "PASS")
n_fail = sum(1 for r in _results if r[0] == "FAIL")
print(f"TOTAL: {n_pass} PASS  |  {n_fail} FAIL  (out of {len(_results)})")
if n_fail:
    print("\n  FAILed checks:")
    for mark, label, exp, act in _results:
        if mark == "FAIL":
            print(f"    - {label}")
            print(f"      expected: {exp!r}")
            print(f"      actual:   {act!r}")
print("═"*60)
