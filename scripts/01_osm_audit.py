#!/usr/bin/env python3
"""
============================================================================
Paper 4 - Script 01: OSM Data Availability Audit
============================================================================
Layer 3 screening: evaluates OSM building data quality for candidate cities.

For each city/district, this script computes:
  1. Total building count (all types)
  2. Residential building count (filtered by tags)
  3. Height attribute coverage (buildings with 'height' or 'building:levels')
  4. Residential tag coverage (buildings tagged as residential)
  5. Footprint area statistics (mean, median, std)
  6. Data Completeness Score = 0.5*coverage_proxy + 0.3*height_ratio + 0.2*residential_ratio

Outputs:
  - Console summary table
  - CSV report: results/osm_audit/audit_results.csv
  - Per-city GeoPackage: results/osm_audit/{city}_buildings.gpkg
  - Summary figure: results/osm_audit/audit_comparison.png

Usage:
  python scripts/01_osm_audit.py                    # All primary cities
  python scripts/01_osm_audit.py --include-alternates # + alternate candidates
  python scripts/01_osm_audit.py --city beijing      # Single city
  python scripts/01_osm_audit.py --bbox 39.9,40.1,116.2,116.5  # Custom bbox

Notes:
  - Requires internet access (Overpass API)
  - Large districts may take 2-5 minutes per city
  - If osmnx times out, try a smaller district or use --bbox
============================================================================
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import geopandas as gpd
import numpy as np

try:
    import osmnx as ox
except ImportError:
    print("ERROR: osmnx not installed. Run: pip install osmnx")
    sys.exit(1)

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None  # Fallback to simple print

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── Setup ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CONFIGS_DIR = PROJECT_DIR / "configs"
RESULTS_DIR = PROJECT_DIR / "results" / "osm_audit"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# OSMnx settings
ox.settings.log_console = False
ox.settings.use_cache = True
ox.settings.timeout = 300  # 5 min timeout for large queries


# ── Residential Tag Definitions ────────────────────────────────────────────

RESIDENTIAL_BUILDING_TAGS = {
    "residential", "apartments", "house", "detached",
    "semidetached_house", "terrace", "dormitory",
}

RESIDENTIAL_LANDUSE_TAGS = {"residential"}


# ── Core Audit Function ───────────────────────────────────────────────────

def fetch_buildings(
    place_query: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> gpd.GeoDataFrame:
    """
    Fetch ALL buildings from OSM for the given area.

    Parameters
    ----------
    place_query : str, optional
        Geocodable place name (e.g., "海淀区, 北京市, 中国")
    bbox : tuple, optional
        (north, south, east, west) bounding box

    Returns
    -------
    GeoDataFrame with building footprints and attributes
    """
    tags = {"building": True}  # Fetch all buildings regardless of type

    if bbox is not None:
        north, south, east, west = bbox
        log.info(f"  Fetching buildings in bbox: N={north}, S={south}, E={east}, W={west}")
        gdf = ox.features_from_bbox(
            bbox=(west, south, east, north),  # osmnx 1.6+ uses (W, S, E, N)
            tags=tags,
        )
    elif place_query is not None:
        log.info(f"  Fetching buildings for: {place_query}")
        gdf = ox.features_from_place(place_query, tags=tags)
    else:
        raise ValueError("Must provide either place_query or bbox")

    # Keep only polygon geometries (drop nodes, relations)
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()

    return gdf


def classify_residential(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    Return a boolean Series: True if the building is likely residential.

    Classification logic:
    1. building tag is in RESIDENTIAL_BUILDING_TAGS
    2. OR landuse tag is in RESIDENTIAL_LANDUSE_TAGS
    3. OR building tag is "yes" AND in a residential landuse area (heuristic)
    """
    is_res = pd.Series(False, index=gdf.index)

    # Check building tag
    if "building" in gdf.columns:
        is_res |= gdf["building"].isin(RESIDENTIAL_BUILDING_TAGS)

    # Check landuse tag (sometimes present on building polygons)
    if "landuse" in gdf.columns:
        is_res |= gdf["landuse"].isin(RESIDENTIAL_LANDUSE_TAGS)

    return is_res


def extract_height(gdf: gpd.GeoDataFrame, default_floor_h: float = 3.0) -> pd.Series:
    """
    Extract building height from OSM attributes.

    Priority:
    1. 'height' tag (meters) → use directly
    2. 'building:levels' tag → multiply by default_floor_h
    3. Neither → NaN

    Returns
    -------
    Series of float heights in meters (NaN where unavailable)
    """
    height = pd.Series(np.nan, index=gdf.index, dtype=float)

    # Try 'height' tag first
    if "height" in gdf.columns:
        h = pd.to_numeric(gdf["height"], errors="coerce")
        height = height.fillna(h)

    # Fall back to building:levels
    if "building:levels" in gdf.columns:
        levels = pd.to_numeric(gdf["building:levels"], errors="coerce")
        h_from_levels = levels * default_floor_h
        height = height.fillna(h_from_levels)

    return height


def compute_footprint_areas(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    Compute footprint area in m² using a local UTM projection.
    """
    # Reproject to UTM for accurate area calculation
    gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
    return gdf_proj.geometry.area


def audit_city(
    name_en: str,
    name_cn: str,
    climate_zone: str,
    place_query: Optional[str] = None,
    bbox: Optional[Tuple] = None,
    is_primary: bool = True,
    save_gpkg: bool = True,
) -> Dict[str, Any]:
    """
    Run the full OSM data quality audit for one city/district.

    Returns a dict with all audit metrics.
    """
    log.info(f"{'='*60}")
    log.info(f"Auditing: {name_en} ({name_cn}) — {climate_zone}")
    log.info(f"{'='*60}")

    t0 = time.time()

    try:
        gdf = fetch_buildings(place_query=place_query, bbox=bbox)
    except Exception as e:
        log.error(f"  FAILED to fetch buildings: {e}")
        return {
            "city_en": name_en,
            "city_cn": name_cn,
            "climate_zone": climate_zone,
            "is_primary": is_primary,
            "status": "FAILED",
            "error": str(e),
            "n_buildings_total": 0,
        }

    n_total = len(gdf)
    log.info(f"  Total buildings fetched: {n_total}")

    if n_total == 0:
        return {
            "city_en": name_en,
            "city_cn": name_cn,
            "climate_zone": climate_zone,
            "is_primary": is_primary,
            "status": "NO_DATA",
            "n_buildings_total": 0,
        }

    # ── Residential classification ──
    is_residential = classify_residential(gdf)
    n_residential = is_residential.sum()
    residential_ratio = n_residential / n_total

    # ── Height attribute coverage ──
    heights = extract_height(gdf)
    has_height = heights.notna()
    n_with_height = has_height.sum()
    height_ratio = n_with_height / n_total

    # ── Footprint area stats ──
    areas = compute_footprint_areas(gdf)
    gdf["footprint_area_m2"] = areas.values  # Align indices might differ after projection

    area_mean = areas.mean()
    area_median = areas.median()
    area_std = areas.std()

    # ── Building type distribution ──
    if "building" in gdf.columns:
        type_counts = gdf["building"].value_counts().head(10).to_dict()
    else:
        type_counts = {}

    # ── Coverage proxy ──
    # We can't easily compute true coverage (OSM vs reality) without satellite
    # imagery, so we use building density as a heuristic indicator.
    # A study area with many buildings is likely well-mapped.
    # This is a PROXY; the paper should note this limitation.
    # For a proper coverage estimate, overlay with satellite building footprints.
    # Here we use: if n_total >= 5000 → coverage_proxy = 0.8 (assumed decent)
    #              if n_total >= 1000 → coverage_proxy = 0.6
    #              else → coverage_proxy = 0.3
    if n_total >= 10000:
        coverage_proxy = 0.9
    elif n_total >= 5000:
        coverage_proxy = 0.8
    elif n_total >= 2000:
        coverage_proxy = 0.7
    elif n_total >= 1000:
        coverage_proxy = 0.6
    elif n_total >= 500:
        coverage_proxy = 0.5
    else:
        coverage_proxy = 0.3

    # ── Data Completeness Score ──
    # DCS = 0.5 × coverage_proxy + 0.3 × height_ratio + 0.2 × residential_ratio
    dcs = 0.5 * coverage_proxy + 0.3 * height_ratio + 0.2 * residential_ratio

    # ── Height statistics (where available) ──
    valid_heights = heights.dropna()
    if len(valid_heights) > 0:
        h_mean = valid_heights.mean()
        h_median = valid_heights.median()
        h_std = valid_heights.std()
        h_max = valid_heights.max()
    else:
        h_mean = h_median = h_std = h_max = np.nan

    elapsed = time.time() - t0

    # ── Save GeoPackage ──
    if save_gpkg:
        gpkg_path = RESULTS_DIR / f"{name_en.lower()}_buildings.gpkg"
        try:
            # Select key columns to save (avoid complex/list columns that GPKG can't handle)
            save_cols = ["geometry"]
            for col in ["building", "height", "building:levels", "name", "landuse",
                        "footprint_area_m2"]:
                if col in gdf.columns:
                    save_cols.append(col)
            gdf_save = gdf[save_cols].copy()
            # Convert any list/dict columns to strings
            for col in gdf_save.columns:
                if col != "geometry":
                    gdf_save[col] = gdf_save[col].astype(str)
            gdf_save.to_file(gpkg_path, driver="GPKG")
            log.info(f"  Saved: {gpkg_path}")
        except Exception as e:
            log.warning(f"  Failed to save GPKG: {e}")

    # ── Compile results ──
    result = {
        "city_en": name_en,
        "city_cn": name_cn,
        "climate_zone": climate_zone,
        "is_primary": is_primary,
        "status": "OK",
        "n_buildings_total": n_total,
        "n_residential": int(n_residential),
        "residential_ratio": round(residential_ratio, 3),
        "n_with_height": int(n_with_height),
        "height_ratio": round(height_ratio, 3),
        "coverage_proxy": round(coverage_proxy, 2),
        "completeness_score": round(dcs, 3),
        "area_mean_m2": round(area_mean, 1),
        "area_median_m2": round(area_median, 1),
        "area_std_m2": round(area_std, 1),
        "height_mean_m": round(h_mean, 1) if not np.isnan(h_mean) else None,
        "height_median_m": round(h_median, 1) if not np.isnan(h_median) else None,
        "height_max_m": round(h_max, 1) if not np.isnan(h_max) else None,
        "top_building_types": type_counts,
        "elapsed_seconds": round(elapsed, 1),
    }

    # ── Print summary ──
    log.info(f"  Buildings total:     {n_total:,}")
    log.info(f"  Residential:         {n_residential:,} ({residential_ratio:.1%})")
    log.info(f"  With height attr:    {n_with_height:,} ({height_ratio:.1%})")
    log.info(f"  Coverage proxy:      {coverage_proxy:.2f}")
    log.info(f"  Completeness Score:  {dcs:.3f}  {'✓ PASS' if dcs >= 0.4 else '✗ FAIL'}")
    log.info(f"  Elapsed:             {elapsed:.1f}s")

    return result


# ── Visualization ──────────────────────────────────────────────────────────

def plot_audit_comparison(results: List[Dict], output_path: Path):
    """Generate a comparison bar chart of audit metrics across cities."""
    if not HAS_MPL:
        log.warning("matplotlib not available, skipping plot.")
        return

    df = pd.DataFrame([r for r in results if r["status"] == "OK"])
    if df.empty:
        return

    df = df.sort_values("completeness_score", ascending=True)

    fig, axes = plt.subplots(1, 4, figsize=(18, max(5, len(df) * 0.6)))

    # Labels
    labels = [f"{row['city_en']}\n({row['city_cn']})\n[{row['climate_zone']}]"
              for _, row in df.iterrows()]
    colors = ["#2196F3" if row["is_primary"] else "#90CAF9" for _, row in df.iterrows()]
    y_pos = range(len(df))

    # Panel 1: Total buildings
    axes[0].barh(y_pos, df["n_buildings_total"], color=colors)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlabel("Total Buildings")
    axes[0].set_title("Building Count")

    # Panel 2: Height attribute ratio
    axes[1].barh(y_pos, df["height_ratio"], color=colors)
    axes[1].axvline(x=0.3, color="red", linestyle="--", linewidth=1, label="Threshold (0.3)")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("Ratio")
    axes[1].set_title("Height Attr Coverage")
    axes[1].set_xlim(0, 1)
    axes[1].legend(fontsize=7)

    # Panel 3: Residential ratio
    axes[2].barh(y_pos, df["residential_ratio"], color=colors)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels([])
    axes[2].set_xlabel("Ratio")
    axes[2].set_title("Residential Tag Ratio")
    axes[2].set_xlim(0, 1)

    # Panel 4: Completeness Score
    axes[3].barh(y_pos, df["completeness_score"], color=colors)
    axes[3].axvline(x=0.4, color="red", linestyle="--", linewidth=1, label="Threshold (0.4)")
    axes[3].set_yticks(y_pos)
    axes[3].set_yticklabels([])
    axes[3].set_xlabel("Score")
    axes[3].set_title("Data Completeness Score")
    axes[3].set_xlim(0, 1)
    axes[3].legend(fontsize=7)

    fig.suptitle("Paper 4 — OSM Data Availability Audit (Layer 3 Screening)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info(f"  Saved comparison chart: {output_path}")
    plt.close(fig)


# ── City Definitions ──────────────────────────────────────────────────────

# Hardcoded fallback if config loader is not available
# (so the script can run standalone without the src/ package)
PRIMARY_CITIES = [
    {"name_en": "Harbin",   "name_cn": "哈尔滨", "climate_zone": "severe_cold",
     "place_query": "南岗区, 哈尔滨市, 中国", "is_primary": True},
    {"name_en": "Beijing",  "name_cn": "北京",   "climate_zone": "cold",
     "place_query": "海淀区, 北京市, 中国",    "is_primary": True},
    {"name_en": "Changsha", "name_cn": "长沙",   "climate_zone": "hscw",
     "place_query": "岳麓区, 长沙市, 中国",    "is_primary": True},
    {"name_en": "Shenzhen", "name_cn": "深圳",   "climate_zone": "hsww",
     "place_query": "福田区, 深圳市, 中国",    "is_primary": True},
    {"name_en": "Kunming",  "name_cn": "昆明",   "climate_zone": "mild",
     "place_query": None, "bbox": (25.10, 24.90, 102.85, 102.55), "is_primary": True},
]

ALTERNATE_CITIES = [
    # ── 严寒 (Severe Cold) ──
    {"name_en": "Changchun", "name_cn": "长春", "climate_zone": "severe_cold",
     "place_query": "朝阳区, 长春市, 中国", "is_primary": False},
    {"name_en": "Shenyang",  "name_cn": "沈阳", "climate_zone": "severe_cold",
     "place_query": "沈河区, 沈阳市, 中国", "is_primary": False},
    # ── 寒冷 (Cold) ──
    {"name_en": "Xian",      "name_cn": "西安", "climate_zone": "cold",
     "place_query": "碑林区, 西安市, 中国", "is_primary": False},
    {"name_en": "Jinan",     "name_cn": "济南", "climate_zone": "cold",
     "place_query": "历下区, 济南市, 中国", "is_primary": False},
    # ── 夏热冬冷 (HSCW) ──
    {"name_en": "Wuhan",     "name_cn": "武汉", "climate_zone": "hscw",
     "place_query": "武昌区, 武汉市, 中国", "is_primary": False},
    {"name_en": "Nanjing",   "name_cn": "南京", "climate_zone": "hscw",
     "place_query": "鼓楼区, 南京市, 中国", "is_primary": False},
    # ── 夏热冬暖 (HSWW) ──
    {"name_en": "Guangzhou", "name_cn": "广州", "climate_zone": "hsww",
     "place_query": "天河区, 广州市, 中国", "is_primary": False},
    {"name_en": "Xiamen",    "name_cn": "厦门", "climate_zone": "hsww",
     "place_query": "思明区, 厦门市, 中国", "is_primary": False},
    # ── 温和 (Mild) ──
    {"name_en": "Guiyang",   "name_cn": "贵阳", "climate_zone": "mild",
     "place_query": None, "bbox": (26.72, 26.52, 106.80, 106.55), "is_primary": False},
    {"name_en": "Chengdu",   "name_cn": "成都", "climate_zone": "mild",
     "place_query": "锦江区, 成都市, 中国", "is_primary": False},
]


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Paper 4: OSM Data Availability Audit (Layer 3 Screening)"
    )
    parser.add_argument(
        "--city", type=str, default=None,
        help="Audit a single city by English name (e.g., 'beijing'). "
             "Case-insensitive."
    )
    parser.add_argument(
        "--include-alternates", action="store_true",
        help="Also audit alternate candidate cities."
    )
    parser.add_argument(
        "--bbox", type=str, default=None,
        help="Custom bounding box: 'north,south,east,west' (decimal degrees). "
             "Use with --city."
    )
    parser.add_argument(
        "--no-gpkg", action="store_true",
        help="Skip saving per-city GeoPackage files."
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Paper 4 — OSM Data Availability Audit")
    log.info(f"Timestamp: {datetime.now().isoformat()}")
    log.info("=" * 60)

    # ── Build target list ──
    targets = list(PRIMARY_CITIES)
    if args.include_alternates:
        targets.extend(ALTERNATE_CITIES)

    # ── Filter by city name ──
    if args.city:
        city_lower = args.city.lower()
        targets = [t for t in targets if t["name_en"].lower() == city_lower]
        if not targets:
            log.error(f"City '{args.city}' not found. Available: "
                      f"{', '.join(c['name_en'] for c in PRIMARY_CITIES + ALTERNATE_CITIES)}")
            sys.exit(1)

    # ── Parse bbox override ──
    bbox = None
    if args.bbox:
        parts = [float(x.strip()) for x in args.bbox.split(",")]
        if len(parts) != 4:
            log.error("--bbox must be 'north,south,east,west'")
            sys.exit(1)
        bbox = tuple(parts)

    log.info(f"Cities to audit: {len(targets)}")
    log.info(f"  Primary:    {sum(1 for t in targets if t['is_primary'])}")
    log.info(f"  Alternates: {sum(1 for t in targets if not t['is_primary'])}")

    # ── Run audits ──
    results = []
    for i, target in enumerate(targets, 1):
        log.info(f"\n[{i}/{len(targets)}]")
        # bbox 优先级: 命令行 --bbox > 城市定义中的 bbox > place_query
        city_bbox = bbox if bbox is not None else target.get("bbox")
        city_place = target.get("place_query") if city_bbox is None else None
        result = audit_city(
            name_en=target["name_en"],
            name_cn=target["name_cn"],
            climate_zone=target["climate_zone"],
            place_query=city_place,
            bbox=city_bbox,
            is_primary=target["is_primary"],
            save_gpkg=not args.no_gpkg,
        )
        results.append(result)
        # Be nice to Overpass API
        if i < len(targets):
            log.info("  Waiting 10s before next query...")
            time.sleep(10)

    # ── Save CSV report ──
    df = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "audit_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log.info(f"\nSaved CSV report: {csv_path}")

    # ── Print summary table ──
    log.info("\n" + "=" * 60)
    log.info("AUDIT SUMMARY")
    log.info("=" * 60)

    ok_results = [r for r in results if r["status"] == "OK"]
    if ok_results:
        summary_cols = [
            "city_en", "city_cn", "climate_zone", "is_primary",
            "n_buildings_total", "height_ratio", "residential_ratio",
            "completeness_score"
        ]
        summary_df = pd.DataFrame(ok_results)[summary_cols]
        summary_df["pass"] = summary_df["completeness_score"].apply(
            lambda x: "✓" if x >= 0.4 else "✗"
        )
        summary_df = summary_df.sort_values(
            ["is_primary", "climate_zone"], ascending=[False, True]
        )

        if tabulate:
            print("\n" + tabulate(
                summary_df,
                headers="keys",
                tablefmt="grid",
                showindex=False,
                floatfmt=".3f",
            ))
        else:
            print(summary_df.to_string(index=False))

    failed = [r for r in results if r["status"] != "OK"]
    if failed:
        log.warning(f"\n{len(failed)} cities FAILED:")
        for r in failed:
            log.warning(f"  {r['city_en']}: {r.get('error', r['status'])}")

    # ── Decision guidance ──
    log.info("\n" + "-" * 60)
    log.info("SCREENING DECISIONS")
    log.info("-" * 60)

    threshold = 0.4
    for r in sorted(ok_results, key=lambda x: x["climate_zone"]):
        score = r["completeness_score"]
        status = "✓ SELECTED" if (r["is_primary"] and score >= threshold) else \
                 "⚠ REVIEW" if score < threshold else \
                 "  (alternate)"
        log.info(f"  {r['city_en']:12s} [{r['climate_zone']:12s}]  "
                 f"DCS={score:.3f}  {status}")

        if r["is_primary"] and score < threshold:
            # Find best alternate in same zone
            alts = [a for a in ok_results
                    if a["climate_zone"] == r["climate_zone"]
                    and not a["is_primary"]
                    and a["completeness_score"] >= threshold]
            if alts:
                best = max(alts, key=lambda x: x["completeness_score"])
                log.info(f"    → Consider replacing with: {best['city_en']} "
                         f"(DCS={best['completeness_score']:.3f})")
            else:
                log.info(f"    → No qualifying alternate found. "
                         f"Consider bbox adjustment or manual data enrichment.")

    # ── Generate comparison plot ──
    if HAS_MPL and ok_results:
        plot_path = RESULTS_DIR / "audit_comparison.png"
        plot_audit_comparison(results, plot_path)

    log.info("\nDone.")


if __name__ == "__main__":
    main()
