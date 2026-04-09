#!/usr/bin/env python3
"""
============================================================================
Paper 4 - Script 02: PVGIS Data Download
============================================================================
Downloads TMY and multi-year hourly solar irradiation data from the
PVGIS API (v5.3) for all confirmed study cities.

Outputs per city:
  - results/pvgis/{city}_tmy.csv          (Typical Meteorological Year)
  - results/pvgis/{city}_hourly.csv       (Multi-year hourly series)
  - results/pvgis/{city}_meta.json        (API response metadata)

Usage:
  python scripts/02_pvgis_download.py              # All primary cities
  python scripts/02_pvgis_download.py --city beijing  # Single city
  python scripts/02_pvgis_download.py --tmy-only      # Skip hourly series

Notes:
  - PVGIS coverage for China uses ERA5 reanalysis (~30 km resolution)
  - Hourly data download can be slow (15+ years of data)
  - Rate limit: be respectful, wait between requests
============================================================================
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests

# ── Setup ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results" / "pvgis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PVGIS_BASE = "https://re.jrc.ec.europa.eu/api/v5_3"

# City coordinates (from configs)
CITIES = {
    # ── Original 5 cities ──
    "harbin":    {"lat": 45.75, "lon": 126.65, "name": "Harbin"},
    "beijing":   {"lat": 39.90, "lon": 116.40, "name": "Beijing"},
    "changsha":  {"lat": 28.23, "lon": 112.94, "name": "Changsha"},
    "shenzhen":  {"lat": 22.54, "lon": 114.06, "name": "Shenzhen"},
    "kunming":   {"lat": 25.04, "lon": 102.68, "name": "Kunming"},
    # ── 10 new cities ──
    "changchun": {"lat": 43.88, "lon": 125.32, "name": "Changchun"},
    "shenyang":  {"lat": 41.80, "lon": 123.43, "name": "Shenyang"},
    "jinan":     {"lat": 36.65, "lon": 116.99, "name": "Jinan"},
    "xian":      {"lat": 34.26, "lon": 108.94, "name": "Xian"},
    "wuhan":     {"lat": 30.58, "lon": 114.30, "name": "Wuhan"},
    "nanjing":   {"lat": 32.06, "lon": 118.77, "name": "Nanjing"},
    "guangzhou": {"lat": 23.13, "lon": 113.27, "name": "Guangzhou"},
    "xiamen":    {"lat": 24.48, "lon": 118.09, "name": "Xiamen"},
    "guiyang":   {"lat": 26.65, "lon": 106.63, "name": "Guiyang"},
    "chengdu":   {"lat": 30.67, "lon": 104.07, "name": "Chengdu"},
    # ── 24 new cities (NC expansion) ──
    "dalian":       {"lat": 38.91, "lon": 121.61, "name": "Dalian"},
    "hohhot":       {"lat": 40.84, "lon": 111.75, "name": "Hohhot"},
    "tangshan":     {"lat": 39.63, "lon": 118.18, "name": "Tangshan"},
    "urumqi":       {"lat": 43.83, "lon":  87.61, "name": "Urumqi"},
    "taiyuan":      {"lat": 37.87, "lon": 112.55, "name": "Taiyuan"},
    "shijiazhuang": {"lat": 38.04, "lon": 114.50, "name": "Shijiazhuang"},
    "lanzhou":      {"lat": 36.06, "lon": 103.83, "name": "Lanzhou"},
    "yinchuan":     {"lat": 38.49, "lon": 106.23, "name": "Yinchuan"},
    "xining":       {"lat": 36.62, "lon": 101.78, "name": "Xining"},
    "qingdao":      {"lat": 36.07, "lon": 120.38, "name": "Qingdao"},
    "wuxi":         {"lat": 31.49, "lon": 120.31, "name": "Wuxi"},
    "suzhou":       {"lat": 31.30, "lon": 120.62, "name": "Suzhou"},
    "tianjin":      {"lat": 39.08, "lon": 117.20, "name": "Tianjin"},
    "zhengzhou":    {"lat": 34.75, "lon": 113.65, "name": "Zhengzhou"},
    "hangzhou":     {"lat": 30.27, "lon": 120.15, "name": "Hangzhou"},
    "hefei":        {"lat": 31.82, "lon": 117.23, "name": "Hefei"},
    "nanchang":     {"lat": 28.68, "lon": 115.86, "name": "Nanchang"},
    "ningbo":       {"lat": 29.87, "lon": 121.55, "name": "Ningbo"},
    "shanghai":     {"lat": 31.23, "lon": 121.47, "name": "Shanghai"},
    "chongqing":    {"lat": 29.56, "lon": 106.55, "name": "Chongqing"},
    "fuzhou":       {"lat": 26.07, "lon": 119.30, "name": "Fuzhou"},
    "nanning":      {"lat": 22.82, "lon": 108.37, "name": "Nanning"},
    "haikou":       {"lat": 20.04, "lon": 110.32, "name": "Haikou"},
    "lhasa":        {"lat": 29.65, "lon":  91.11, "name": "Lhasa"},
    # ── 2 non-mainland cities (NC extension) ──
    "hongkong":     {"lat": 22.32, "lon": 114.17, "name": "Hong Kong"},
    "taipei":       {"lat": 25.03, "lon": 121.57, "name": "Taipei"},
}


def download_tmy(city_key: str, lat: float, lon: float) -> bool:
    """Download TMY data from PVGIS."""
    log.info(f"  Downloading TMY for {city_key} ({lat}, {lon})...")

    params = {
        "lat": lat,
        "lon": lon,
        "outputformat": "csv",
        "browser": 0,
    }

    try:
        resp = requests.get(f"{PVGIS_BASE}/tmy", params=params, timeout=120)
        resp.raise_for_status()

        out_path = RESULTS_DIR / f"{city_key}_tmy.csv"
        out_path.write_text(resp.text, encoding="utf-8")
        log.info(f"  Saved: {out_path}")
        return True

    except Exception as e:
        log.error(f"  TMY download failed: {e}")
        return False


def download_hourly(
    city_key: str, lat: float, lon: float,
    start_year: int = 2005, end_year: int = 2020,
) -> bool:
    """Download multi-year hourly irradiation series from PVGIS."""
    log.info(f"  Downloading hourly series for {city_key} "
             f"({start_year}-{end_year})...")

    params = {
        "lat": lat,
        "lon": lon,
        "startyear": start_year,
        "endyear": end_year,
        "outputformat": "csv",
        "browser": 0,
        "components": 1,       # Include beam, diffuse, reflected components
    }

    try:
        resp = requests.get(f"{PVGIS_BASE}/seriescalc", params=params, timeout=300)
        resp.raise_for_status()

        out_path = RESULTS_DIR / f"{city_key}_hourly.csv"
        out_path.write_text(resp.text, encoding="utf-8")
        log.info(f"  Saved: {out_path}")
        return True

    except Exception as e:
        log.error(f"  Hourly download failed: {e}")
        return False


def save_metadata(city_key: str, lat: float, lon: float, results: dict):
    """Save download metadata."""
    meta = {
        "city": city_key,
        "latitude": lat,
        "longitude": lon,
        "api_version": "v5_3",
        "download_timestamp": datetime.now().isoformat(),
        "results": results,
    }
    meta_path = RESULTS_DIR / f"{city_key}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Paper 4: PVGIS Data Download"
    )
    parser.add_argument("--city", type=str, default=None)
    parser.add_argument("--tmy-only", action="store_true")
    parser.add_argument("--start-year", type=int, default=2005)
    parser.add_argument("--end-year", type=int, default=2020)
    args = parser.parse_args()

    cities = CITIES
    if args.city:
        key = args.city.lower()
        if key not in CITIES:
            log.error(f"Unknown city: {args.city}. Available: {list(CITIES.keys())}")
            sys.exit(1)
        cities = {key: CITIES[key]}

    log.info(f"Downloading PVGIS data for {len(cities)} cities")

    for city_key, info in cities.items():
        log.info(f"\n{'='*50}")
        log.info(f"{info['name']} ({info['lat']}°N, {info['lon']}°E)")
        log.info(f"{'='*50}")

        dl_results = {}

        dl_results["tmy"] = download_tmy(city_key, info["lat"], info["lon"])
        time.sleep(5)

        if not args.tmy_only:
            dl_results["hourly"] = download_hourly(
                city_key, info["lat"], info["lon"],
                args.start_year, args.end_year,
            )
            time.sleep(5)

        save_metadata(city_key, info["lat"], info["lon"], dl_results)

    log.info("\nAll downloads complete.")


if __name__ == "__main__":
    main()
