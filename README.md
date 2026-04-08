# Paper 4: Multi-Factor Suitability Framework for Residential BIPV Deployment Across China's Climate Zones

## Project Structure

```
paper4/
├── configs/                    # Per-city YAML configuration files
│   ├── _template.yaml          # Template for new cities
│   ├── harbin.yaml             # 严寒区 - Severe Cold
│   ├── beijing.yaml            # 寒冷区 - Cold
│   ├── changsha.yaml           # 夏热冬冷 - Hot Summer Cold Winter
│   ├── shenzhen.yaml           # 夏热冬暖 - Hot Summer Warm Winter
│   └── kunming.yaml            # 温和区 - Mild
├── scripts/                    # Standalone runnable scripts
│   ├── 01_osm_audit.py         # Layer 3 data availability audit
│   ├── 02_pvgis_download.py    # PVGIS TMY + hourly data download
│   └── 03_batch_run.py         # Batch runner (placeholder)
├── src/                        # Core library modules
│   ├── data/                   # Data acquisition & preprocessing
│   │   ├── __init__.py
│   │   ├── osm_fetcher.py      # OSM building data retrieval
│   │   └── pvgis_fetcher.py    # PVGIS API wrapper
│   ├── suitability/            # FDSI indicator system (NEW for Paper 4)
│   │   ├── __init__.py
│   │   ├── indicators.py       # D1-D5 sub-indicator calculators
│   │   ├── weighting.py        # AHP / entropy / combined weighting
│   │   └── fdsi.py             # FDSI scoring engine
│   ├── comparison/             # Cross-city analysis
│   │   ├── __init__.py
│   │   └── cross_city.py       # Cross-city comparison utilities
│   └── utils/                  # Shared utilities
│       ├── __init__.py
│       └── config_loader.py    # YAML config loader
├── results/                    # Output directories
│   ├── osm_audit/              # OSM audit reports
│   ├── pvgis/                  # Downloaded PVGIS data
│   └── indicators/             # Computed indicator values
├── tools/                      # Level 2 deliverables
│   └── bipv_lookup.py          # Simple lookup tool (later)
└── docs/                       # Documentation & drafts
```

## Quick Start

### Step 1: OSM Data Audit (Layer 3 Screening)
```bash
# Install dependencies
pip install osmnx geopandas pandas pyyaml tabulate matplotlib

# Run audit for all candidate cities
python scripts/01_osm_audit.py

# Run audit for a specific city
python scripts/01_osm_audit.py --city beijing

# Include alternate candidates
python scripts/01_osm_audit.py --include-alternates
```

### Step 2: PVGIS Data Download
```bash
python scripts/02_pvgis_download.py
```

## Dependencies
- Python 3.10+
- osmnx >= 1.6
- geopandas >= 0.14
- pandas >= 2.0
- numpy >= 1.24
- pyyaml >= 6.0
- tabulate >= 0.9
- matplotlib >= 3.7
- seaborn >= 0.12
- pvlib >= 0.10 (for later steps)
- SALib >= 1.4 (for later steps)
- scipy >= 1.10 (for later steps)

## Relationship to Other Papers
- **Paper 2** (Changsha): Reuses OSM fetcher, proxy scoring framework
- **Paper 3** (Shenzhen+Changsha): Reuses pvlib ModelChain, MC engine, Sobol analysis
- **Paper 4** (this): Extends to 5 cities, adds FDSI indicator system
