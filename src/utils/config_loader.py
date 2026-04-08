"""
Utility for loading city YAML configurations.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional


CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


def load_city_config(city_name: str) -> Dict[str, Any]:
    """Load a single city configuration by name (e.g., 'beijing')."""
    path = CONFIG_DIR / f"{city_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all_primary_configs() -> Dict[str, Dict[str, Any]]:
    """Load all primary city configs (excluding template and alternates)."""
    configs = {}
    skip = {"_template.yaml", "alternates.yaml"}
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        if path.name in skip:
            continue
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        city_key = path.stem
        configs[city_key] = cfg
    return configs


def load_alternates_config() -> Dict[str, Any]:
    """Load the alternates configuration."""
    path = CONFIG_DIR / "alternates.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Alternates config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_all_audit_targets(include_alternates: bool = False) -> List[Dict[str, Any]]:
    """
    Build a flat list of all cities to audit.
    Each entry has: name_en, name_cn, climate_zone, latitude, longitude,
                    place_query, is_primary
    """
    targets = []

    # Primary cities
    for city_key, cfg in load_all_primary_configs().items():
        c = cfg["city"]
        targets.append({
            "name_en": c["name_en"],
            "name_cn": c["name_cn"],
            "climate_zone": c["climate_zone"],
            "latitude": c["latitude"],
            "longitude": c["longitude"],
            "place_query": cfg["osm"]["place_query"],
            "is_primary": True,
        })

    # Alternates
    if include_alternates:
        alt_cfg = load_alternates_config()
        for zone, cities in alt_cfg.get("alternates", {}).items():
            for ac in cities:
                targets.append({
                    "name_en": ac["name_en"],
                    "name_cn": ac["name_cn"],
                    "climate_zone": zone,
                    "latitude": ac["latitude"],
                    "longitude": ac["longitude"],
                    "place_query": ac["place_query"],
                    "is_primary": False,
                })

    return targets
