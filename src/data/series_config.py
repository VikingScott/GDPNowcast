# src/data/series_config.py

import yaml
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SeriesMeta:
    name: str
    vendor: str
    code: str
    freq: str
    transform: str
    release_rule: str
    release_lag_days: int
    revision_type: str
    vintage_source: str
    vintage_mode: str
    missing_policy: str


def load_series_meta(config_path: str | Path = "config/series.yaml") -> dict[str, SeriesMeta]:
    """
    Load macro series metadata from YAML config file.
    Returns: dict[str, SeriesMeta]
    """

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    series_meta_dict = {}

    for series_name, attrs in yaml_data.items():

        # Fill missing optional fields with defaults
        meta = SeriesMeta(
            name=attrs.get("name", series_name),
            vendor=attrs["vendor"],
            code=attrs["code"],
            freq=attrs["freq"],
            transform=attrs.get("transform", ""),
            release_rule=attrs["release_rule"],
            release_lag_days=attrs["release_lag_days"],
            revision_type=attrs.get("revision_type", "single"),
            vintage_source=attrs.get("vintage_source", "final"),
            vintage_mode=attrs.get("vintage_mode", "latest"),
            missing_policy=attrs.get("missing_policy", "skip"),
        )

        series_meta_dict[series_name] = meta

    return series_meta_dict
