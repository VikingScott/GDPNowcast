# test/test_data_layer.py

import pytest
from src.data.series_config import load_series_meta, SeriesMeta


def test_load_series_meta_basic():
    meta_dict = load_series_meta("config/series.yaml")

    # Must be a dict
    assert isinstance(meta_dict, dict)
    assert len(meta_dict) > 0

    # Check one known series
    assert "gdp_growth" in meta_dict
    gdp = meta_dict["gdp_growth"]

    # Correct type
    assert isinstance(gdp, SeriesMeta)

    # Check required fields
    assert isinstance(gdp.vendor, str)
    assert isinstance(gdp.code, str)
    assert isinstance(gdp.freq, str)
    assert isinstance(gdp.release_rule, str)
    assert isinstance(gdp.release_lag_days, int)

    # Optional fields should exist
    assert hasattr(gdp, "missing_policy")
    assert isinstance(gdp.missing_policy, str)


def test_all_series_have_valid_fields():
    meta_dict = load_series_meta("config/series.yaml")

    for name, meta in meta_dict.items():
        # vendor, code, freq, release_rule are required
        assert isinstance(meta.vendor, str)
        assert isinstance(meta.code, str)
        assert meta.freq in ("Q", "M", "W", "D")

        # release lag must be >= 0
        assert isinstance(meta.release_lag_days, int)
        assert meta.release_lag_days >= 0
