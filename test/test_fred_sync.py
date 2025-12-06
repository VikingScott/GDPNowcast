# test/test_fred_sync.py

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# ----- ensure src/ is on sys.path -----
ROOT = Path(__file__).resolve().parents[1]  # project root: GDPNowcast/
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.series_config import load_series_meta  # noqa: E402
from data.fred_sync import sync_all_series      # noqa: E402


@pytest.mark.slow
def test_sync_all_series_integration(tmp_path: Path):
    """
    Integration test:
      - Requires a valid FRED_API_KEY in environment / .env
      - Uses the real config/series.yaml
      - Downloads ALL series into a temporary data/raw directory
      - Verifies that for each series:
          * the expected CSV file exists
          * the columns match our standard:
              - vintage_source == "alfred":
                    ref_period, vintage_date, value
              - vintage_source != "alfred":
                    ref_period, value
    """

    # 1) 如果没有 API key，直接跳过这条测试，而不是报错
    if not os.getenv("FRED_API_KEY"):
        pytest.skip("FRED_API_KEY not set; skipping integration download test.")

    config_path = ROOT / "config" / "series.yaml"

    # 2) 加载元数据，看看总共有多少 series
    meta_dict = load_series_meta(config_path)
    assert len(meta_dict) > 0

    # 3) 让 sync_all_series 真正去 FRED/ALFRED 下载全部数据
    data_dir = tmp_path / "raw"

    results = sync_all_series(
        config_path=config_path,
        data_dir=data_dir,
        observation_start="1980-01-01",  # 你可以根据需要调整起始时间
        observation_end=None,
        client=None,  # 使用内部的 FredClient.from_env()
    )

    # 应该每个 series 都有返回路径
    assert set(results.keys()) == set(meta_dict.keys())

    # 4) 逐个 series 检查文件存在 & 列结构正确
    for name, meta in meta_dict.items():
        code = meta.code

        if meta.vintage_source == "alfred":
            expected_path = data_dir / f"{code}_vintage.csv"
            assert expected_path.exists(), f"Missing vintage file for {name}: {expected_path}"
            df = pd.read_csv(expected_path)

            # 至少这三列存在
            assert "ref_period" in df.columns, f"ref_period missing in {expected_path}"
            assert "vintage_date" in df.columns, f"vintage_date missing in {expected_path}"
            assert "value" in df.columns, f"value missing in {expected_path}"

            # 不要求行数，只要有列就行，但你可以检查非空
            assert len(df) > 0, f"Vintage file {expected_path} is empty"

        else:
            expected_path = data_dir / f"{code}.csv"
            assert expected_path.exists(), f"Missing final file for {name}: {expected_path}"
            df = pd.read_csv(expected_path)

            assert "ref_period" in df.columns, f"ref_period missing in {expected_path}"
            assert "value" in df.columns, f"value missing in {expected_path}"
            assert len(df) > 0, f"Final file {expected_path} is empty"
