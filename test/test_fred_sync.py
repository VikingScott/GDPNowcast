# test/test_fred_sync.py
"""
Integration-style check for real local data:

1. Use sync_all_series() to download/update data into data/raw.
2. Validate that all CSVs under data/raw follow the expected structure.

This is closer to a health-check / maintenance script than a unit test,
but pytest can still run it.
"""

import os
import sys
from pathlib import Path

import pandas as pd

# 如果你用 pytest 跑，这里可以用 pytest.skip；如果只当脚本跑也没问题
try:
    import pytest
except ImportError:  # 允许纯 python 直接运行
    pytest = None  # type: ignore

# ----- 确保 src/ 在 sys.path 里 -----
ROOT = Path(__file__).resolve().parents[1]  # 项目根目录: GDPNowcast/
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Ensure 'src/data/series_config.py' exists
if not (SRC / "data" / "series_config.py").exists():
    raise ImportError(f"Cannot find 'series_config.py' in {SRC / 'data'}")

from data.series_config import load_series_meta  # noqa: E402
from data.fred_sync import sync_all_series  # noqa: E402


def _sync_and_check() -> None:
    """
    真正的逻辑：
      1) 调用 sync_all_series() 更新 data/raw/
      2) 对 data/raw/ 里的每一个 series 做结构检查
    """
    # 1) 确认有 FRED_API_KEY
    if not os.getenv("FRED_API_KEY"):
        msg = "FRED_API_KEY not set; please configure .env or environment."
        if pytest is not None:
            pytest.skip(msg)
        else:
            raise RuntimeError(msg)

    config_path = ROOT / "config" / "series.yaml"
    data_dir = ROOT / "data" / "raw"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    data_dir.mkdir(parents=True, exist_ok=True)

    # 2) 读元数据
    meta_dict = load_series_meta(config_path)
    if len(meta_dict) == 0:
        raise RuntimeError("No series found in config/series.yaml")

    # 3) 同步所有 series（真正去 FRED/ALFRED 下载，并写入 data/raw/）
    print(f"[SYNC] Updating all series into {data_dir} ...")
    results = sync_all_series(
        config_path=config_path,
        data_dir=data_dir,
        observation_start="1980-01-01",  # 可以按需调整起始年份
        observation_end=None,
        client=None,  # 使用内部的 FredClient.from_env()
    )
    print(f"[SYNC] Done. {len(results)} series updated.")

    # 4) 检查 data/raw/ 里每个文件的结构
    for name, meta in meta_dict.items():
        code = meta.code

        if meta.vintage_source == "alfred":
            path = data_dir / f"{code}_vintage.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"[{name}] expected vintage file not found: {path}"
                )

            df = pd.read_csv(path)
            cols = set(df.columns)

            required = {"ref_period", "vintage_date", "value"}
            missing = required - cols
            if missing:
                raise AssertionError(
                    f"[{name}] vintage file {path} missing columns: {missing}; "
                    f"got {df.columns.tolist()}"
                )

            if len(df) == 0:
                raise AssertionError(
                    f"[{name}] vintage file {path} is empty."
                )

        else:
            path = data_dir / f"{code}.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"[{name}] expected final file not found: {path}"
                )

            df = pd.read_csv(path)
            cols = set(df.columns)

            required = {"ref_period", "value"}
            missing = required - cols
            if missing:
                raise AssertionError(
                    f"[{name}] final file {path} missing columns: {missing}; "
                    f"got {df.columns.tolist()}"
                )

            if len(df) == 0:
                raise AssertionError(
                    f"[{name}] final file {path} is empty."
                )

    print("[CHECK] All local data/raw files look structurally OK.")


# ---------- pytest 接口 ----------

def test_sync_and_check_real_data():
    """
    pytest 用的入口：跑一次真实同步 + 结构检查。

    注意：
      - 这不是传统“快小单测”，而是一个 integration / health-check。
      - 如果你不想每次 CI 都跑，可以用 `-m "not slow"` 之类的标记控制。
    """
    _sync_and_check()


# ---------- 直接 python 运行时的入口 ----------

if __name__ == "__main__":
    _sync_and_check()