# src/data/panel.py

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .series_config import SeriesMeta, load_series_meta
from .calendar import build_calendar_for_all_series


# -----------------------------------------------------------------------------
# 单个 series：把 calendar DataFrame 变成 “每天一行 + 4 列” 的小 panel
# -----------------------------------------------------------------------------


def build_series_daily_panel(
    df_cal: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """
    对单个 series，基于 calendar DataFrame（含 available_date）构造每日面板：

    index: as_of_date (daily)
    columns:
        - value          : 截至该 as_of_date 能看到的最新观测值
        - ref_period     : 该值对应的 ref_period（哪一月 / 哪一季）
        - age_days       : 这条值“活了多少天”（as_of_date - last_available_date）
        - is_new_release : 当天是否有新值发布（1 = 今天有新 available_date，0 = 否）

    重要说明（简化假设）：
    ------------------------
    - 对同一个 available_date，如果这个 series 有多个 ref_period 更新
      （例如同一天既发布新月份，又修订旧月份）：
        * 我们只保留 ref_period 最大的那条（认为“headline”更新是最新期）。
        * 旧期的修订在这个 daily panel 中不会单独体现。
      这对于 bridge baseline 已经足够：我们只关心“当前最新期的 headline 值”。

    参数
    ----
    df_cal : pd.DataFrame
        对单个 series 调用 calendar.add_calendar_columns_for_series() 得到的结果。
        必须至少包含：
            - ref_period
            - available_date
            - value
    start_date, end_date : str or pd.Timestamp
        构造 as_of_date 的日历范围（闭区间）。

    返回
    ----
    pd.DataFrame
        index = as_of_date (每天)
        columns = ["value", "ref_period", "age_days", "is_new_release"]
    """
    if "available_date" not in df_cal.columns:
        raise ValueError("df_cal must contain 'available_date' column.")
    if "ref_period" not in df_cal.columns:
        raise ValueError("df_cal must contain 'ref_period' column.")
    if "value" not in df_cal.columns:
        raise ValueError("df_cal must contain 'value' column.")

    df = df_cal.copy()
    df["available_date"] = pd.to_datetime(df["available_date"])
    df["ref_period"] = pd.to_datetime(df["ref_period"])

    # 只保留有 available_date 的记录
    df = df.dropna(subset=["available_date"])

    # 按 available_date 和 ref_period 排序
    df = df.sort_values(["available_date", "ref_period"])

    # 对于同一个 available_date，只保留 ref_period 最大的一条：
    #   -> 认为这是“该次发布中最新期”的 headline 数值
    # groupby().tail(1) 保留每组中的最后一行
    df_event = df.groupby("available_date", as_index=False).tail(1)

    # 事件表：index = available_date
    df_event = df_event.set_index("available_date").sort_index()

    # 记录每条值的“最后可用日期”（即 available_date 本身）
    df_event["last_available_date"] = df_event.index

    # 构造每日 as_of_date 轴
    as_of_index = pd.date_range(start=start_date, end=end_date, freq="D")

    # 重新索引到每日，并向前填充，得到每天的“当前最新值”
    df_daily = df_event[["value", "ref_period", "last_available_date"]].reindex(as_of_index)
    df_daily = df_daily.ffill()
    df_daily.index.name = "as_of_date"

    # is_new_release：当天是否刚好是一个 available_date（有新观测进入）
    # 条件：as_of_date == last_available_date
    is_new = df_daily["last_available_date"] == df_daily.index

    # age_days：信息年龄（当前 as_of_date - 最近一次 last_available_date）
    # 注意：在尚未有任何发布之前，这里会是 NaT，对应的 age_days 为 NaN
    age_delta = df_daily.index.to_series() - df_daily["last_available_date"]
    age_days = age_delta.dt.days.astype("float")

    out = pd.DataFrame(
        {
            "value": df_daily["value"],
            "ref_period": df_daily["ref_period"],
            "age_days": age_days,
            "is_new_release": is_new.astype("int8"),
        },
        index=as_of_index,
    )

    return out


# -----------------------------------------------------------------------------
# 全部 series：构造 4 张 macro panel
# -----------------------------------------------------------------------------


def build_macro_panels(
    cal_data: Dict[str, pd.DataFrame],
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    """
    给定所有 series 的 calendar DataFrame（已经有 available_date 等），
    构造全局的 daily macro 面板。

    输入
    ----
    cal_data : dict[str, DataFrame]
        key   : series_name (e.g. "gdp_growth", "industrial_production")
        value : 对应 series 的 calendar DataFrame
                必须已经包含：
                    - ref_period
                    - available_date
                    - value
    start_date, end_date : str or pd.Timestamp
        面板的 as_of_date 范围（闭区间）。

    输出
    ----
    dict[str, DataFrame]
        {
          "value":      value_panel,      # as_of_date × series_name
          "ref_period": ref_period_panel, # as_of_date × series_name
          "age_days":   age_panel,        # as_of_date × series_name
          "is_new":     is_new_panel,     # as_of_date × series_name
        }

        - 所有 panel：
            index  = as_of_date (daily)
            columns = series_name 一致
        - value_panel      : 浮点数
        - ref_period_panel : datetime64[ns]
        - age_panel        : float / Int64（若部分日期没有发布过则为 NaN）
        - is_new_panel     : 0/1 （int8）
    """
    as_of_index = pd.date_range(start=start_date, end=end_date, freq="D")
    series_names = sorted(cal_data.keys())

    # 初始化 4 张大表
    value_panel = pd.DataFrame(index=as_of_index, columns=series_names, dtype="float64")
    ref_panel = pd.DataFrame(index=as_of_index, columns=series_names, dtype="datetime64[ns]")
    age_panel = pd.DataFrame(index=as_of_index, columns=series_names, dtype="float64")
    is_new_panel = pd.DataFrame(index=as_of_index, columns=series_names, dtype="int8")

    for name in series_names:
        df_cal = cal_data[name]

        daily = build_series_daily_panel(df_cal, start_date=start_date, end_date=end_date)

        value_panel[name] = daily["value"].astype("float64")
        ref_panel[name] = pd.to_datetime(daily["ref_period"])
        age_panel[name] = daily["age_days"].astype("float64")
        is_new_panel[name] = daily["is_new_release"].astype("int8")

    panels = {
        "value": value_panel,
        "ref_period": ref_panel,
        "age_days": age_panel,
        "is_new": is_new_panel,
    }

    return panels


# -----------------------------------------------------------------------------
# 便捷封装：从 raw CSV 一路构造到 macro panels（可选）
# -----------------------------------------------------------------------------


def build_macro_panels_from_raw(
    config_path: str | Path = "config/series.yaml",
    raw_data_dir: str | Path = "data/raw",
    start_date: str | pd.Timestamp = "1980-01-01",
    end_date: str | pd.Timestamp | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    便捷函数：从 raw CSV + series.yaml 一路构造到 daily macro panels。

    步骤：
      1) 加载元数据 series.yaml
      2) 使用 loaders.load_all_raw_series() 读取 data/raw 下的 CSV
      3) 使用 calendar.build_calendar_for_all_series() 添加 period_end / release_date / available_date
      4) 使用 build_macro_panels() 构造 4 张 daily panel

    注意：
      - 这是一个“一条龙”封装，方便你在 playground 中快速得到面板。
      - 在正式 pipeline 中，你也可以分步调用 loaders / calendar / panel 以保持更清晰的结构。
    """
    from .loaders import load_all_raw_series  # 局部导入避免循环依赖

    config_path = Path(config_path)
    raw_data_dir = Path(raw_data_dir)

    if end_date is None:
        # 如果未指定 end_date，就用今天
        end_date = pd.Timestamp.today().normalize()

    meta_dict: Dict[str, SeriesMeta] = load_series_meta(config_path)
    raw_all: Dict[str, pd.DataFrame] = load_all_raw_series(
        data_dir=raw_data_dir,
        config_path=config_path,
    )

    cal_data = build_calendar_for_all_series(
        raw_data=raw_all,
        meta_dict=meta_dict,
        config_path=config_path,
    )

    panels = build_macro_panels(
        cal_data=cal_data,
        start_date=start_date,
        end_date=end_date,
    )

    return panels