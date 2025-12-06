            ┌───────────────┐
            │ series.yaml    │
            └───────┬───────┘
                    │
            series_config.py
                    │  SeriesMeta dict
                    ▼
         ┌────────────────────────┐
         │     fred_sync.py       │  ← 调用 fred_client 下载数据
         └───────────┬────────────┘
                     │ writes CSV
                     ▼
              data/raw/*.csv
                     │
         ┌────────────────────────┐
         │      loaders.py        │  ← 从本地读 final/vintage CSV
         └───────────┬────────────┘
                     │
           raw DataFrames (no release_date)
                     │
                     ▼
         ┌────────────────────────┐
         │     calendar.py        │  ← 根据 rules 生成 release_date
         └───────────┬────────────┘
                     │
       DF with release_date column
                     │
                     ▼
         ┌────────────────────────┐
         │        panel.py        │  ← vintage + as-of 面板构建
         └────────────────────────┘
                     │
                     ▼
        “as-of 实时宏观面板” —— 输入给模型层
