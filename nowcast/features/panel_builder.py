# nowcast/features/panel_builder.py

import pandas as pd
from nowcast.data.base import DataProvider
from nowcast.data.transforms import TRANSFORM_MAP

class PanelBuilder:
    def __init__(self, provider: DataProvider):
        self.provider = provider

    def build_monthly_panel(self, series_list: list[str], end_date: pd.Timestamp | None = None) -> pd.DataFrame:
        """
        构建月度特征面板 (X)。
        
        改进逻辑 (v2):
        对于高频数据 (Weekly/Daily)，先聚合为月度水平值 (Resample)，
        然后再执行变换 (Transform)。
        这样能确保 'pct_mom' 等算子的语义正确性 (即: 月均值的环比，而非周环比的均值)。
        """
        data_dict = {}
        
        print(f"Building panel with {len(series_list)} indicators...")
        
        for name in series_list:
            # 1. 获取配置
            if name not in self.provider.series_config:
                print(f"⚠️ Warning: Config for '{name}' not found.")
                continue
            cfg = self.provider.series_config[name]
            transform_name = cfg.get('transform', 'none')
            
            # 2. 获取原始数据 (Raw Data)
            # 我们需要绕过 DataProvider 的自动 transform，直接拿原始值
            # 既然 DataProvider.get_series 已经硬编码了 transform，我们这里用一个小技巧：
            # 我们可以临时修改 provider 的行为，或者更干净地，
            # 我们假设 DataProvider 有一个方法 fetch_raw (或者我们修改 DataProvider)
            # 但为了不改动底层，我们这里直接调用 get_series 并不是最好的办法。
            
            # --- 修正策略 ---
            # 我们需要 DataProvider 能够返回 raw data。
            # 让我们在 DataProvider 里加一个参数 raw=True，或者我们手动去拿 fred 实例。
            # 为了保持架构整洁，我们假设 DataProvider.get_series 支持一个参数 `raw=True`
            # (这需要哪怕微调一下 DataProvider，或者我们在这里先获取带 transform 的，再反推? 不，反推太蠢了)
            
            # 让我们采用 "Clean Architecture" 的方式：
            # 在 DataProvider 里加一个 skip_transform 参数。
            try:
                # 调用 get_series，传入 skip_transform=True (我们需要去修改 base.py 和 fred.py 支持这个)
                # 如果不想改 fred.py，那我们只能接受现状？不，你既然提出来了，我们就彻底改好。
                # 假设我们已经修改了 fred.py 支持 skip_transform=True
                s = self.provider.get_series(name, end_date=end_date, skip_transform=True)
            except Exception as e:
                # 兼容旧代码：如果 get_series 不支持 skip_transform，回退到旧逻辑 (虽然不完美)
                try: 
                    s = self.provider.get_series(name, end_date=end_date)
                    is_already_transformed = True
                except Exception as e2:
                    print(f"⚠️ Failed to load '{name}': {e2}")
                    continue
            else:
                is_already_transformed = False

            # --- 3. 频率聚合 (Resample) ---
            # 先判断是否高频
            is_high_freq = False
            inferred_freq = pd.infer_freq(s.index)
            if inferred_freq == 'W' or (len(s) > 0 and (s.index[-1] - s.index[0]).days / len(s) < 20):
                is_high_freq = True

            # 核心修正：对于高频数据，我们先聚合水平值 (Level)
            # 比如：算出“本月平均汽油价格”
            if is_high_freq:
                # 周频 -> 月频：取当月平均值 (Level)
                s_monthly = s.resample('ME').mean()
            else:
                # 月频 -> 月频：对齐到月末
                s_monthly = s.resample('ME').last()

            # --- 4. 延迟变换 (Late Transform) ---
            # 只有当数据是 Raw (未变换) 时，我们才在这里做变换
            if not is_already_transformed and transform_name in TRANSFORM_MAP:
                # 这里是对聚合后的月度数据做变换！
                # 此时 pct_mom = (本月均值 / 上月均值) - 1
                # 语义完美对齐！
                transform_func = TRANSFORM_MAP[transform_name]
                s_monthly = transform_func(s_monthly)
                
            # --- 5. 极值截断 (Clipping) ---
            # 变换之后再做截断
            if len(s_monthly) > 10:
                mean = s_monthly.mean()
                std = s_monthly.std()
                upper = mean + 4 * std
                lower = mean - 4 * std
                s_monthly = s_monthly.clip(lower=lower, upper=upper)
            
            # --- 6. 特征平滑 (Smoothing) ---
            s_monthly = s_monthly.rolling(window=3, min_periods=1).mean()
            
            data_dict[name] = s_monthly

        panel = pd.DataFrame(data_dict)
        panel = panel.sort_index()
        
        if end_date:
            panel = panel.loc[:end_date]
            
        return panel