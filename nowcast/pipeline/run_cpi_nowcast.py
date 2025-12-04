# nowcast/pipeline/run_cpi_nowcast.py

import pandas as pd
import numpy as np
from tqdm import tqdm

from nowcast.data.fred import FredDataProvider
from nowcast.features.targets import get_target_series
from nowcast.features.panel_builder import PanelBuilder
from nowcast.features.asof_dataset import AsOfDatasetGenerator
from nowcast.models.ridge import GDPNowcasterRidge 

# --- CPI 目标列表 (多维度) ---
CPI_TARGETS = [
    'cpi_headline', 
    'cpi_core',     
    'cpi_food',     
    'cpi_shelter',  
    'cpi_sticky'    
]

# --- 特征分类 (用于归因) ---
HARD_DATA = ['oil_wti', 'gas_price', 'ppi_all', 'hourly_earnings']
SOFT_DATA = ['inflation_breakeven', 'consumer_sentiment']

def run_cpi_backtest(start_date="auto", end_date=None, freq="W-FRI"):
    print("🚀 Initializing CPI Nowcast Pipeline (Multi-Target)...")
    
    provider = FredDataProvider(api_key="offline_mode") 
    
    # 1. 准备所有目标变量 (y_dict)
    y_dict = {}
    print("🎯 Fetching CPI Targets...")
    for target_name in CPI_TARGETS:
        try:
            s = get_target_series(provider, target_name=target_name, freq="M")
            y_dict[target_name] = s.dropna()
        except Exception as e:
            print(f"⚠️ Warning: Could not load target {target_name}: {e}")

    # 以 headline 为主时间轴 (因为它历史最长)
    y_main = y_dict['cpi_headline']

    # 2. 构建特征面板 (X)
    all_series = list(provider.series_config.keys())
    # 排除所有可能作为 Target 的列
    exclude_list = CPI_TARGETS + ['gdp_real']
    features_list = [k for k in all_series if k not in exclude_list]
    
    panel_full = PanelBuilder(provider).build_monthly_panel(features_list)
    
    # 确定 Hard/Soft 索引
    feat_cols = panel_full.columns.tolist()
    hard_indices = [i for i, col in enumerate(feat_cols) if col in HARD_DATA]
    soft_indices = [i for i, col in enumerate(feat_cols) if col in SOFT_DATA]

    # 3. 初始化生成器
    gen = AsOfDatasetGenerator(panel_full, y_main, target_freq="M")
    
    if end_date is None:
        end_date = pd.Timestamp.now()
    
    # 自动推断 start_date
    if start_date == "auto":
        # 统一从 1990 开始，避开早期数据质量问题
        # 如果非要更早，可以设为 y_main.index.min() + 2年
        start_date = pd.Timestamp("1990-01-01")
        print(f"📅 Auto-detected CPI start date: {start_date.date()}")
    else:
        start_date = pd.Timestamp(start_date)

    # 强制包含今天
    eval_dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    if eval_dates[-1].date() < pd.Timestamp.now().date():
        eval_dates = eval_dates.union([pd.Timestamp.now()])

    print(f"📅 Running from {start_date.date()} to {eval_dates[-1].date()}...")

    # --- 预计算历史特征 (X 是共用的) ---
    print("⚡ Pre-computing feature vectors...")
    historical_X_map = {}
    for t_date in y_main.index:
        months = gen.get_period_months(t_date)
        X_vec, _ = gen.create_feature_vector(months, panel_full)
        historical_X_map[t_date] = X_vec
        
    results = []
    
    # --- 主循环 ---
    for as_of_date in tqdm(eval_dates):
        # 训练窗口
        training_cutoff = as_of_date - pd.Timedelta(days=30)
        
        # 生成当前 X_test (一次生成，多次使用)
        current_samples = gen.generate_dataset([as_of_date])
        test_sample = current_samples[0]
        X_test = test_sample.X.reshape(1, -1)
        
        # 计算元数据
        hard_z = np.mean(test_sample.X[hard_indices]) if hard_indices else 0
        soft_z = np.mean(test_sample.X[soft_indices]) if soft_indices else 0
        
        row = {
            "date": as_of_date,
            "target_period": test_sample.label,
            "data_completeness": test_sample.completeness,
            "hard_data_z": hard_z,
            "soft_data_z": soft_z
        }

        # --- 多目标循环训练与预测 ---
        # 1. 确定 Headline 的可用历史区间
        valid_periods = y_main.index[y_main.index <= training_cutoff]
        if len(valid_periods) < 24: continue

        # 2. 获取基础 X_train (对应 Headline 的全量历史)
        X_train_base = np.array([historical_X_map[d] for d in valid_periods])

        for target_name, y_series in y_dict.items():
            # [关键修复] 获取该 target 的有效历史 (取交集)
            # 比如 Sticky CPI 只有 1967-2025，而 valid_periods 是 1947-2025
            # 我们必须只取 1967-2025 的部分
            common_idx = valid_periods.intersection(y_series.index)
            
            if len(common_idx) < 24: 
                # 如果这个分项历史太短，就不预测了
                row[target_name] = np.nan
                continue
            
            # [关键修复] 对齐 X 和 y
            # 我们需要知道 common_idx 在 valid_periods 里的位置，以便切分 X_train_base
            # 使用 get_indexer 获取整数索引
            indices = valid_periods.get_indexer(common_idx)
            
            X_train_sub = X_train_base[indices]
            y_train_sub = y_series.loc[common_idx].values
            
            # 训练
            model = GDPNowcasterRidge()
            model.fit(X_train_sub, y_train_sub)
            pred = model.predict(X_test)[0]
            
            row[target_name] = pred
            
        results.append(row)

    df_res = pd.DataFrame(results).set_index("date")
    print("\n✅ CPI Backtest Complete!")
    return df_res

if __name__ == "__main__":
    run_cpi_backtest(start_date="1990-01-01")