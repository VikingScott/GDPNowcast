# nowcast/pipeline/run_cpi_nowcast.py

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error

from nowcast.data.fred import FredDataProvider
from nowcast.features.targets import get_target_series
from nowcast.features.panel_builder import PanelBuilder
from nowcast.features.asof_dataset import AsOfDatasetGenerator
from nowcast.models.ols import GDPNowcasterOLS

# --- 特征映射 ---
FEATURE_MAPPING = {
    'cpi_energy': ['oil_wti', 'gas_price'],
    'cpi_shelter': ['cpi_shelter_lag'],
    'cpi_food': ['cpi_food_lag', 'ppi_all'],
    'cpi_core': ['cpi_core_lag', 'hourly_earnings', 'consumer_sentiment'],
    'cpi_sticky': ['cpi_sticky_lag'],
    'cpi_headline': ['cpi_headline_lag', 'oil_wti', 'gas_price']
}

ALL_REQUIRED_FEATURES = sorted(list(set(
    [feat for feats in FEATURE_MAPPING.values() for feat in feats]
)))

def run_cpi_backtest(start_date="auto", end_date=None, freq="W-FRI"):
    print("🚀 Initializing CPI Nowcast Pipeline (Robust Alignment)...")
    
    provider = FredDataProvider(api_key="offline_mode") 
    
    # 1. 准备 Target
    y_dict = {}
    print("🎯 Fetching CPI Targets...")
    targets_to_fetch = list(FEATURE_MAPPING.keys())
    for target_name in targets_to_fetch:
        try:
            if target_name not in provider.series_config: continue
            s = get_target_series(provider, target_name=target_name, freq="M")
            y_dict[target_name] = s.dropna()
        except Exception as e:
            print(f"⚠️ Warning: {target_name}: {e}")

    if not y_dict:
        print("❌ No valid targets found.")
        return

    # 以 headline 为主轴
    y_main = y_dict.get('cpi_headline', list(y_dict.values())[0])

    # 2. 构建特征面板
    base_features = [f for f in ALL_REQUIRED_FEATURES if not f.endswith('_lag')]
    # 强制加入 payrolls 防止空面板
    if 'payrolls' not in base_features: base_features.append('payrolls')
    
    panel_full = PanelBuilder(provider).build_monthly_panel(base_features)
    
    # 构建滞后项
    print("✨ Engineering AR features...")
    lag_features = [f for f in ALL_REQUIRED_FEATURES if f.endswith('_lag')]
    for lag_feat in lag_features:
        original_name = lag_feat.replace('_lag', '')
        if original_name in provider.series_config:
            s_raw = provider.get_series(original_name)
            s_raw.index = pd.to_datetime(s_raw.index)
            
            s_aligned = s_raw.copy()
            s_aligned.index = s_aligned.index + pd.tseries.offsets.MonthEnd(0)
            
            # Shift 1 month
            s_aligned = s_aligned.shift(1)
            
            s_aligned = s_aligned.reindex(panel_full.index)
            panel_full[lag_feat] = s_aligned

    # 3. 初始化 Generator
    gen = AsOfDatasetGenerator(panel_full, y_main, target_freq="M")
    
    if end_date is None: end_date = pd.Timestamp.now()
    if start_date == "auto": start_date = pd.Timestamp("1990-01-01")
    else: start_date = pd.Timestamp(start_date)

    eval_dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    if eval_dates[-1].date() < pd.Timestamp.now().date():
        eval_dates = eval_dates.union([pd.Timestamp.now()])

    print(f"📅 Running from {start_date.date()} to {eval_dates[-1].date()}...")

    # --- 预计算 ---
    print("⚡ Pre-computing feature vectors...")
    historical_X_map = {}
    
    # [关键修复 1] 这里只计算 panel_full 中存在的日期
    # 这样就能知道我们到底有哪些历史特征可用
    available_dates = panel_full.dropna(how='all').index
    # 取 y_main 和 available_dates 的交集作为有效历史
    valid_history_dates = y_main.index.intersection(available_dates)

    for t_date in valid_history_dates:
        months = gen.get_period_months(t_date)
        X_vec, _ = gen.create_feature_vector(months, panel_full)
        historical_X_map[t_date] = X_vec
        
    results = []
    feat_cols = panel_full.columns.tolist()
    feat_indices_map = {col: i for i, col in enumerate(feat_cols)}

    # --- 主循环 ---
    for as_of_date in tqdm(eval_dates):
        training_cutoff = as_of_date - pd.Timedelta(days=30)
        
        # 1. 生成测试样本
        current_samples = gen.generate_dataset([as_of_date])
        test_sample = current_samples[0]
        X_test_df = pd.DataFrame([test_sample.X], columns=panel_full.columns)
        
        row = {
            "date": as_of_date,
            "target_period": test_sample.label,
            "data_completeness": test_sample.completeness
        }

        # 2. [关键修复 2] 确定可用训练集
        # 必须同时满足：(1) 在截止日期之前 (2) 在 historical_X_map 预计算缓存里
        valid_periods = y_main.index[y_main.index <= training_cutoff]
        # 取交集：确保我们既有 Y，又有预计算好的 X
        valid_periods = valid_periods.intersection(valid_history_dates)
        
        if len(valid_periods) < 24: 
            # 样本太少，跳过
            continue
        
        # 从缓存提取 X_train
        X_train_full = np.array([historical_X_map[d] for d in valid_periods])

        # 3. 针对每个目标训练
        for target_name, y_series in y_dict.items():
            required_feats = FEATURE_MAPPING.get(target_name, [])
            if not required_feats: continue
            
            feature_indices = [feat_indices_map[f] for f in required_feats if f in feat_indices_map]
            if not feature_indices: continue
            
            # 再做一次交集：确保该 target 也有历史数据
            common_idx = valid_periods.intersection(y_series.index)
            if len(common_idx) < 24:
                row[target_name] = np.nan
                continue
            
            # 找到 common_idx 在 valid_periods 中的位置索引
            time_indices = valid_periods.get_indexer(common_idx)
            
            # 切片
            X_train_sub = X_train_full[time_indices][:, feature_indices]
            y_train_sub = y_series.loc[common_idx].values
            X_test_sub = X_test_df[required_feats].values
            
            # OLS
            model = GDPNowcasterOLS()
            model.fit(X_train_sub, y_train_sub)
            
            if np.isnan(X_test_sub).any():
                X_test_sub = np.nan_to_num(X_test_sub)
                
            row[target_name] = model.predict(X_test_sub)[0]
            
        results.append(row)

    df_res = pd.DataFrame(results).set_index("date")
    print("\n✅ CPI Backtest Complete!")
    
    # 评估
    from nowcast.evaluation import evaluate_and_print
    for target_name in y_dict.keys():
        if target_name not in df_res.columns: continue
        y_truth_map = y_dict[target_name].to_dict()
        temp_df = df_res[[target_name]].copy()
        temp_df['actual'] = df_res['target_period'].apply(lambda x: y_truth_map.get(pd.Timestamp(x), np.nan))
        evaluate_and_print(temp_df, target_col=target_name, actual_col='actual', label=f"{target_name}")

    return df_res

if __name__ == "__main__":
    run_cpi_backtest(start_date="1990-01-01")