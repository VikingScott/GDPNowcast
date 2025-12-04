# nowcast/pipeline/run_cpi_nowcast.py

import pandas as pd
import numpy as np
from tqdm import tqdm
from nowcast.data.fred import FredDataProvider
from nowcast.features.targets import get_target_series
from nowcast.features.panel_builder import PanelBuilder
from nowcast.features.asof_dataset import AsOfDatasetGenerator
from nowcast.models.ridge import GDPNowcasterRidge 

def run_cpi_backtest(start_date="auto", end_date=None, freq="W-FRI"):
    print("🚀 Initializing CPI Nowcast Pipeline...")
    
    provider = FredDataProvider(api_key="offline_mode") 
    y_full = get_target_series(provider, target_name="cpi_headline", freq="M")
    y_full = y_full.dropna()

    # --- [新增] 自动推断 ---
    if start_date == "auto":
        min_date = y_full.index.min()
        start_date = min_date + pd.DateOffset(years=2)
        print(f"📅 Auto-detected CPI start date: {start_date.date()}")
    # ---------------------

    all_series = list(provider.series_config.keys())
    exclude_list = ['cpi_headline', 'gdp_real'] # 排除 Target
    features_list = [k for k in all_series if k not in exclude_list]
    panel_full = PanelBuilder(provider).build_monthly_panel(features_list)
    
    gen = AsOfDatasetGenerator(panel_full, y_full, target_freq="M")
    
    if end_date is None: end_date = pd.Timestamp.now()
    eval_dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # 预计算
    print("⚡ Pre-computing CPI features...")
    historical_X_map = {}
    for t_date in y_full.index:
        months = gen.get_period_months(t_date)
        X_vec, _ = gen.create_feature_vector(months, panel_full)
        historical_X_map[t_date] = X_vec
        
    results = []
    
    for as_of_date in tqdm(eval_dates):
        training_cutoff = as_of_date - pd.Timedelta(days=30)
        valid_periods = y_full.index[y_full.index <= training_cutoff]
        if len(valid_periods) < 24: continue
            
        X_train_list = [historical_X_map[d] for d in valid_periods]
        y_train = y_full.loc[valid_periods].values
        X_train = np.array(X_train_list)
        
        current_sample_list = gen.generate_dataset([as_of_date])
        test_sample = current_sample_list[0]
        X_test = test_sample.X.reshape(1, -1)
        
        model = GDPNowcasterRidge() 
        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]
        
        results.append({
            "date": as_of_date,
            "nowcast": pred, # 统一 Key 名
            "data_completeness": test_sample.completeness
        })

    df_res = pd.DataFrame(results).set_index("date")
    return df_res

if __name__ == "__main__":
    run_cpi_backtest(start_date="auto")