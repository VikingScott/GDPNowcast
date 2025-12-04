# nowcast/evaluation.py

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

def evaluate_and_print(df: pd.DataFrame, 
                       target_col: str, 
                       actual_col: str, 
                       label: str = "Model") -> dict:
    """
    通用评估函数：计算指标并打印报告。
    
    Args:
        df: 包含预测值和真实值的 DataFrame
        target_col: 预测值列名 (e.g. 'nowcast')
        actual_col: 真实值列名 (e.g. 'actual')
        label: 报告标题前缀
        
    Returns:
        dict: 包含 r2, mse 等指标的字典
    """
    # 1. 清洗数据 (去除空值)
    df_eval = df.dropna(subset=[target_col, actual_col]).copy()
    
    # 2. 如果样本太少，跳过
    if len(df_eval) < 12:
        print(f"⚠️ {label}: Not enough samples to evaluate ({len(df_eval)}).")
        return {}

    # 3. 计算指标
    r2 = r2_score(df_eval[actual_col], df_eval[target_col])
    mse = mean_squared_error(df_eval[actual_col], df_eval[target_col])
    
    # 计算波动率比值 (Variance Ratio)
    # < 1 说明模型保守(欠拟合)，> 1 说明模型激进(过拟合)
    var_pred = np.var(df_eval[target_col])
    var_actual = np.var(df_eval[actual_col])
    var_ratio = var_pred / (var_actual + 1e-6)

    # 4. 打印报告表
    print("-" * 60)
    print(f"📊 EVALUATION REPORT: {label}")
    print("-" * 60)
    print(f"   Samples       : {len(df_eval)}")
    print(f"   R² Score      : {r2:.4f}  (>0 is good, 1.0 is perfect)")
    print(f"   MSE           : {mse:.4f}")
    print(f"   Var Ratio     : {var_ratio:.2f}    (Pred Var / Actual Var)")
    print("-" * 60)
    
    # 5. 打印最大的 3 个误差 (帮助定位问题)
    df_eval['error'] = df_eval[target_col] - df_eval[actual_col]
    top_errors = df_eval.sort_values(by='error', key=abs, ascending=False).head(3)
    
    print("   [Top 3 Largest Errors]")
    for idx, row in top_errors.iterrows():
        print(f"   {idx.date()}: Pred={row[target_col]:.2f}, Act={row[actual_col]:.2f}, Err={row['error']:.2f}")
    print("\n")
    
    return {
        "r2": r2,
        "mse": mse,
        "var_ratio": var_ratio
    }