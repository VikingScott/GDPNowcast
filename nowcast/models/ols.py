# nowcast/models/ols.py

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from .base import NowcastModel

class GDPNowcasterOLS(NowcastModel):
    def __init__(self):
        """
        朴素线性回归模型 (Ordinary Least Squares).
        
        特点：
        1. 无正则化 (No Regularization)：允许系数很大，适应 Target 被放大 100 倍的情况。
        2. 配合 RobustScaler：依然需要缩放输入，防止 2020 年极值破坏梯度。
        """
        self.target_name = "gdp_real"
        self._build_model()

    def _build_model(self):
        # 使用 RobustScaler + LinearRegression
        self.model = make_pipeline(
            RobustScaler(),
            LinearRegression()
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)