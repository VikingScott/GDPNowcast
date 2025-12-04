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
        1. 无正则化 (No Regularization)：不会因为系数过大而惩罚模型。
           这对于 Target 被放大100倍、或者 Sticky CPI 这种本身数值很大的指标至关重要。
        2. 极简 (Simple)：就是最基本的 y = ax + b。
        3. 稳健 (Robust)：配合 RobustScaler 处理 2020 年的极值输入。
        """
        self.target_name = "gdp_real"
        self._build_model()

    def _build_model(self):
        # 依然保留 RobustScaler，防止输入的异常值(如油价暴负)破坏梯度
        # 但后端换成了最普通的 LinearRegression
        self.model = make_pipeline(
            RobustScaler(),
            LinearRegression()
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)