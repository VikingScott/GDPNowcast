# nowcast/models/svr.py

# Currently not functional due to para setting issues and multi-threading bugs.

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler  # <--- 新增 RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import loguniform, uniform
from .base import NowcastModel

class GDPNowcasterSVR(NowcastModel):
    def __init__(self, 
                 kernel='rbf', 
                 C=1.0, 
                 epsilon=0.1, 
                 gamma='scale',
                 auto_tune=False):
        self.target_name = "gdp_real"
        self.auto_tune = auto_tune
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        
        self._build_model()

    def _build_model(self):
        # [关键修改] 使用 RobustScaler 替代 StandardScaler
        # RobustScaler 利用中位数和四分位距 (IQR) 进行缩放，
        # 对 2020 年这种极端异常值不敏感，能防止模型被"带偏"。
        self.model = make_pipeline(
            RobustScaler(), 
            SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon, gamma=self.gamma)
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.auto_tune and len(y) > 20:
            self.tune_and_fit(X, y)
        else:
            self.model.fit(X, y)
        return self

    def tune_and_fit(self, X, y):
        """
        使用 RandomizedSearchCV 寻找最佳参数。
        """
        param_dist = {
            'svr__kernel': ['linear', 'rbf'],     # 让数据决定是用线性还是非线性
            'svr__C': loguniform(1e-1, 1e3),      # C 的搜索范围
            'svr__epsilon': uniform(0.01, 0.5),   # 容错范围
            'svr__gamma': ['scale', 'auto']
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        
        search = RandomizedSearchCV(
            self.model, 
            param_distributions=param_dist,
            n_iter=10, 
            cv=tscv, 
            scoring='neg_mean_squared_error',
            n_jobs=1,
            random_state=42
        )
        search.fit(X, y)
        
        # 调试时可以取消注释查看选了什么参数
        print(f"🔍 Best Params: {search.best_params_}") 
        
        self.model = search.best_estimator_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)