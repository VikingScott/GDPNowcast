# src/models/bridge.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class BridgeFitResult:
    coef: np.ndarray          # shape (n_features,)
    intercept: float
    feature_names: list[str]


class SimpleBridgeModel:
    """
    Simple linear bridge baseline:
        y = a + b'X

    Assumes X is already transformed in Step 2
    and optionally standardized in Step 3.
    """

    def __init__(self):
        self.result: Optional[BridgeFitResult] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BridgeFitResult:
        X = X.copy()
        y = pd.to_numeric(y, errors="coerce")

        # Drop rows with missing y or any X
        valid = y.notna()
        valid &= X.notna().all(axis=1)

        X = X.loc[valid]
        y = y.loc[valid]

        feature_names = list(X.columns)

        if len(X) == 0:
            raise ValueError("No valid rows to fit SimpleBridgeModel.")

        # Add intercept column
        X_mat = np.column_stack([np.ones(len(X)), X.values.astype(float)])
        y_vec = y.values.astype(float)

        # OLS via least squares
        beta, *_ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
        intercept = float(beta[0])
        coef = beta[1:].astype(float)

        self.result = BridgeFitResult(
            coef=coef,
            intercept=intercept,
            feature_names=feature_names,
        )
        return self.result

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.result is None:
            raise RuntimeError("Model is not fit yet.")

        # Ensure same column order
        X = X[self.result.feature_names].copy()
        yhat = self.result.intercept + X.values.astype(float) @ self.result.coef
        return pd.Series(yhat, index=X.index, name="y_hat")


class StageAwareBridgeModel:
    """
    Stage-aware bridge:
        Fit 1 global bridge + up to 3 stage-specific bridges
        (early / mid / late).

    Prediction:
        Use stage-specific model if available,
        else fall back to global.

    This class is deliberately thin:
        - It does NOT compute buckets itself.
        - Caller provides `stage_labels` aligned with X/y index.
    """

    def __init__(self):
        self.global_model = SimpleBridgeModel()
        self.stage_models: Dict[str, SimpleBridgeModel] = {}
        self.feature_names: list[str] = []
        self.is_fit: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        stage_labels: pd.Series,
        min_stage_rows: int = 80,
        stages: tuple[str, ...] = ("early", "mid", "late"),
    ) -> "StageAwareBridgeModel":
        X = X.copy()
        y = pd.to_numeric(y, errors="coerce")
        stage_labels = stage_labels.reindex(X.index)

        # Fit global first (after dropping missing)
        self.global_model.fit(X, y)
        self.feature_names = list(X.columns)

        # Fit per-stage
        self.stage_models = {}
        for s in stages:
            mask = (stage_labels == s)
            Xs = X.loc[mask]
            ys = y.loc[mask]

            # Let SimpleBridgeModel handle NA dropping
            try:
                # Quick size guard before expensive fit
                rough_valid = ys.notna() & Xs.notna().all(axis=1)
                if int(rough_valid.sum()) < min_stage_rows:
                    continue

                m = SimpleBridgeModel()
                m.fit(Xs, ys)
                self.stage_models[s] = m
            except ValueError:
                # No valid rows
                continue

        self.is_fit = True
        return self

    def predict(
        self,
        X: pd.DataFrame,
        stage_labels: pd.Series,
    ) -> pd.Series:
        if not self.is_fit:
            raise RuntimeError("StageAwareBridgeModel is not fit yet.")

        X = X[self.feature_names].copy()
        stage_labels = stage_labels.reindex(X.index)

        # Default to global prediction
        yhat = self.global_model.predict(X).copy()

        # Override where stage model exists
        for s, m in self.stage_models.items():
            mask = (stage_labels == s)
            if mask.any():
                yhat.loc[mask] = m.predict(X.loc[mask]).values

        yhat.name = "y_hat"
        return yhat