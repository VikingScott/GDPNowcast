# src/models/ar_baseline.py
"""
AR baseline model for quarterly real GDP growth (H=0 nowcast).

Design goals:
- Use ONLY the quarterly GDP target series (no macro X),
- Strictly respect real-time availability: only quarters whose first release
  occurred before the evaluation time can be used for training,
- Rolling estimation with a fixed-length window in years,
- Outputs a per-quarter prediction table and a per-origin-date table
  compatible with the rest of the nowcast pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Core AR(p) model on a quarterly y_t series
# ----------------------------------------------------------------------


@dataclass
class ARQuarterlyModel:
    """
    Simple autoregressive model for quarterly y_t:

        y_t = c + phi_1 * y_{t-1} + ... + phi_p * y_{t-p} + eps_t

    This class does NOT know anything about dates or GDP specifically.
    It only operates on an ordered 1D array of y-values.
    """

    p: int = 1        # lag order
    include_const: bool = True

    # Fitted parameters (set by fit)
    coef_: Optional[np.ndarray] = None  # shape: (p + 1,) if include_const, else (p,)

    def fit(self, y: np.ndarray) -> None:
        """
        Fit AR(p) coefficients using ordinary least squares.

        Parameters
        ----------
        y : np.ndarray
            1D array of observations, ordered in time: [y_0, y_1, ..., y_{T-1}].
        """
        y = np.asarray(y, dtype=float)
        T = len(y)
        if T <= self.p:
            raise ValueError(
                f"Not enough observations to fit AR({self.p}): got {T}."
            )

        # Construct design matrix X and target vector Y_target:
        # For t = p, ..., T-1:
        #   Y_target[t - p] = y_t
        #   X[t - p, :] = [1, y_{t-1}, ..., y_{t-p}]  (if include_const)
        n_rows = T - self.p
        Y_target = y[self.p :]

        rows = []
        for t in range(self.p, T):
            # Lag vector: [y_{t-1}, ..., y_{t-p}]
            lags = [y[t - lag] for lag in range(1, self.p + 1)]
            if self.include_const:
                row = [1.0] + lags
            else:
                row = lags
            rows.append(row)

        X = np.asarray(rows, dtype=float)

        # OLS estimate: coef = (X' X)^{-1} X' Y
        # Use lstsq for numerical stability.
        coef, _, _, _ = np.linalg.lstsq(X, Y_target, rcond=None)
        self.coef_ = coef

    def predict_next(self, y_history: np.ndarray) -> float:
        """
        Predict y_{T} given past history [y_0, ..., y_{T-1}].

        Parameters
        ----------
        y_history : np.ndarray
            1D array of historical observations.

        Returns
        -------
        float
            Forecast for the next quarter.
        """
        if self.coef_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        y_history = np.asarray(y_history, dtype=float)
        T = len(y_history)
        if T < self.p:
            raise ValueError(
                f"Need at least {self.p} observations in y_history to predict next value."
            )

        # Take the last p observations
        recent = y_history[-self.p :]
        recent = recent[::-1]  # [y_{T-1}, y_{T-2}, ..., y_{T-p}]

        if self.include_const:
            x = np.concatenate(([1.0], recent))
        else:
            x = recent

        return float(np.dot(self.coef_, x))


# ----------------------------------------------------------------------
# AR baseline nowcaster: from gdp_cal to per-origin-date predictions
# ----------------------------------------------------------------------


@dataclass
class ARBaselineNowcaster:
    """
    AR baseline nowcaster using only the quarterly GDP target series.

    Responsibilities:
    - Extract a quarterly target table from gdp_cal (one row per quarter),
    - For each target quarter in the test period:
        * Build a rolling training window (in years),
        * Fit AR(p) on already-released quarters,
        * Produce a quarterly forecast y_hat,
    - Expand quarterly forecasts to daily origin_date rows.

    The class does NOT depend on project-specific loaders or panels.
    It operates purely on DataFrames provided by upstream layers.
    """

    p: int = 1
    train_window_years: int = 15
    min_train_quarters: int = 40
    model_name: str = "AR_baseline"

    def _make_quarterly_target_table(self, gdp_cal: pd.DataFrame) -> pd.DataFrame:
        """
        Construct a quarterly target table from gdp_cal.

        Expected columns in gdp_cal:
          - 'ref_period'    : datetime64[ns], quarter start date
          - 'release_date'  : datetime64[ns], first-release date for that quarter
          - 'value'         : float, first-release GDP growth (% SAAR)

        This function:
          - drops rows with missing ref_period / release_date / value,
          - sorts by (ref_period, release_date),
          - keeps the earliest release per ref_period,
          - renames 'value' to 'y'.

        Returns
        -------
        DataFrame with columns:
          - ref_period
          - release_date
          - y
        """
        required_cols = {"ref_period", "release_date", "value"}
        missing = required_cols - set(gdp_cal.columns)
        if missing:
            raise ValueError(
                f"gdp_cal is missing required columns: {missing}. "
                f"Available columns: {gdp_cal.columns.tolist()}"
            )

        df = gdp_cal.copy()
        df = df.dropna(subset=["ref_period", "release_date", "value"])

        df["ref_period"] = pd.to_datetime(df["ref_period"])
        df["release_date"] = pd.to_datetime(df["release_date"])

        df = df.sort_values(["ref_period", "release_date"])
        # groupby().first() keeps the earliest release_date per ref_period
        df_q = df.groupby("ref_period", as_index=False).first()

        df_q = df_q[["ref_period", "release_date", "value"]].rename(
            columns={"value": "y"}
        )

        df_q = df_q.sort_values("ref_period").reset_index(drop=True)
        return df_q

    def generate_quarterly_predictions(
        self,
        gdp_cal: pd.DataFrame,
        test_start_date: str | pd.Timestamp = "2010-01-01",
        test_end_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """
        Generate rolling AR(p) forecasts at the quarterly level.

        For each target quarter with ref_period in [test_start_date, test_end_date]:
          - Choose evaluation time t_eval = ref_period (quarter start),
          - Construct training set:
              * quarters with release_date < t_eval
              * and ref_period >= (t_eval - train_window_years),
          - Fit ARQuarterlyModel(p),
          - Predict y_hat for the target quarter.

        Parameters
        ----------
        gdp_cal : DataFrame
            Calendar DataFrame for gdp_growth with columns:
              - 'ref_period'
              - 'release_date'
              - 'value'
            as produced by the calendar layer.
        test_start_date : str or Timestamp
            Lower bound for target_ref_period.
        test_end_date : str or Timestamp, optional
            Upper bound for target_ref_period. If None, use last available ref_period.

        Returns
        -------
        DataFrame
            Columns:
              - target_ref_period
              - target_release_date
              - y
              - y_hat
              - train_start
              - train_end
              - model_name
        """
        df_q = self._make_quarterly_target_table(gdp_cal)

        if df_q.empty:
            raise ValueError("Quarterly GDP target table is empty.")

        test_start = pd.to_datetime(test_start_date)
        if test_end_date is None:
            test_end = df_q["ref_period"].max()
        else:
            test_end = pd.to_datetime(test_end_date)

        # Filter quarters in the test period
        mask_test = (df_q["ref_period"] >= test_start) & (df_q["ref_period"] <= test_end)
        df_test = df_q.loc[mask_test].copy()

        records = []

        for _, row in df_test.iterrows():
            q_ref = row["ref_period"]          # quarter start
            q_release = row["release_date"]    # first-release date
            y_true = float(row["y"])

            # Evaluation time: quarter start (H=0 nowcast)
            t_eval = q_ref

            # Training window in calendar years
            window_start = (t_eval - pd.DateOffset(years=self.train_window_years)).normalize()

            # Training quarters: released before t_eval, ref_period within window
            mask_train = (df_q["release_date"] < t_eval) & (df_q["ref_period"] >= window_start)
            df_train = df_q.loc[mask_train].copy()

            if len(df_train) < self.min_train_quarters or len(df_train) <= self.p:
                # Not enough data to fit AR(p) reliably
                y_hat = np.nan
                train_start = pd.NaT
                train_end = pd.NaT
            else:
                df_train = df_train.sort_values("ref_period")
                y_train = df_train["y"].to_numpy()

                ar = ARQuarterlyModel(p=self.p, include_const=True)
                ar.fit(y_train)
                y_hat = ar.predict_next(y_train)

                train_start = df_train["ref_period"].min()
                train_end = df_train["ref_period"].max()

            records.append(
                {
                    "target_ref_period": q_ref,
                    "target_release_date": q_release,
                    "y": y_true,
                    "y_hat": y_hat,
                    "train_start": train_start,
                    "train_end": train_end,
                    "model_name": self.model_name,
                }
            )

        df_pred = pd.DataFrame(records)
        df_pred = df_pred.sort_values("target_ref_period").reset_index(drop=True)
        return df_pred

    def expand_to_daily(
        self,
        quarterly_preds: pd.DataFrame,
        origin_start_date: str | pd.Timestamp | None = None,
        origin_end_date: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """
        Expand quarterly predictions to daily origin_date rows.

        For each quarter in quarterly_preds:
          - Let q_start = target_ref_period,
          - Let q_end   = quarter_end(q_start),
          - Define origin_date range as [q_start, q_end],
          - For each origin_date <= target_release_date - 1 day:
                * y_hat is constant within the quarter,
                * distance_to_release = (target_release_date - origin_date).days.

        Parameters
        ----------
        quarterly_preds : DataFrame
            Output of generate_quarterly_predictions().
        origin_start_date : str or Timestamp, optional
            Global lower bound for origin_date. If None, use earliest q_start.
        origin_end_date : str or Timestamp, optional
            Global upper bound for origin_date. If None, use latest quarter end.

        Returns
        -------
        DataFrame
            Columns:
              - model_name
              - origin_date
              - target_ref_period
              - target_release_date
              - distance_to_release
              - y
              - y_hat
              - train_start
              - train_end
        """
        if quarterly_preds.empty:
            raise ValueError("quarterly_preds is empty.")

        df_q = quarterly_preds.copy()
        df_q["target_ref_period"] = pd.to_datetime(df_q["target_ref_period"])
        df_q["target_release_date"] = pd.to_datetime(df_q["target_release_date"])

        # Determine global origin_date bounds
        if origin_start_date is None:
            origin_start = df_q["target_ref_period"].min()
        else:
            origin_start = pd.to_datetime(origin_start_date)

        if origin_end_date is None:
            # quarter_end for the latest quarter
            latest_q_start = df_q["target_ref_period"].max()
            origin_end = latest_q_start + pd.offsets.QuarterEnd(0)
        else:
            origin_end = pd.to_datetime(origin_end_date)

        records = []

        for _, row in df_q.iterrows():
            q_start = row["target_ref_period"]
            q_release = row["target_release_date"]
            y_true = float(row["y"])
            y_hat = float(row["y_hat"]) if not pd.isna(row["y_hat"]) else np.nan
            train_start = row["train_start"]
            train_end = row["train_end"]

            # Quarter end date
            q_end = q_start + pd.offsets.QuarterEnd(0)

            # Limit origin_date range by global bounds
            start = max(q_start, origin_start)
            end = min(q_end, origin_end)

            if start > end:
                continue

            dates = pd.date_range(start=start, end=end, freq="D")

            # We only keep origin_date strictly before the release date
            # (true pre-release nowcasting).
            mask_pre_release = dates < q_release
            dates = dates[mask_pre_release]

            if len(dates) == 0:
                continue

            # Distance to release
            dist = (q_release - dates).days

            for d, d_to_rel in zip(dates, dist):
                records.append(
                    {
                        "model_name": self.model_name,
                        "origin_date": d,
                        "target_ref_period": q_start,
                        "target_release_date": q_release,
                        "distance_to_release": int(d_to_rel),
                        "y": y_true,
                        "y_hat": y_hat,
                        "train_start": train_start,
                        "train_end": train_end,
                    }
                )

        df_daily = pd.DataFrame(records)
        if not df_daily.empty:
            df_daily = df_daily.sort_values(["origin_date", "target_ref_period"]).reset_index(drop=True)
        return df_daily