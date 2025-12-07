# src/training/rolling_trainer.py

from __future__ import annotations

# --- allow running this file directly ---
if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[2]  # .../GDPNowcast
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# --- imports that work in both modes ---
try:
    from ..models.bridge import SimpleBridgeModel
    from .bridge_dataset import build_bridge_dataset
except ImportError:
    from src.models.bridge import SimpleBridgeModel
    from src.training.bridge_dataset import build_bridge_dataset


# -----------------------------------------------------------------------------
# Rolling config (code-level for v1)
# -----------------------------------------------------------------------------

@dataclass
class RollingConfig:
    test_start_date: str = "1990-01-01"
    window_years: int = 15
    refit_freq: str = "MS"   # Month Start
    min_train_rows: int = 200

    # Step2 dataset location
    dataset_path: str = "data/processed/bridge_dataset/bridge_dataset_step2.parquet"

    # Output locations
    results_dir: str = "data/results/bridge"
    eval_dir: str = "eval"
    nowcast_dir: str = "data/results/nowcast"

    # CI config
    ci_z_95: float = 1.96


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _ensure_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _zscore_fit(X: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0).replace(0.0, np.nan)
    return mu, sd


def _zscore_apply(X: pd.DataFrame, mu: pd.Series, sd: pd.Series) -> pd.DataFrame:
    return (X - mu) / sd


def _get_refit_dates(start: pd.Timestamp, end: pd.Timestamp, freq: str) -> list[pd.Timestamp]:
    idx = pd.date_range(start=start, end=end, freq=freq)
    return list(idx)


def _slice_train_window(
    ds: pd.DataFrame,
    as_of: pd.Timestamp,
    window_years: int,
) -> pd.DataFrame:
    start = as_of - pd.DateOffset(years=window_years)
    mask = (ds["origin_date"] >= start) & (ds["origin_date"] < as_of)
    return ds.loc[mask].copy()


def _distance_bucket(origin: pd.Timestamp, release: pd.Timestamp) -> str:
    if pd.isna(release) or pd.isna(origin):
        return "unknown"
    d = (release - origin).days
    if d >= 61:
        return "early"
    elif d >= 21:
        return "mid"
    else:
        return "late"


def _time_slice_label(dt: pd.Timestamp) -> str:
    y = dt.year
    if 2010 <= y <= 2014:
        return "2010_2014"
    if 2015 <= y <= 2019:
        return "2015_2019"
    if 2020 <= y <= 2022:
        return "2020_2022"
    return "2023_plus"


def compute_metrics(df: pd.DataFrame, pred_col: str = "y_hat") -> Dict[str, float]:
    y = pd.to_numeric(df["y"], errors="coerce")
    yhat = pd.to_numeric(df[pred_col], errors="coerce")
    valid = y.notna() & yhat.notna()
    y = y[valid]
    yhat = yhat[valid]

    if len(y) == 0:
        return {"n_obs": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "hit_rate": np.nan}

    err = yhat - y
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))

    hit = np.sign(yhat) == np.sign(y)
    hit_rate = float(np.mean(hit.astype(float)))

    return {"n_obs": int(len(y)), "rmse": rmse, "mae": mae, "bias": bias, "hit_rate": hit_rate}


def _compute_stage_sigmas(
    train_df: pd.DataFrame,
    yhat_col: str = "y_hat_in",
) -> Tuple[Dict[str, float], float]:
    """
    Compute stage-aware residual sigma from training window.

    Returns
    -------
    (sigma_by_bucket, sigma_global)
    """
    y = pd.to_numeric(train_df["y"], errors="coerce")
    yhat = pd.to_numeric(train_df[yhat_col], errors="coerce")
    valid = y.notna() & yhat.notna()
    df = train_df.loc[valid].copy()
    if len(df) == 0:
        return {}, np.nan

    err = pd.to_numeric(df[yhat_col], errors="coerce") - pd.to_numeric(df["y"], errors="coerce")
    sigma_global = float(np.nanstd(err.values, ddof=0))

    # stage buckets require model_target_release_date
    if "model_target_release_date" not in df.columns:
        return {}, sigma_global

    df["distance_bucket"] = df.apply(
        lambda r: _distance_bucket(pd.Timestamp(r["origin_date"]),
                                   pd.Timestamp(r["model_target_release_date"])),
        axis=1,
    )

    sigma_by_bucket: Dict[str, float] = {}
    for b, g in df.groupby("distance_bucket"):
        e = pd.to_numeric(g[yhat_col], errors="coerce") - pd.to_numeric(g["y"], errors="coerce")
        s = float(np.nanstd(e.values, ddof=0))
        if np.isfinite(s) and s > 0:
            sigma_by_bucket[b] = s

    return sigma_by_bucket, sigma_global


# -----------------------------------------------------------------------------
# Main trainer
# -----------------------------------------------------------------------------

class RollingBridgeTrainer:
    """
    Step 3 (Simple Bridge + stage-aware uncertainty):

      - Load Step-2 dataset (transformed X).
      - Rolling training with 15y window, monthly refit.
      - Rolling z-score standardization using only training window.
      - Fit SimpleBridgeModel and predict daily for next refit interval.

    Stage-aware uncertainty (Option 3):
      - In each refit:
            * compute in-sample residuals on training window
            * estimate sigma for early/mid/late buckets
      - Apply bucket sigma to OOS predictions
      - Produce CI columns:
            nowcast_sigma, ci_low_95, ci_high_95

    Distance buckets MUST be computed against model_target_release_date.
    """

    def __init__(
        self,
        config: Optional[RollingConfig] = None,
        config_path: str | Path = "config/series.yaml",
        raw_data_dir: str | Path = "data/raw",
    ):
        self.cfg = config or RollingConfig()
        self.config_path = Path(config_path)
        self.raw_data_dir = Path(raw_data_dir)

    def load_or_build_step2(self) -> pd.DataFrame:
        path = Path(self.cfg.dataset_path)

        if path.exists():
            ds = pd.read_parquet(path)
        else:
            ds = build_bridge_dataset(
                panels=None,
                config_path=self.config_path,
                raw_data_dir=self.raw_data_dir,
                start_date="1980-01-01",
                end_date=None,
                target_name="gdp_growth",
                feature_names=None,
                save_path=path,
            )

        # Normalize key fields
        ds["origin_date"] = _ensure_datetime(ds["origin_date"])

        # Visible-fact columns (if exist)
        if "target_ref_period" in ds.columns:
            ds["target_ref_period"] = _ensure_datetime(ds["target_ref_period"])
        if "target_release_date" in ds.columns:
            ds["target_release_date"] = _ensure_datetime(ds["target_release_date"])

        # Model-task columns must exist for v1
        if "model_target_ref_period" in ds.columns:
            ds["model_target_ref_period"] = _ensure_datetime(ds["model_target_ref_period"])
        if "model_target_release_date" in ds.columns:
            ds["model_target_release_date"] = _ensure_datetime(ds["model_target_release_date"])

        return ds.sort_values("origin_date").reset_index(drop=True)

    def run(self) -> pd.DataFrame:
        ds = self.load_or_build_step2()

        test_start = pd.Timestamp(self.cfg.test_start_date)
        end_date = ds["origin_date"].max()

        x_cols = [c for c in ds.columns if c.startswith("x_")]
        if len(x_cols) == 0:
            raise ValueError("No x_ feature columns found in Step-2 dataset.")

        ds_test = ds.loc[ds["origin_date"] >= test_start].copy()
        if len(ds_test) == 0:
            raise ValueError("No rows found after test_start_date.")

        refit_dates = _get_refit_dates(test_start, end_date, self.cfg.refit_freq)
        if len(refit_dates) == 0 or refit_dates[0] != test_start:
            refit_dates = [test_start] + refit_dates

        preds = []

        for i, refit_date in enumerate(refit_dates):
            next_refit = (
                refit_dates[i + 1]
                if i + 1 < len(refit_dates)
                else (end_date + pd.Timedelta(days=1))
            )

            train = _slice_train_window(ds, refit_date, self.cfg.window_years)
            train = train.dropna(subset=["y"])

            if len(train) < self.cfg.min_train_rows:
                continue

            X_train = train[x_cols].copy()
            y_train = train["y"].copy()

            mu, sd = _zscore_fit(X_train)
            X_train_z = _zscore_apply(X_train, mu, sd)

            model = SimpleBridgeModel()
            model.fit(X_train_z, y_train)

            # -----------------------------
            # Stage-aware sigma estimation
            # -----------------------------
            # In-sample prediction on training window
            train = train.copy()
            train["y_hat_in"] = model.predict(X_train_z).values

            sigma_by_bucket, sigma_global = _compute_stage_sigmas(train, yhat_col="y_hat_in")

            # -----------------------------
            # Predict OOS for [refit, next_refit)
            # -----------------------------
            mask_oos = (ds_test["origin_date"] >= refit_date) & (ds_test["origin_date"] < next_refit)
            oos = ds_test.loc[mask_oos].copy()
            if len(oos) == 0:
                continue

            X_oos = oos[x_cols].copy()
            X_oos_z = _zscore_apply(X_oos, mu, sd)

            oos["y_hat"] = model.predict(X_oos_z).values
            oos["model_name"] = "Bridge_simple_15y"

            # distance bucket based on model-task release date
            if "model_target_release_date" in oos.columns:
                oos["distance_bucket"] = oos.apply(
                    lambda r: _distance_bucket(pd.Timestamp(r["origin_date"]),
                                               pd.Timestamp(r["model_target_release_date"])),
                    axis=1,
                )
            else:
                oos["distance_bucket"] = "unknown"

            # map sigma
            def _pick_sigma(b: str) -> float:
                if b in sigma_by_bucket:
                    return sigma_by_bucket[b]
                return sigma_global

            oos["nowcast_sigma"] = oos["distance_bucket"].map(_pick_sigma).astype("float64")

            z = float(self.cfg.ci_z_95)
            oos["ci_low_95"] = oos["y_hat"] - z * oos["nowcast_sigma"]
            oos["ci_high_95"] = oos["y_hat"] + z * oos["nowcast_sigma"]

            preds.append(oos)

        if not preds:
            raise RuntimeError("No predictions generated. Check min_train_rows or dataset coverage.")

        pred_df = pd.concat(preds, axis=0).sort_values("origin_date").reset_index(drop=True)
        return pred_df

    def save_outputs(self) -> None:
        """
        One-call end-to-end saver:
          - run rolling predictions
          - save predictions
          - save eval summaries
          - save clean nowcast output for All Weather
        """
        pred_df = self.run()

        results_dir = Path(self.cfg.results_dir)
        eval_dir = Path(self.cfg.eval_dir)
        nowcast_dir = Path(self.cfg.nowcast_dir)

        results_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)
        nowcast_dir.mkdir(parents=True, exist_ok=True)

        # 1) Save predictions (rich)
        pred_path = results_dir / "simple_bridge_predictions.csv"
        pred_df.to_csv(pred_path, index=False)

        # 2) Build eval slices
        df = pred_df.copy()
        df["origin_date"] = _ensure_datetime(df["origin_date"])

        # time slice
        df["time_slice"] = df["origin_date"].map(lambda d: _time_slice_label(pd.Timestamp(d)))

        # distance bucket already in run() but make robust
        if "distance_bucket" not in df.columns:
            if "model_target_release_date" in df.columns:
                df["distance_bucket"] = df.apply(
                    lambda r: _distance_bucket(pd.Timestamp(r["origin_date"]),
                                               pd.Timestamp(r["model_target_release_date"])),
                    axis=1,
                )
            else:
                df["distance_bucket"] = "unknown"

        # 3) Overall
        overall = compute_metrics(df)
        overall_df = pd.DataFrame(
            [{
                "model_name": "Bridge_simple_15y",
                "slice_type": "overall",
                "slice": "overall",
                **overall
            }]
        )
        overall_df.to_csv(eval_dir / "bridge_overall_simple.csv", index=False)

        # 4) Time-slice metrics
        rows = []
        for ts, g in df.groupby("time_slice"):
            m = compute_metrics(g)
            rows.append({"model_name": "Bridge_simple_15y", "slice_type": "time_slice", "slice": ts, **m})
        time_eval = pd.DataFrame(rows).sort_values("slice")
        time_eval.to_csv(eval_dir / "bridge_time_slice_simple.csv", index=False)

        # 5) Distance-bucket metrics
        rows = []
        for b, g in df.groupby("distance_bucket"):
            m = compute_metrics(g)
            rows.append({"model_name": "Bridge_simple_15y", "slice_type": "distance_bucket", "slice": b, **m})
        dist_eval = pd.DataFrame(rows).sort_values("slice")
        dist_eval.to_csv(eval_dir / "bridge_distance_bucket_simple.csv", index=False)

        # 6) Clean nowcast output for All Weather (Layer B)
        # Minimal + uncertainty
        keep_cols = ["origin_date", "y_hat"]
        if "model_target_ref_period" in df.columns:
            keep_cols.append("model_target_ref_period")
        if "nowcast_sigma" in df.columns:
            keep_cols.append("nowcast_sigma")
        if "ci_low_95" in df.columns:
            keep_cols.append("ci_low_95")
        if "ci_high_95" in df.columns:
            keep_cols.append("ci_high_95")

        out = df[keep_cols].copy()
        out = out.rename(columns={"y_hat": "nowcast_gdp_growth"})
        out = out.sort_values("origin_date")

        out_path = nowcast_dir / "gdp_nowcast_daily.csv"
        out.to_csv(out_path, index=False)

        # console hints
        print("[OK] Simple Bridge rolling + stage-aware CI finished.")
        print(f"Predictions: {pred_path}")
        print(f"Eval overall: {eval_dir / 'bridge_overall_simple.csv'}")
        print(f"Eval time   : {eval_dir / 'bridge_time_slice_simple.csv'}")
        print(f"Eval dist   : {eval_dir / 'bridge_distance_bucket_simple.csv'}")
        print(f"Nowcast     : {out_path}")


# -----------------------------------------------------------------------------
# CLI runner (not pytest)
# -----------------------------------------------------------------------------

def main():
    trainer = RollingBridgeTrainer()
    trainer.save_outputs()


if __name__ == "__main__":
    main()