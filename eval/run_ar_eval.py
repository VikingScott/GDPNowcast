# eval/run_ar_eval.py

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


# -----------------------------
# Config (minimal, hard-coded)
# -----------------------------

DISTANCE_BUCKETS = [
    ("early", 60, None),
    ("mid", 30, 59),
    ("late", 10, 29),
    ("final", 0, 9),
]

TIME_SLICES = [
    ("2010_2014", "2010-01-01", "2014-12-31"),
    ("2015_2019", "2015-01-01", "2019-12-31"),
    ("2020_2022", "2020-01-01", "2022-12-31"),
    ("2023_plus", "2023-01-01", None),
]


def rmse(y, yhat) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = ~np.isnan(y) & ~np.isnan(yhat)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((yhat[mask] - y[mask]) ** 2)))


def mae(y, yhat) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = ~np.isnan(y) & ~np.isnan(yhat)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(yhat[mask] - y[mask])))


def bias(y, yhat) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = ~np.isnan(y) & ~np.isnan(yhat)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(yhat[mask] - y[mask]))


def hit_rate(y, yhat) -> float:
    """Directional accuracy: sign(y_hat) == sign(y)."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = ~np.isnan(y) & ~np.isnan(yhat)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.sign(yhat[mask]) == np.sign(y[mask])))


def add_distance_bucket(df: pd.DataFrame) -> pd.DataFrame:
    def bucket_one(x: float) -> str:
        for name, lo, hi in DISTANCE_BUCKETS:
            if hi is None:
                if x >= lo:
                    return name
            else:
                if lo <= x <= hi:
                    return name
        return "unknown"

    out = df.copy()
    out["distance_bucket"] = out["distance_to_release"].apply(bucket_one)
    return out


def compute_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(group_cols, dropna=False)

    for key, g in grouped:
        if not isinstance(key, tuple):
            key = (key,)

        y = g["y"].to_numpy()
        yhat = g["y_hat"].to_numpy()

        rows.append(
            dict(
                **{col: val for col, val in zip(group_cols, key)},
                n_obs=int(np.sum(~np.isnan(y) & ~np.isnan(yhat))),
                rmse=rmse(y, yhat),
                mae=mae(y, yhat),
                bias=bias(y, yhat),
                hit_rate=hit_rate(y, yhat),
            )
        )

    return pd.DataFrame(rows)


def main():
    ROOT = Path(__file__).resolve().parents[1]
    pred_path = ROOT / "data" / "results" / "ar_daily_predictions.csv"
    out_dir = ROOT / "eval" / "outputs" / "ar"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pred_path.exists():
        raise FileNotFoundError(
            f"Missing AR predictions file:\n  {pred_path}\n"
            "Please run your AR baseline script first to generate:\n"
            "  data/results/ar_daily_predictions.csv"
        )

    df = pd.read_csv(pred_path)

    # Required columns check
    required = {
        "model_name",
        "origin_date",
        "target_ref_period",
        "target_release_date",
        "distance_to_release",
        "y",
        "y_hat",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Predictions file missing columns: {missing}\n"
            f"Found columns: {df.columns.tolist()}"
        )

    # Normalize dates
    df["origin_date"] = pd.to_datetime(df["origin_date"])
    df["target_ref_period"] = pd.to_datetime(df["target_ref_period"])
    df["target_release_date"] = pd.to_datetime(df["target_release_date"])

    # 1) Overall metrics (by model_name)
    overall = compute_metrics(df, ["model_name"])
    overall.to_csv(out_dir / "metrics_overall.csv", index=False)

    # 2) By distance_to_release bucket
    df_b = add_distance_bucket(df)
    by_dist = compute_metrics(df_b, ["model_name", "distance_bucket"])
    # Optional: order buckets
    order = {name: i for i, (name, _, _) in enumerate(DISTANCE_BUCKETS)}
    by_dist["bucket_order"] = by_dist["distance_bucket"].map(order).fillna(999).astype(int)
    by_dist = by_dist.sort_values(["model_name", "bucket_order"]).drop(columns=["bucket_order"])
    by_dist.to_csv(out_dir / "metrics_by_distance.csv", index=False)

    # 3) By time slices
    slice_rows = []
    for slice_name, start, end in TIME_SLICES:
        start_dt = pd.to_datetime(start)
        if end is None:
            g = df[df["origin_date"] >= start_dt].copy()
        else:
            end_dt = pd.to_datetime(end)
            g = df[(df["origin_date"] >= start_dt) & (df["origin_date"] <= end_dt)].copy()

        if len(g) == 0:
            continue

        m = compute_metrics(g, ["model_name"])
        m.insert(1, "time_slice", slice_name)
        slice_rows.append(m)

    if slice_rows:
        by_time = pd.concat(slice_rows, ignore_index=True)
    else:
        by_time = pd.DataFrame(columns=["model_name", "time_slice", "n_obs", "rmse", "mae", "bias", "hit_rate"])

    by_time.to_csv(out_dir / "metrics_by_time_slice.csv", index=False)

    # Console quick summary
    print("[OK] AR evaluation completed.")
    print(f"  - {out_dir / 'metrics_overall.csv'}")
    print(f"  - {out_dir / 'metrics_by_distance.csv'}")
    print(f"  - {out_dir / 'metrics_by_time_slice.csv'}")


if __name__ == "__main__":
    main()