from __future__ import annotations

from pathlib import Path
import pandas as pd


def find_project_root(start: Path) -> Path:
    """
    Walk upward from 'start' to find a directory that looks like the project root.
    Heuristic:
      - has config/series.yaml
      - and has src/ directory
    """
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "config" / "series.yaml").exists() and (p / "src").exists():
            return p
    # Fallback: assume parent of this file's folder is root-like
    # (useful if user hasn't created config/series.yaml yet)
    return start.parent


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def _standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure we have:
      model_name, n_obs, rmse, mae, bias, hit_rate

    We keep it tolerant because AR and Bridge might not be perfectly aligned.
    """
    df = df.copy()

    # Normalize count column variants
    rename_map = {
        "n": "n_obs",
        "obs": "n_obs",
        "num_obs": "n_obs",
        "N": "n_obs",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # Ensure model_name exists
    if "model_name" not in df.columns:
        df["model_name"] = "UNKNOWN"

    # Ensure metric columns exist
    for col in ["n_obs", "rmse", "mae", "bias", "hit_rate"]:
        if col not in df.columns:
            df[col] = pd.NA

    return df


def _to_long(
    df: pd.DataFrame,
    slice_type: str,
    slice_col: str | None = None,
    default_model_name: str | None = None,
) -> pd.DataFrame:
    df = df.copy()

    if default_model_name is not None:
        df["model_name"] = default_model_name

    df = _standardize_cols(df)

    # Overall case
    if slice_col is None:
        df["slice_type"] = "overall"
        df["slice"] = "overall"
        return df[["model_name", "slice_type", "slice", "n_obs", "rmse", "mae", "bias", "hit_rate"]]

    # If the expected slice column doesn't exist, try to infer
    if slice_col not in df.columns:
        candidates = ["time_slice", "distance_bucket", "bucket", "slice"]
        found = next((c for c in candidates if c in df.columns), None)
        if found is None:
            raise ValueError(
                f"Cannot find slice column for {slice_type}. "
                f"Expected '{slice_col}'. Got columns={df.columns.tolist()}"
            )
        slice_col = found

    df["slice_type"] = slice_type
    df["slice"] = df[slice_col].astype(str)

    return df[["model_name", "slice_type", "slice", "n_obs", "rmse", "mae", "bias", "hit_rate"]]


def build_model_compare(
    eval_dir: Path | None = None,
    save_path: Path | None = None,
) -> pd.DataFrame:
    # Robust root detection no matter where this script lives
    ROOT = find_project_root(Path(__file__).resolve())

    if eval_dir is None:
        eval_dir = ROOT / "eval"
    else:
        eval_dir = Path(eval_dir)

    # -----------------------------
    # AR files (prefer your current structure)
    # -----------------------------
    ar_dir = eval_dir / "outputs" / "ar"
    ar_dist = _read_csv_if_exists(ar_dir / "metrics_by_distance.csv")
    ar_time = _read_csv_if_exists(ar_dir / "metrics_by_time_slice.csv")
    ar_over = _read_csv_if_exists(ar_dir / "metrics_overall.csv")

    # Fallbacks if you ever flatten AR outputs
    if ar_dist is None:
        ar_dist = _read_csv_if_exists(eval_dir / "metrics_by_distance.csv")
    if ar_time is None:
        ar_time = _read_csv_if_exists(eval_dir / "metrics_by_time_slice.csv")
    if ar_over is None:
        ar_over = _read_csv_if_exists(eval_dir / "metrics_overall.csv")

    # -----------------------------
    # Bridge files (authoritative)
    # -----------------------------
    b_dist_simple = _read_csv_if_exists(eval_dir / "bridge_distance_bucket_simple.csv")
    b_dist_stage = _read_csv_if_exists(eval_dir / "bridge_distance_bucket_stage.csv")
    b_time_simple = _read_csv_if_exists(eval_dir / "bridge_time_slice_simple.csv")
    b_time_stage = _read_csv_if_exists(eval_dir / "bridge_time_slice_stage.csv")

    rows = []

    # AR
    if ar_dist is not None:
        rows.append(_to_long(ar_dist, "distance_bucket", "distance_bucket", "AR_p1_15y"))
    if ar_time is not None:
        rows.append(_to_long(ar_time, "time_slice", "time_slice", "AR_p1_15y"))
    if ar_over is not None:
        rows.append(_to_long(ar_over, "overall", None, "AR_p1_15y"))

    # Bridge simple
    if b_dist_simple is not None:
        rows.append(_to_long(b_dist_simple, "distance_bucket", "distance_bucket", "Bridge_simple_15y"))
    if b_time_simple is not None:
        rows.append(_to_long(b_time_simple, "time_slice", "time_slice", "Bridge_simple_15y"))

    # Bridge stage-aware
    if b_dist_stage is not None:
        rows.append(_to_long(b_dist_stage, "distance_bucket", "distance_bucket", "Bridge_stage_aware_15y"))
    if b_time_stage is not None:
        rows.append(_to_long(b_time_stage, "time_slice", "time_slice", "Bridge_stage_aware_15y"))

    if not rows:
        raise RuntimeError(
            "No eval files found.\n"
            "Checked paths:\n"
            "  eval/outputs/ar/metrics_by_distance.csv\n"
            "  eval/outputs/ar/metrics_by_time_slice.csv\n"
            "  eval/outputs/ar/metrics_overall.csv\n"
            "  eval/bridge_distance_bucket_simple.csv\n"
            "  eval/bridge_distance_bucket_stage.csv\n"
            "  eval/bridge_time_slice_simple.csv\n"
            "  eval/bridge_time_slice_stage.csv\n"
        )

    out = pd.concat(rows, axis=0, ignore_index=True)

    # Numeric coercion
    for col in ["n_obs", "rmse", "mae", "bias", "hit_rate"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.sort_values(["slice_type", "slice", "model_name"]).reset_index(drop=True)

    if save_path is None:
        save_path = eval_dir / "model_compare.csv"
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(save_path, index=False)

    return out


def main():
    out = build_model_compare()
    print("[OK] model compare saved to: eval/model_compare.csv")
    print(out.head(30).to_string(index=False))


if __name__ == "__main__":
    main()