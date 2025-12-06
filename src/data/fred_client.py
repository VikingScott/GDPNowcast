# src/data/fred_client.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

from pathlib import Path

import requests
import pandas as pd

try:
    # Optional: load .env automatically if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # It's fine if dotenv is not installed; environment may already have the key
    pass


FRED_BASE_URL = "https://api.stlouisfed.org/fred"


class FredAPIError(Exception):
    """Custom exception for FRED API errors."""


@dataclass
class FredClient:
    """
    Thin wrapper around the FRED/ALFRED API.

    Responsibilities (v1):
      - Read API key from env/.env
      - Provide methods to fetch:
          * final series (latest vintage only)
          * full vintage history (ALFRED-style)
      - Always return pandas DataFrame with normalized columns.
    """

    api_key: str

    @classmethod
    def from_env(cls) -> "FredClient":
        """
        Construct a FredClient by reading FRED_API_KEY from environment.
        """
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            raise RuntimeError(
                "FRED_API_KEY is not set. "
                "Please add it to your environment or .env file."
            )
        return cls(api_key=api_key)

    # ---------------------------------------------------------------------
    # Low-level HTTP helper
    # ---------------------------------------------------------------------

    def _get(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal helper to call the FRED API.

        Parameters
        ----------
        endpoint : str
            e.g. "series/observations"
        params : dict
            Query parameters; api_key and file_type=json will be added here.

        Returns
        -------
        dict : Parsed JSON response.

        Raises
        ------
        FredAPIError on non-200 response or JSON error.
        """
        url = f"{FRED_BASE_URL}/{endpoint}"

        full_params = dict(params)
        full_params["api_key"] = self.api_key
        full_params["file_type"] = "json"

        resp = requests.get(url, params=full_params, timeout=30)
        if resp.status_code != 200:
            raise FredAPIError(
                f"FRED API error {resp.status_code}: {resp.text[:200]}"
            )

        try:
            data = resp.json()
        except Exception as e:  # noqa: BLE001
            raise FredAPIError(f"Failed to parse JSON from FRED: {e}") from e

        if "error_code" in data:
            raise FredAPIError(f"FRED error: {data}")

        return data

    # ---------------------------------------------------------------------
    # Public methods
    # ---------------------------------------------------------------------

    def fetch_final_series(
        self,
        series_id: str,
        observation_start: Optional[str] = None,
        observation_end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch the latest (final) version of a time series.

        This corresponds to FRED "series/observations" without explicit
        realtime_start/realtime_end; FRED returns the latest available vintage.

        Parameters
        ----------
        series_id : str
            FRED series ID (e.g. "INDPRO", "PAYEMS").
        observation_start : str, optional
            Start date, e.g. "1950-01-01".
        observation_end : str, optional
            End date, e.g. "2025-12-31".

        Returns
        -------
        DataFrame with columns:
            ref_period : datetime64[ns]
            value      : float
        """
        params: Dict[str, Any] = {
            "series_id": series_id,
        }
        if observation_start:
            params["observation_start"] = observation_start
        if observation_end:
            params["observation_end"] = observation_end

        data = self._get("series/observations", params)

        obs = data.get("observations", [])
        if not obs:
            return pd.DataFrame(columns=["ref_period", "value"])

        df = pd.DataFrame(obs)
        # FRED returns 'date' and 'value' as strings.
        if "date" not in df.columns or "value" not in df.columns:
            raise FredAPIError(
                f"Unexpected response format for series {series_id}: "
                f"columns={df.columns.tolist()}"
            )

        df["ref_period"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        df = df[["ref_period", "value"]].dropna(subset=["value"]).sort_values(
            "ref_period"
        )
        df.reset_index(drop=True, inplace=True)
        return df

    def fetch_vintage_series_full(
        self,
        series_id: str,
        observation_start: Optional[str] = None,
        observation_end: Optional[str] = None,
        realtime_start: str = "1980-01-01",
        realtime_end: str = "9999-12-31",
    ) -> pd.DataFrame:
        """
        Fetch full vintage history (ALFRED-style) for a time series.

        This uses the same FRED endpoint with realtime_start/realtime_end
        spanning a wide range, so each observation row includes
        `realtime_start` and `realtime_end` fields.

        Parameters
        ----------
        series_id : str
            FRED series ID.
        observation_start : str, optional
            Start of the reference period range (e.g. "1950-01-01").
        observation_end : str, optional
            End of the reference period range.
        realtime_start : str
            Earliest vintage date to include.
        realtime_end : str
            Latest vintage date to include.

        Returns
        -------
        DataFrame with columns:
            ref_period   : datetime64[ns]
            vintage_date : datetime64[ns]
            value        : float

        Note
        ----
        - For many series, this can be a large dataset.
        - You may want to restrict observation_start / realtime_start
          for practical purposes.
        """
        params: Dict[str, Any] = {
            "series_id": series_id,
            "realtime_start": realtime_start,
            "realtime_end": realtime_end,
        }
        if observation_start:
            params["observation_start"] = observation_start
        if observation_end:
            params["observation_end"] = observation_end

        data = self._get("series/observations", params)

        obs = data.get("observations", [])
        if not obs:
            return pd.DataFrame(columns=["ref_period", "vintage_date", "value"])

        df = pd.DataFrame(obs)

        required_cols = {"date", "realtime_start", "value"}
        missing = required_cols - set(df.columns)
        if missing:
            raise FredAPIError(
                f"Unexpected vintage response for {series_id}, missing: {missing}"
            )

        df["ref_period"] = pd.to_datetime(df["date"])
        df["vintage_date"] = pd.to_datetime(df["realtime_start"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        df = df[["ref_period", "vintage_date", "value"]].dropna(
            subset=["value"]
        )
        df = df.sort_values(["ref_period", "vintage_date"]).reset_index(drop=True)
        return df
