"""
Stage 1 - Data pipeline.

Loads the raw anonymised CSV, cleans it, converts timestamps to IST, parses the
JSON violation arrays, attaches a per-record Congestion Impact Weight and writes
a tidy parquet that every downstream module + the dashboard consumes.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config as C


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def _parse_json_list(value: object) -> list:
    """Robustly parse the '[""A"",""B""]' style cells into a python list."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return value
    s = str(value).strip()
    if not s or s.upper() == "NULL":
        return []
    try:
        out = json.loads(s)
        return out if isinstance(out, list) else [out]
    except Exception:
        # Fall back: strip brackets/quotes and split on comma.
        s = s.strip("[]")
        return [p.strip().strip('"').strip() for p in s.split(",") if p.strip()]


def _record_weight(viol_list: list[str]) -> float:
    """Congestion weight of a record = worst (max) weight among its violations."""
    if not viol_list:
        return C.NON_PARKING_WEIGHT
    weights = [C.CONGESTION_WEIGHTS.get(v, C.NON_PARKING_WEIGHT) for v in viol_list]
    return float(max(weights))


def _primary_violation(viol_list: list[str]) -> str:
    """The single most congestion-relevant violation in the record."""
    if not viol_list:
        return "OTHER"
    return max(viol_list, key=lambda v: C.CONGESTION_WEIGHTS.get(v, C.NON_PARKING_WEIGHT))


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #
USE_COLS = [
    "id", "latitude", "longitude", "location", "vehicle_type",
    "violation_type", "offence_code", "created_datetime",
    "police_station", "junction_name", "validation_status",
    "data_sent_to_scita",
]


def build(verbose: bool = True) -> pd.DataFrame:
    log = print if verbose else (lambda *a, **k: None)

    log("[pipeline] reading raw csv ...")
    df = pd.read_csv(
        C.RAW_CSV,
        usecols=USE_COLS,
        dtype=str,
        keep_default_na=False,
        na_values=["NULL", ""],
    )
    n0 = len(df)
    log(f"[pipeline] raw rows: {n0:,}")

    # --- numeric coords + bounding box -------------------------------------
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    in_box = (
        df["latitude"].between(C.LAT_MIN, C.LAT_MAX)
        & df["longitude"].between(C.LON_MIN, C.LON_MAX)
    )
    df = df[in_box].copy()
    log(f"[pipeline] after geo-clean: {len(df):,} "
        f"(dropped {n0 - len(df):,})")

    # --- timestamps -> IST -------------------------------------------------
    ts = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True)
    df = df[ts.notna()].copy()
    ts = ts[ts.notna()].dt.tz_convert(C.LOCAL_TZ)
    df["ts"] = ts.values
    df["date"] = ts.dt.date.values
    df["hour"] = ts.dt.hour.values
    df["dow"] = ts.dt.dayofweek.values          # 0 = Monday
    df["day_name"] = ts.dt.day_name().values
    df["month"] = ts.dt.month.values
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    def _bucket(h: int) -> str:
        for name, hours in C.HOUR_BUCKETS.items():
            if h in hours:
                return name
        return "other"

    df["hour_bucket"] = df["hour"].map(_bucket)
    df["is_morning_peak"] = df["hour"].between(*[C.MORNING_PEAK[0], C.MORNING_PEAK[1] - 1]).astype(int)
    df["is_evening_peak"] = df["hour"].between(*[C.EVENING_PEAK[0], C.EVENING_PEAK[1] - 1]).astype(int)
    df["is_peak"] = (df["is_morning_peak"] | df["is_evening_peak"]).astype(int)

    # --- validation filter -------------------------------------------------
    vs = df["validation_status"].astype("string").str.lower()
    df["validation_status"] = vs.fillna("pending")
    before = len(df)
    df = df[~df["validation_status"].isin(C.DROP_VALIDATION)].copy()
    log(f"[pipeline] dropped {before - len(df):,} rejected/duplicate rows")

    # --- violations --------------------------------------------------------
    log("[pipeline] parsing violation arrays ...")
    viol = df["violation_type"].map(_parse_json_list)
    df["violations"] = viol.map(lambda L: [str(x).strip().upper() for x in L])
    df["primary_violation"] = df["violations"].map(_primary_violation)
    df["congestion_weight"] = df["violations"].map(_record_weight)
    df["n_violations"] = df["violations"].map(len).clip(lower=1)
    df["is_parking"] = df["violations"].map(
        lambda L: int(any(v in C.PARKING_VIOLATIONS for v in L))
    )

    # --- junction ----------------------------------------------------------
    jn = df["junction_name"].astype("string").fillna("No Junction").str.strip()
    df["junction_name"] = jn
    df["at_junction"] = (~jn.str.lower().eq("no junction")).astype(int)

    df["police_station"] = df["police_station"].astype("string").fillna("Unknown").str.strip()
    df["vehicle_type"] = df["vehicle_type"].astype("string").fillna("UNKNOWN").str.strip()

    # --- snapped grids (reused by hotspots + risk model) -------------------
    df["cell_lat"] = (df["latitude"] / C.RISK_CELL_DEG).round() * C.RISK_CELL_DEG
    df["cell_lon"] = (df["longitude"] / C.RISK_CELL_DEG).round() * C.RISK_CELL_DEG
    df["cell_id"] = (
        df["cell_lat"].round(5).astype(str) + "_" + df["cell_lon"].round(5).astype(str)
    )

    keep = [
        "id", "latitude", "longitude", "location", "vehicle_type",
        "violations", "primary_violation", "congestion_weight", "n_violations",
        "is_parking", "ts", "date", "hour", "dow", "day_name", "month",
        "is_weekend", "hour_bucket", "is_morning_peak", "is_evening_peak",
        "is_peak", "police_station", "junction_name", "at_junction",
        "validation_status", "cell_lat", "cell_lon", "cell_id",
    ]
    df = df[keep].reset_index(drop=True)

    # violations is a list column -> store as JSON string for parquet safety
    out = df.copy()
    out["violations"] = out["violations"].map(json.dumps)
    out.to_parquet(C.CLEAN_PARQUET, index=False)
    log(f"[pipeline] wrote {C.CLEAN_PARQUET}  ({len(df):,} rows)")
    return df


def load_clean() -> pd.DataFrame:
    """Reload the processed parquet (decoding the violations list column)."""
    df = pd.read_parquet(C.CLEAN_PARQUET)
    df["violations"] = df["violations"].map(lambda s: json.loads(s) if isinstance(s, str) else s)
    df["ts"] = pd.to_datetime(df["ts"])
    return df


if __name__ == "__main__":
    build()
