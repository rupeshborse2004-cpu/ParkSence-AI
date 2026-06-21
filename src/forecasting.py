"""
Stage 4 - Temporal intelligence.

Answers the "reactive enforcement" pain point on the time axis: *when* do
violations actually happen? Builds hour-of-day, day-of-week and hour x dow
profiles (city-wide and per top hotspot) so patrols can be scheduled into the
windows that matter instead of patrolling blindly.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config as C


def build_profiles(df: pd.DataFrame, hotspots: pd.DataFrame, top_k: int = 10) -> dict:
    prof: dict = {}

    # City-wide hour-of-day (severity-weighted)
    by_hour = (
        df.groupby("hour")["congestion_weight"].sum().reindex(range(24), fill_value=0)
    )
    prof["by_hour"] = {int(h): round(float(v), 1) for h, v in by_hour.items()}

    # Day-of-week
    by_dow = (
        df.groupby("dow")["congestion_weight"].sum().reindex(range(7), fill_value=0)
    )
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    prof["by_dow"] = {dow_names[d]: round(float(v), 1) for d, v in by_dow.items()}

    # hour x dow matrix (counts) - powers the dashboard heatmap
    mat = (
        df.pivot_table(index="dow", columns="hour", values="id",
                       aggfunc="count", fill_value=0)
        .reindex(index=range(7), columns=range(24), fill_value=0)
    )
    prof["hour_dow_matrix"] = mat.to_numpy().tolist()
    prof["dow_labels"] = dow_names

    # Monthly trend
    by_month = df.groupby(df["ts"].dt.to_period("M").astype(str))["id"].count()
    prof["by_month"] = {k: int(v) for k, v in by_month.items()}

    # Vehicle mix
    veh = df["vehicle_type"].value_counts().head(8)
    prof["vehicle_mix"] = {str(k): int(v) for k, v in veh.items()}

    # Violation-type mix (exploded, parking only emphasised)
    vc: dict[str, int] = {}
    for L in df["violations"]:
        for v in L:
            vc[v] = vc.get(v, 0) + 1
    prof["violation_mix"] = dict(sorted(vc.items(), key=lambda kv: -kv[1])[:12])

    # Peak share
    prof["peak_share"] = round(float(df["is_peak"].mean()), 3)
    prof["morning_peak_share"] = round(float(df["is_morning_peak"].mean()), 3)
    prof["evening_peak_share"] = round(float(df["is_evening_peak"].mean()), 3)

    # Per top hotspot: best 3-hour patrol window (max rolling severity)
    prof["hotspot_windows"] = _hotspot_windows(df, hotspots, top_k)

    return prof


def _hotspot_windows(df: pd.DataFrame, hotspots: pd.DataFrame, top_k: int) -> list[dict]:
    out = []
    top = hotspots.nlargest(top_k, "cis")
    shift = C.PATROL_SHIFT_HOURS
    for r in top.itertuples(index=False):
        g = df[df["hotspot_id"] == r.hotspot_id]
        hourly = g.groupby("hour")["congestion_weight"].sum().reindex(range(24), fill_value=0)
        vals = hourly.to_numpy()
        # best contiguous `shift`-hour window (wrap-around aware)
        ext = np.concatenate([vals, vals[:shift]])
        roll = np.array([ext[i:i + shift].sum() for i in range(24)])
        start = int(roll.argmax())
        end = (start + shift) % 24
        out.append({
            "hotspot_id": int(r.hotspot_id),
            "name": _label(r),
            "best_window": f"{start:02d}:00-{end:02d}:00",
            "window_start": start,
            "window_severity": round(float(roll.max()), 1),
            "peak_hour": int(hourly.to_numpy().argmax()),
            "day_focus": _busiest_day(g),
        })
    return out


def _busiest_day(g: pd.DataFrame) -> str:
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    d = g.groupby("dow")["congestion_weight"].sum()
    if d.empty:
        return "-"
    return names[int(d.idxmax())]


def _label(r) -> str:
    if getattr(r, "junction_name", ""):
        return str(r.junction_name)
    return f"{r.police_station} zone"


def save(profiles: dict) -> None:
    with open(C.TEMPORAL_JSON, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)


if __name__ == "__main__":
    import data_pipeline as dp
    import hotspot_detection as hd
    import congestion_impact as ci

    d = dp.load_clean()
    d, hs = hd.detect(d, verbose=False)
    hs = ci.score(hs)
    p = build_profiles(d, hs)
    save(p)
    print("peak share:", p["peak_share"])
    print("busiest hours:", sorted(p["by_hour"].items(), key=lambda kv: -kv[1])[:5])
    for w in p["hotspot_windows"][:5]:
        print(w["name"], "->", w["best_window"], "day", w["day_focus"])
