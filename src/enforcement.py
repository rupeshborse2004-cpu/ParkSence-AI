"""
Stage 6 - Enforcement prioritisation & patrol optimisation.

Turns scores into a deployable plan and answers "difficult to prioritise
enforcement zones". Ranks hotspots by Congestion Impact Score, then greedily
assigns the limited pool of patrol units to the highest-impact zones inside
their busiest time window - maximising severity-weighted violations covered.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def build_plan(hotspots: pd.DataFrame, windows: list[dict],
               n_units: int = C.PATROL_UNITS) -> pd.DataFrame:
    """One row per recommended deployment (a patrol unit -> a zone + window)."""
    win = {w["hotspot_id"]: w for w in windows}
    ranked = hotspots.sort_values("cis", ascending=False).head(n_units)

    rows = []
    for unit, r in enumerate(ranked.itertuples(index=False), start=1):
        w = win.get(int(r.hotspot_id), {})
        rows.append({
            "unit": f"Unit-{unit:02d}",
            "priority_rank": int(r.rank),
            "tier": r.tier,
            "zone": w.get("name") or (r.junction_name or f"{r.police_station} zone"),
            "police_station": r.police_station,
            "lat": r.lat,
            "lon": r.lon,
            "cis": r.cis,
            "recommended_window": w.get("best_window", _fallback_window(r)),
            "day_focus": w.get("day_focus", "-"),
            "expected_violations": int(r.n_violations),
            "junction_share_pct": round(r.junction_share * 100, 0),
            "top_violations": r.top_violations,
            "radius_m": r.radius_m,
        })
    plan = pd.DataFrame(rows)
    return plan


def _fallback_window(r) -> str:
    h = int(getattr(r, "peak_hour", 9))
    end = (h + C.PATROL_SHIFT_HOURS) % 24
    return f"{h:02d}:00-{end:02d}:00"


def coverage_summary(hotspots: pd.DataFrame, plan: pd.DataFrame) -> dict:
    """How much of the city-wide impact does the plan target?"""
    total_sev = float(hotspots["severity_sum"].sum())
    planned_ids = set(
        hotspots.sort_values("cis", ascending=False)
        .head(len(plan))["hotspot_id"]
    )
    covered = float(
        hotspots[hotspots["hotspot_id"].isin(planned_ids)]["severity_sum"].sum()
    )
    crit = hotspots[hotspots["tier"].isin(["Critical", "High"])]
    return {
        "units_deployed": int(len(plan)),
        "zones_targeted": int(len(planned_ids)),
        "severity_covered_pct": round(100 * covered / total_sev, 1) if total_sev else 0,
        "critical_high_zones": int(len(crit)),
        "critical_high_violation_share_pct": round(
            100 * crit["n_violations"].sum() / hotspots["n_violations"].sum(), 1
        ) if len(hotspots) else 0,
    }


if __name__ == "__main__":
    import data_pipeline as dp
    import hotspot_detection as hd
    import congestion_impact as ci
    import forecasting as fc

    d = dp.load_clean()
    d, hs = hd.detect(d, verbose=False)
    hs = ci.score(hs)
    prof = fc.build_profiles(d, hs)
    plan = build_plan(hs, prof["hotspot_windows"])
    plan.to_csv(C.ENFORCEMENT_CSV, index=False)
    print(plan[["unit", "zone", "tier", "cis", "recommended_window",
                "expected_violations"]].to_string(index=False))
    print("\nCoverage:", coverage_summary(hs, plan))
