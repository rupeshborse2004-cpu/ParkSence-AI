"""
Stage 3 - Congestion Impact Score (CIS).

The judges' brief explicitly asks to *quantify the impact on traffic flow*, not
just count tickets. CIS blends five explainable, normalised components into a
0-100 score per hotspot:

    CIS = 100 * ( w_vol * Volume        # how many violations  (log-scaled)
                + w_sev * Severity       # how flow-choking they are
                + w_jct * Junction       # share at signalised junctions
                + w_per * Persistence    # chronic vs one-off (distinct days)
                + w_pk  * Peak )         # share during rush hour

Every component is stored so the score is fully auditable / defensible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-9:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def score(hotspots: pd.DataFrame) -> pd.DataFrame:
    h = hotspots.copy()
    if h.empty:
        return h

    # --- five normalised components (0..1) ---------------------------------
    # Volume: log-scaled so a handful of mega-hotspots don't flatten the rest.
    comp_volume = _minmax(np.log1p(h["n_violations"]))
    # Severity: mean congestion weight already 0..1, rescaled across hotspots.
    comp_severity = _minmax(h["severity_mean"])
    # Junction criticality: violations at signalised junctions hurt flow most.
    comp_junction = h["junction_share"].clip(0, 1)
    # Persistence: chronic spots active on many distinct days.
    comp_persist = _minmax(np.log1p(h["active_days"]))
    # Peak concentration: share happening during rush-hour windows.
    comp_peak = h["peak_share"].clip(0, 1)

    w = C.CIS_WEIGHTS
    cis = (
        w["volume"] * comp_volume
        + w["severity"] * comp_severity
        + w["junction"] * comp_junction
        + w["persistence"] * comp_persist
        + w["peak"] * comp_peak
    ) * 100.0

    h["cis"] = cis.round(2)
    h["c_volume"] = (comp_volume * 100).round(1)
    h["c_severity"] = (comp_severity * 100).round(1)
    h["c_junction"] = (comp_junction * 100).round(1)
    h["c_persistence"] = (comp_persist * 100).round(1)
    h["c_peak"] = (comp_peak * 100).round(1)

    h["tier"] = h["cis"].map(_tier)
    h["rank"] = h["cis"].rank(ascending=False, method="first").astype(int)
    h = h.sort_values("cis", ascending=False).reset_index(drop=True)
    return h


def _tier(v: float) -> str:
    for lo, label in C.CIS_TIERS:
        if v >= lo:
            return label
    return "Low"


def concentration_stats(df: pd.DataFrame, hotspots: pd.DataFrame, top_k: int = 20) -> dict:
    """How concentrated is the problem? Powers the 'targeting efficiency' pitch."""
    total_sev = float(df["congestion_weight"].sum())
    clustered = hotspots["severity_sum"].sum()
    top = hotspots.nlargest(top_k, "cis")["severity_sum"].sum()
    return {
        "total_severity": round(total_sev, 1),
        "hotspot_count": int(len(hotspots)),
        "severity_in_hotspots_pct": round(100 * clustered / total_sev, 1) if total_sev else 0,
        f"severity_in_top{top_k}_pct": round(100 * top / total_sev, 1) if total_sev else 0,
        "top_k": top_k,
    }


if __name__ == "__main__":
    import data_pipeline as dp
    import hotspot_detection as hd

    d = dp.load_clean()
    d, hs = hd.detect(d, verbose=False)
    hs = score(hs)
    hs.to_csv(C.HOTSPOTS_CSV, index=False)
    print(hs.head(15)[
        ["rank", "police_station", "junction_name", "n_violations",
         "cis", "tier", "c_severity", "c_junction"]
    ].to_string(index=False))
    print("\nConcentration:", concentration_stats(d, hs))
