"""
Stage 2 - Hotspot detection.

Illegal-parking is spatially clustered. We snap violations to a ~39 m grid,
aggregate counts/severity per cell, then run weighted DBSCAN (haversine metric)
to discover organically-shaped *hotspots*. Each violation is tagged with its
hotspot id; per-hotspot statistics feed the Congestion Impact Score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

import config as C


def _aggregate_to_fine_grid(df: pd.DataFrame) -> pd.DataFrame:
    snap = C.HOTSPOT_SNAP_DEG
    g = df.assign(
        glat=(df["latitude"] / snap).round() * snap,
        glon=(df["longitude"] / snap).round() * snap,
    )
    agg = (
        g.groupby(["glat", "glon"])
        .agg(
            n=("id", "size"),
            sev=("congestion_weight", "sum"),
            junction=("at_junction", "mean"),
        )
        .reset_index()
    )
    return agg


def detect(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (df_with_hotspot_id, hotspots_table)."""
    log = print if verbose else (lambda *a, **k: None)

    agg = _aggregate_to_fine_grid(df)
    log(f"[hotspot] {len(agg):,} occupied ~39m cells")

    coords_rad = np.radians(agg[["glat", "glon"]].to_numpy())
    eps_rad = C.DBSCAN_EPS_M / C.EARTH_RADIUS_M

    db = DBSCAN(
        eps=eps_rad,
        min_samples=C.DBSCAN_MIN_WEIGHT,
        metric="haversine",
        algorithm="ball_tree",
    )
    # sample_weight lets a cell count as `n` points -> dense real hotspots win.
    labels = db.fit_predict(coords_rad, sample_weight=agg["n"].to_numpy())
    agg["cluster"] = labels
    n_clusters = int((agg["cluster"] >= 0).sum() and agg.loc[agg["cluster"] >= 0, "cluster"].nunique())
    log(f"[hotspot] DBSCAN found {n_clusters} hotspots "
        f"({(labels == -1).sum():,} cells noise)")

    # Map every violation to a hotspot via its fine-grid cell.
    snap = C.HOTSPOT_SNAP_DEG
    df = df.copy()
    df["glat"] = (df["latitude"] / snap).round() * snap
    df["glon"] = (df["longitude"] / snap).round() * snap
    cell2cluster = {
        (r.glat, r.glon): r.cluster for r in agg.itertuples(index=False)
    }
    df["hotspot_id"] = [
        cell2cluster.get((la, lo), -1) for la, lo in zip(df["glat"], df["glon"])
    ]

    hotspots = _summarise(df)
    log(f"[hotspot] summarised {len(hotspots)} hotspots covering "
        f"{df['hotspot_id'].ge(0).mean()*100:.1f}% of violations")
    return df, hotspots


def _top_name(series: pd.Series) -> str:
    s = series[series.astype(str).str.lower() != "no junction"]
    if len(s):
        return s.value_counts().idxmax()
    return ""


def _summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    clustered = df[df["hotspot_id"] >= 0]
    for hid, g in clustered.groupby("hotspot_id"):
        # centroid weighted by severity so the marker sits on the worst spot
        w = g["congestion_weight"].to_numpy()
        clat = float(np.average(g["latitude"], weights=w))
        clon = float(np.average(g["longitude"], weights=w))
        # spatial extent (radius, metres) ~ 95th pct distance from centroid
        dlat = (g["latitude"] - clat) * 111_320.0
        dlon = (g["longitude"] - clon) * 111_320.0 * np.cos(np.radians(clat))
        dist = np.sqrt(dlat**2 + dlon**2)
        radius_m = float(np.percentile(dist, 95)) if len(dist) else 0.0

        viol_counts: dict[str, int] = {}
        for L in g["violations"]:
            for v in L:
                viol_counts[v] = viol_counts.get(v, 0) + 1
        top_viol = sorted(viol_counts.items(), key=lambda kv: -kv[1])[:3]

        junction = _top_name(g["junction_name"])
        station = g["police_station"].value_counts().idxmax()
        n_days = g["date"].nunique()

        rows.append({
            "hotspot_id": int(hid),
            "lat": round(clat, 6),
            "lon": round(clon, 6),
            "radius_m": round(radius_m, 1),
            "n_violations": int(len(g)),
            "severity_sum": round(float(g["congestion_weight"].sum()), 2),
            "severity_mean": round(float(g["congestion_weight"].mean()), 3),
            "junction_share": round(float(g["at_junction"].mean()), 3),
            "peak_share": round(float(g["is_peak"].mean()), 3),
            "active_days": int(n_days),
            "parking_share": round(float(g["is_parking"].mean()), 3),
            "police_station": station,
            "junction_name": junction,
            "top_violations": "; ".join(f"{k} ({v})" for k, v in top_viol),
            "top_vehicle": g["vehicle_type"].value_counts().idxmax(),
            "peak_hour": int(g["hour"].value_counts().idxmax()),
        })
    out = pd.DataFrame(rows).sort_values("severity_sum", ascending=False)
    return out.reset_index(drop=True)


if __name__ == "__main__":
    import data_pipeline as dp

    d = dp.load_clean()
    d, hs = detect(d)
    hs.to_csv(C.HOTSPOTS_CSV, index=False)
    print(hs.head(15).to_string())
