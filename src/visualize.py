"""
Stage 7a - Static visual artefacts (Folium maps + matplotlib figures).

Generates the "heatmap of parking violations vs. congestion impact" the brief
says is missing today, plus a ranked-hotspot map and supporting charts. Folium
is imported lazily so the data/ML pipeline still completes if it is absent.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import config as C

TIER_COLORS = {
    "Critical": "#b71c1c",
    "High":     "#e65100",
    "Medium":   "#f9a825",
    "Low":      "#2e7d32",
}


# --------------------------------------------------------------------------- #
# Folium maps
# --------------------------------------------------------------------------- #
def build_maps(df: pd.DataFrame, hotspots: pd.DataFrame, verbose: bool = True) -> list[str]:
    log = print if verbose else (lambda *a, **k: None)
    try:
        import folium
        from folium.plugins import HeatMap
    except Exception as e:  # folium not installed yet
        log(f"[viz] folium unavailable ({e}); skipping static maps")
        return []

    saved = []

    # 1) Congestion-impact heatmap (severity-weighted) ----------------------
    m = folium.Map(location=list(C.CITY_CENTER), zoom_start=12,
                   tiles="CartoDB positron")
    # subsample for a light HTML, weight by congestion severity
    samp = df.sample(min(40000, len(df)), random_state=1)
    heat = samp[["latitude", "longitude", "congestion_weight"]].to_numpy().tolist()
    HeatMap(heat, radius=9, blur=12, min_opacity=0.25,
            max_zoom=14).add_to(m)
    folium.LayerControl().add_to(m)
    p1 = os.path.join(C.MAPS_DIR, "congestion_heatmap.html")
    m.save(p1); saved.append(p1)

    # 2) Ranked-hotspot map -------------------------------------------------
    m2 = folium.Map(location=list(C.CITY_CENTER), zoom_start=12,
                    tiles="CartoDB positron")
    for r in hotspots.itertuples(index=False):
        color = TIER_COLORS.get(r.tier, "#555")
        popup = folium.Popup(html=_popup_html(r), max_width=320)
        folium.Circle(
            location=[r.lat, r.lon],
            radius=max(60, float(r.radius_m)),
            color=color, weight=1, fill=True, fill_color=color,
            fill_opacity=0.35, popup=popup,
        ).add_to(m2)
        folium.CircleMarker(
            location=[r.lat, r.lon], radius=4, color=color,
            fill=True, fill_opacity=0.9,
        ).add_to(m2)
    p2 = os.path.join(C.MAPS_DIR, "hotspot_map.html")
    m2.save(p2); saved.append(p2)

    log(f"[viz] saved {len(saved)} maps -> {C.MAPS_DIR}")
    return saved


def _popup_html(r) -> str:
    return (
        f"<b>#{r.rank} {r.tier} hotspot</b><br>"
        f"<b>CIS:</b> {r.cis}/100<br>"
        f"<b>Zone:</b> {r.junction_name or r.police_station}<br>"
        f"<b>Station:</b> {r.police_station}<br>"
        f"<b>Violations:</b> {r.n_violations:,}<br>"
        f"<b>At junction:</b> {r.junction_share*100:.0f}%<br>"
        f"<b>Peak hour:</b> {r.peak_hour}:00<br>"
        f"<b>Top:</b> {r.top_violations}"
    )


# --------------------------------------------------------------------------- #
# Matplotlib figures (for README / offline pitch)
# --------------------------------------------------------------------------- #
def build_figures(df: pd.DataFrame, hotspots: pd.DataFrame, profiles: dict,
                  verbose: bool = True) -> list[str]:
    log = print if verbose else (lambda *a, **k: None)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log(f"[viz] matplotlib unavailable ({e}); skipping figures")
        return []

    saved = []

    # hour x dow heatmap
    mat = np.array(profiles["hour_dow_matrix"])
    fig, ax = plt.subplots(figsize=(11, 3.6))
    im = ax.imshow(mat, aspect="auto", cmap="inferno")
    ax.set_yticks(range(7)); ax.set_yticklabels(profiles["dow_labels"])
    ax.set_xticks(range(0, 24, 2)); ax.set_xticklabels(range(0, 24, 2))
    ax.set_xlabel("Hour of day"); ax.set_title("Parking violations by hour x day")
    fig.colorbar(im, ax=ax, label="violations")
    fig.tight_layout()
    p = os.path.join(C.FIG_DIR, "hour_dow_heatmap.png"); fig.savefig(p, dpi=120); plt.close(fig)
    saved.append(p)

    # top hotspots bar (CIS)
    top = hotspots.head(15)
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"#{r.rank} {(r.junction_name or r.police_station)[:24]}" for r in top.itertuples(index=False)]
    colors = [TIER_COLORS.get(t, "#555") for t in top["tier"]]
    ax.barh(labels[::-1], top["cis"].to_numpy()[::-1], color=colors[::-1])
    ax.set_xlabel("Congestion Impact Score (0-100)")
    ax.set_title("Top 15 enforcement-priority hotspots")
    fig.tight_layout()
    p = os.path.join(C.FIG_DIR, "top_hotspots.png"); fig.savefig(p, dpi=120); plt.close(fig)
    saved.append(p)

    log(f"[viz] saved {len(saved)} figures -> {C.FIG_DIR}")
    return saved


if __name__ == "__main__":
    import data_pipeline as dp
    import hotspot_detection as hd
    import congestion_impact as ci
    import forecasting as fc

    d = dp.load_clean()
    d, hs = hd.detect(d, verbose=False)
    hs = ci.score(hs)
    prof = fc.build_profiles(d, hs)
    build_maps(d, hs)
    build_figures(d, hs, prof)
