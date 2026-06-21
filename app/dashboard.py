"""
ParkSense AI - interactive command centre (Streamlit).

    streamlit run app/dashboard.py

Reads the artefacts produced by `python src/build_all.py` and presents the
parking-intelligence story: live congestion-impact heatmap, ranked hotspots,
temporal patterns, the predictive risk model and a deployable patrol plan.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import streamlit as st

# --- make src importable + locate artefacts --------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
sys.path.insert(0, SRC)
import config as C  # noqa: E402

st.set_page_config(page_title="ParkSense AI - Bengaluru",
                   page_icon="🅿️", layout="wide")

st.markdown(
    """
    <style>
      #MainMenu, footer {visibility: hidden;}
      .block-container {padding-top: 1.3rem; padding-bottom: 1rem;}
      [data-testid="stMetricValue"] {font-size: 1.55rem; font-weight: 700;}
      [data-testid="stMetricLabel"] {opacity: 0.75;}
      .stTabs [data-baseweb="tab"] {font-size: 1.0rem; font-weight: 600;}
      div[data-testid="stMetric"] {background: rgba(150,150,150,0.08);
          border-radius: 10px; padding: 8px 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

TIER_COLORS = {"Critical": "#b71c1c", "High": "#e65100",
               "Medium": "#f9a825", "Low": "#2e7d32"}
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# --------------------------------------------------------------------------- #
# Loaders (cached + parallel, off the main thread)
# --------------------------------------------------------------------------- #
def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def _read_points(_path=None):
    if not os.path.exists(C.CLEAN_PARQUET):
        return None
    cols = ["latitude", "longitude", "congestion_weight", "hour", "dow",
            "police_station", "vehicle_type", "primary_violation",
            "at_junction", "is_peak", "is_parking", "date",
            "hotspot_id", "cell_id", "cell_lat", "cell_lon"]
    return pd.read_parquet(C.CLEAN_PARQUET, columns=cols)


@st.cache_data(show_spinner="Loading parking-intelligence artefacts…")
def load_all():
    """Load every artefact concurrently on worker threads (not the main thread)."""
    specs = {
        "kpis": (_read_json, C.KPI_JSON),
        "hotspots": (_read_csv, C.HOTSPOTS_CSV),
        "temporal": (_read_json, C.TEMPORAL_JSON),
        "plan": (_read_csv, C.ENFORCEMENT_CSV),
        "risk_metrics": (_read_json, C.RISK_METRICS_JSON),
        "points": (_read_points, C.CLEAN_PARQUET),
    }
    out = {}
    with ThreadPoolExecutor(max_workers=len(specs)) as ex:
        futs = {k: ex.submit(fn, path) for k, (fn, path) in specs.items()}
        for k, fut in futs.items():
            out[k] = fut.result()
    return out


@st.cache_data(show_spinner=False)
def load_csv(path):
    return _read_csv(path)


@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(C.RISK_MODEL_PKL):
        return None
    with open(C.RISK_MODEL_PKL, "rb") as f:
        return pickle.load(f)


_ART = load_all()
kpis = _ART["kpis"]
hotspots = _ART["hotspots"]
temporal = _ART["temporal"]
plan = _ART["plan"]
risk_metrics = _ART["risk_metrics"]


def load_points():
    return _ART["points"]


if kpis is None or hotspots is None:
    st.title("🅿️ ParkSense AI")
    st.warning("Artefacts not found. Run the pipeline first:\n\n"
               "```\npython src/build_all.py\n```")
    st.stop()


# --------------------------------------------------------------------------- #
# Live KPIs - recomputed on the fly from the loaded data (not a baked JSON)
# --------------------------------------------------------------------------- #
def compute_live_kpis(points, hs, top_k: int = 20) -> dict:
    k = {"top_k": top_k}
    total_sev = float(hs["severity_sum"].sum())
    if points is not None and len(points):
        cols = points.columns
        k["total_violations"] = int(len(points))
        k["parking_violations"] = int(points["is_parking"].sum()) if "is_parking" in cols else int(len(points))
        k["n_police_stations"] = int(points["police_station"].nunique()) if "police_station" in cols else 0
        k["junction_pct"] = round(float(points["at_junction"].mean()) * 100, 1) if "at_junction" in cols else 0.0
        k["peak_pct"] = round(float(points["is_peak"].mean()) * 100, 1) if "is_peak" in cols else 0.0
        k["date_start"] = str(points["date"].min()) if "date" in cols else "-"
        k["date_end"] = str(points["date"].max()) if "date" in cols else "-"
        if "congestion_weight" in cols:
            total_sev = float(points["congestion_weight"].sum())
    else:
        k.update(total_violations=int(hs["n_violations"].sum()), parking_violations=0,
                 n_police_stations=int(hs["police_station"].nunique()),
                 junction_pct=0.0, peak_pct=0.0, date_start="-", date_end="-")
    k["n_hotspots"] = int(len(hs))
    k["n_critical"] = int((hs["tier"] == "Critical").sum())
    k["n_high"] = int((hs["tier"] == "High").sum())
    k_eff = min(top_k, len(hs)) if len(hs) else top_k
    k["top_k"] = k_eff

    # Concentration is measured against the violations *in scope* (via their
    # hotspot_id) so it can never exceed 100% - a hotspot's cluster total can
    # otherwise include points that belong to a neighbouring station.
    top_ids = set(hs.nlargest(k_eff, "cis")["hotspot_id"].tolist()) if k_eff else set()
    if (points is not None and total_sev
            and {"hotspot_id", "congestion_weight"}.issubset(points.columns)):
        cw = points["congestion_weight"]
        in_top = float(cw[points["hotspot_id"].isin(top_ids)].sum())
        in_any = float(cw[points["hotspot_id"] >= 0].sum())
        k["top_impact_pct"] = round(min(100.0, 100 * in_top / total_sev), 1)
        k["severity_in_hotspots_pct"] = round(min(100.0, 100 * in_any / total_sev), 1)
    else:
        hs_total = float(hs["severity_sum"].sum()) or 1.0
        k["top_impact_pct"] = (round(100 * float(hs.nlargest(k_eff, "cis")["severity_sum"].sum())
                                     / hs_total, 1) if k_eff else 0.0)
        k["severity_in_hotspots_pct"] = 100.0
    return k


# --------------------------------------------------------------------------- #
# Sidebar filters (defined first so every KPI below reacts to them live)
# --------------------------------------------------------------------------- #
st.sidebar.header("Filters")
tiers = st.sidebar.multiselect("Hotspot tier",
                               ["Critical", "High", "Medium", "Low"],
                               default=["Critical", "High", "Medium"])
stations = ["(all)"] + sorted(hotspots["police_station"].dropna().unique().tolist())
station = st.sidebar.selectbox("Police station", stations)
scope = "All Bengaluru" if station == "(all)" else station

# Scope the data to the chosen police station so the KPIs recompute live
points_df = load_points()
if station != "(all)":
    hs_scope = hotspots[hotspots["police_station"] == station]
    pts_scope = (points_df[points_df["police_station"] == station]
                 if points_df is not None and "police_station" in points_df.columns
                 else points_df)
else:
    hs_scope, pts_scope = hotspots, points_df

# the tier filter refines only the map / list view
hs_view = hs_scope[hs_scope["tier"].isin(tiers)] if tiers else hs_scope

live = compute_live_kpis(pts_scope, hs_scope)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Scope:** {scope}  \n"
    f"**Data:** {live['date_start']} → {live['date_end']}  \n"
    f"**Stations in scope:** {live['n_police_stations']}  \n"
    f"**Risk model MAE:** {kpis['risk_model']['mae']} "
    f"(beats baseline by {kpis['risk_model']['mae_improvement_pct']:.1f}%)"
)


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🅿️ ParkSense AI — Parking-Induced Congestion Intelligence")
st.caption("Bengaluru Traffic Police · Detect illegal-parking hotspots · "
           "Quantify traffic-flow impact · Target enforcement")

c = st.columns(6)
c[0].metric("Violations", f"{live['total_violations']:,}")
c[1].metric("Hotspots", live["n_hotspots"])
c[2].metric("Critical zones", live["n_critical"])
c[3].metric("At junctions", f"{live['junction_pct']}%")
c[4].metric("In rush-hour", f"{live['peak_pct']}%")
c[5].metric(f"Top-{live['top_k']} impact", f"{live['top_impact_pct']}%")
st.caption(f"↻ Live from {live['total_violations']:,} records · scope: **{scope}** · "
           f"{live['n_hotspots']} hotspots · "
           f"{live['severity_in_hotspots_pct']}% of impact inside hotspots")

_k = live["top_k"]
_hs_phrase = "the top hotspot" if _k == 1 else f"the top {_k} hotspots"
_verb = "concentrates" if _k == 1 else "concentrate"
st.info(
    f"**Insight:** in **{scope}**, {_hs_phrase} {_verb} "
    f"**{live['top_impact_pct']}% of all congestion impact** — "
    f"target a handful of zones instead of patrolling everywhere.",
    icon="🎯",
)


# --------------------------------------------------------------------------- #
# Fragments - isolate heavy / interactive widgets so a map zoom or a what-if
# selection only re-renders that fragment, never the whole application.
# --------------------------------------------------------------------------- #
@st.fragment
def render_hotspot_map(hs_view):
    import folium
    from folium.plugins import HeatMap
    from streamlit_folium import st_folium

    pts = load_points()
    m = folium.Map(location=list(C.CITY_CENTER), zoom_start=12,
                   tiles="CartoDB positron")
    if pts is not None:
        samp = pts.sample(min(25000, len(pts)), random_state=1)
        HeatMap(samp[["latitude", "longitude", "congestion_weight"]].values.tolist(),
                radius=9, blur=12, min_opacity=0.25).add_to(
            folium.FeatureGroup(name="Impact heatmap").add_to(m))
    hg = folium.FeatureGroup(name="Hotspots").add_to(m)
    for r in hs_view.itertuples(index=False):
        col = TIER_COLORS.get(r.tier, "#555")
        folium.Circle([r.lat, r.lon], radius=max(60, float(r.radius_m)),
                      color=col, fill=True, fill_color=col, fill_opacity=0.35,
                      weight=1, tooltip=f"#{int(r.rank)} {r.tier} · CIS {r.cis}").add_to(hg)
        folium.CircleMarker([r.lat, r.lon], radius=4, color=col, fill=True,
                            fill_opacity=0.9, popup=folium.Popup(
                                f"<b>#{int(r.rank)} {r.tier}</b><br>"
                                f"CIS {r.cis}/100<br>{r.junction_name or r.police_station}"
                                f"<br>{int(r.n_violations):,} violations"
                                f"<br>{r.top_violations}", max_width=300)).add_to(hg)
    folium.LayerControl(collapsed=False).add_to(m)
    # returned_objects=[] -> pan/zoom stop streaming state back, so no app rerun
    st_folium(m, height=560, use_container_width=True, returned_objects=[])


@st.fragment
def render_whatif():
    model_bundle = load_model()
    cell_stats = load_csv(C.CELL_FEATURES_CSV)
    if model_bundle is None or cell_stats is None:
        st.info("Risk-model artefacts not found — run `python src/build_all.py`.")
        return
    cc = st.columns(3)
    day = cc[0].selectbox("Day", DOW, index=0)
    bucket = cc[1].selectbox("Time window", model_bundle["buckets"], index=1)
    topk = cc[2].slider("Show top-N risk cells", 10, 100, 30, step=10)

    d_idx = DOW.index(day)
    cs = cell_stats[(cell_stats["dow"] == d_idx)
                    & (cell_stats["bucket"] == bucket)].copy()
    feats = model_bundle["features"]
    cs["pred"] = np.clip(model_bundle["model"].predict(cs[feats]), 0, None)
    top_cells = cs.nlargest(topk, "pred")
    try:
        import folium
        from streamlit_folium import st_folium
        mm = folium.Map(location=list(C.CITY_CENTER), zoom_start=12,
                        tiles="CartoDB positron")
        vmax = float(top_cells["pred"].max()) or 1.0
        for r in top_cells.itertuples(index=False):
            folium.CircleMarker([r.cell_lat, r.cell_lon],
                                radius=4 + 10 * r.pred / vmax,
                                color="#c62828", fill=True, fill_opacity=0.6,
                                tooltip=f"predicted load {r.pred:.1f}").add_to(mm)
        st_folium(mm, height=460, use_container_width=True, returned_objects=[])
    except Exception:
        st.dataframe(top_cells[["cell_lat", "cell_lon", "pred"]],
                     hide_index=True, use_container_width=True)
    st.caption(f"Predicted severity-weighted parking load for **{day}, {bucket}** "
               f"across {len(cs):,} enforcement cells.")


tab_map, tab_impact, tab_time, tab_risk, tab_plan = st.tabs(
    ["🗺️ Hotspot Map", "📊 Congestion Impact", "🕒 When & Where",
     "🔮 Predictive Risk", "🚓 Enforcement Plan"]
)


# --------------------------------------------------------------------------- #
# Tab 1 - Map
# --------------------------------------------------------------------------- #
with tab_map:
    st.subheader("Congestion-impact heatmap & ranked hotspots")
    left, right = st.columns([3, 2])
    with left:
        render_hotspot_map(hs_view)
    with right:
        st.markdown("**Top priority hotspots**")
        st.dataframe(
            hs_view.head(12)[["rank", "tier", "cis", "police_station",
                              "junction_name", "n_violations", "peak_hour"]],
            hide_index=True, use_container_width=True, height=520,
        )


# --------------------------------------------------------------------------- #
# Tab 2 - Congestion impact
# --------------------------------------------------------------------------- #
with tab_impact:
    st.subheader("Congestion Impact Score — what drives each hotspot")
    st.markdown(
        "CIS blends five auditable components: **volume**, **severity** "
        "(how flow-choking the violations are), **junction** criticality, "
        "**persistence** (chronic vs one-off) and **peak**-hour concentration."
    )
    import plotly.express as px

    topn = hs_view.head(15).copy()
    topn["label"] = "#" + topn["rank"].astype(str) + " " + topn["police_station"].astype(str)
    fig = px.bar(topn.sort_values("cis"), x="cis", y="label", color="tier",
                 color_discrete_map=TIER_COLORS, orientation="h",
                 labels={"cis": "Congestion Impact Score", "label": ""},
                 height=520)
    st.plotly_chart(fig, use_container_width=True)

    comp_cols = ["c_volume", "c_severity", "c_junction", "c_persistence", "c_peak"]
    if set(comp_cols).issubset(hs_view.columns):
        sel = st.selectbox("Inspect a hotspot's score breakdown",
                           hs_view["rank"].head(20))
        row = hs_view[hs_view["rank"] == sel].iloc[0]
        comp = pd.DataFrame({
            "component": ["Volume", "Severity", "Junction", "Persistence", "Peak"],
            "score": [row[c] for c in comp_cols],
        })
        st.plotly_chart(px.line_polar(comp, r="score", theta="component",
                                      line_close=True, range_r=[0, 100],
                                      height=380), use_container_width=True)
        st.dataframe(hs_view.head(25)[["rank", "tier", "cis", "police_station",
                     "junction_name", "n_violations", "severity_mean",
                     "junction_share", "active_days", "top_violations"]],
                     hide_index=True, use_container_width=True)


# --------------------------------------------------------------------------- #
# Tab 3 - Temporal
# --------------------------------------------------------------------------- #
with tab_time:
    st.subheader("When do violations happen?")
    import plotly.express as px
    import plotly.graph_objects as go

    if temporal:
        mat = np.array(temporal["hour_dow_matrix"])
        fig = go.Figure(go.Heatmap(z=mat, x=list(range(24)),
                                   y=temporal["dow_labels"], colorscale="Inferno"))
        fig.update_layout(height=320, xaxis_title="Hour of day",
                          title="Violations by hour × day-of-week")
        st.plotly_chart(fig, use_container_width=True)

        a, b = st.columns(2)
        byh = pd.Series(temporal["by_hour"]).reset_index()
        byh.columns = ["hour", "impact"]
        a.plotly_chart(px.bar(byh, x="hour", y="impact",
                              title="Severity-weighted impact by hour", height=300),
                       use_container_width=True)
        bym = pd.Series(temporal["by_month"]).reset_index()
        bym.columns = ["month", "violations"]
        b.plotly_chart(px.line(bym, x="month", y="violations", markers=True,
                               title="Monthly trend", height=300),
                       use_container_width=True)

        a2, b2 = st.columns(2)
        veh = pd.Series(temporal["vehicle_mix"]).reset_index()
        veh.columns = ["vehicle", "n"]
        a2.plotly_chart(px.bar(veh, x="n", y="vehicle", orientation="h",
                               title="Top offending vehicle types", height=320),
                        use_container_width=True)
        vio = pd.Series(temporal["violation_mix"]).reset_index()
        vio.columns = ["violation", "n"]
        b2.plotly_chart(px.bar(vio, x="n", y="violation", orientation="h",
                               title="Violation-type mix", height=320),
                        use_container_width=True)


# --------------------------------------------------------------------------- #
# Tab 4 - Predictive risk
# --------------------------------------------------------------------------- #
with tab_risk:
    st.subheader("Predictive risk model — from reactive to proactive")
    if risk_metrics:
        m = risk_metrics
        k = st.columns(4)
        k[0].metric("Test MAE", m["mae"])
        k[1].metric("Beats baseline", f"{m['mae_improvement_pct']:.1f}%")
        k[2].metric("R²", m["r2"])
        k[3].metric("Top-10% capture", f"{m['load_capture_top10pct']*100:.0f}%")
        st.caption(f"Time-based hold-out after {m['cutoff_date']} — the model is "
                   "validated on *future* days it never saw during training.")

        import plotly.express as px
        imp = pd.Series(m["feature_importance"]).sort_values().reset_index()
        imp.columns = ["feature", "importance"]
        st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h",
                               title="What the model uses to predict risk",
                               height=380), use_container_width=True)

    st.markdown("#### What-if: predict tomorrow's hotspots")
    render_whatif()


# --------------------------------------------------------------------------- #
# Tab 5 - Enforcement plan
# --------------------------------------------------------------------------- #
with tab_plan:
    st.subheader("Targeted patrol deployment plan")
    cov = kpis.get("coverage", {})
    k = st.columns(4)
    k[0].metric("Patrol units", cov.get("units_deployed", "-"))
    k[1].metric("Impact covered", f"{cov.get('severity_covered_pct', '-')}%")
    k[2].metric("Critical+High zones", cov.get("critical_high_zones", "-"))
    k[3].metric("Their violation share",
                f"{cov.get('critical_high_violation_share_pct', '-')}%")
    st.markdown(
        f"Deploying **{cov.get('units_deployed','?')} units** to the top zones "
        f"covers **{cov.get('severity_covered_pct','?')}% of hotspot congestion "
        "impact** — each unit gets a specific zone, time window and day focus.")
    if plan is not None:
        st.dataframe(plan, hide_index=True, use_container_width=True)
        st.download_button("⬇️ Download patrol plan (CSV)",
                           plan.to_csv(index=False).encode("utf-8"),
                           "parksense_patrol_plan.csv", "text/csv")
