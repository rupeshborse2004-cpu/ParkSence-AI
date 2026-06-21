"""
ParkSense AI - one-shot pipeline runner.

    python src/build_all.py

Runs every stage in order and writes all artefacts the dashboard reads:
clean parquet, hotspots+CIS, temporal profiles, trained risk model, enforcement
plan, KPI summary, Folium maps and figures.
"""
from __future__ import annotations

import json
import os
import sys
import time

# make `import config`, `import data_pipeline`, ... work from anywhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as C
import data_pipeline as dp
import hotspot_detection as hd
import congestion_impact as ci
import forecasting as fc
import risk_model as rm
import enforcement as en
import visualize as viz


def main() -> None:
    t0 = time.time()
    print("=" * 64)
    print(" ParkSense AI - building parking-intelligence artefacts")
    print("=" * 64)

    # 1. clean
    df = dp.build()

    # 2. hotspots
    df, hotspots = hd.detect(df)

    # 3. congestion impact score
    hotspots = ci.score(hotspots)
    hotspots.to_csv(C.HOTSPOTS_CSV, index=False)
    conc = ci.concentration_stats(df, hotspots)
    print(f"[cis] {len(hotspots)} hotspots | "
          f"top{conc['top_k']} hold {conc[f'severity_in_top'+str(conc['top_k'])+'_pct']}% of impact")

    # persist per-violation hotspot tag (for the dashboard map)
    df_out = df.copy()
    df_out["violations"] = df_out["violations"].map(json.dumps)
    df_out.to_parquet(C.CLEAN_PARQUET, index=False)

    # 4. temporal profiles
    profiles = fc.build_profiles(df, hotspots)
    fc.save(profiles)

    # 5. risk model
    metrics = rm.train(df)

    # 6. enforcement plan
    plan = en.build_plan(hotspots, profiles["hotspot_windows"])
    plan.to_csv(C.ENFORCEMENT_CSV, index=False)
    coverage = en.coverage_summary(hotspots, plan)
    print(f"[plan] {coverage['units_deployed']} units cover "
          f"{coverage['severity_covered_pct']}% of hotspot impact")

    # 7. maps + figures
    viz.build_maps(df, hotspots)
    viz.build_figures(df, hotspots, profiles)

    # 8. KPIs
    kpis = {
        "total_violations": int(len(df)),
        "parking_violations": int(df["is_parking"].sum()),
        "date_start": str(df["date"].min()),
        "date_end": str(df["date"].max()),
        "n_police_stations": int(df["police_station"].nunique()),
        "n_junctions": int(
            df.loc[df["junction_name"].str.lower() != "no junction",
                   "junction_name"].nunique()
        ),
        "n_hotspots": int(len(hotspots)),
        "n_critical": int((hotspots["tier"] == "Critical").sum()),
        "n_high": int((hotspots["tier"] == "High").sum()),
        "junction_violation_share_pct": round(float(df["at_junction"].mean()) * 100, 1),
        "peak_violation_share_pct": round(float(df["is_peak"].mean()) * 100, 1),
        "concentration": conc,
        "coverage": coverage,
        "risk_model": {k: v for k, v in metrics.items() if k != "feature_importance"},
        "top_feature": list(metrics["feature_importance"])[0],
    }
    with open(C.KPI_JSON, "w", encoding="utf-8") as f:
        json.dump(kpis, f, indent=2)

    dt = time.time() - t0
    print("-" * 64)
    print(f" DONE in {dt:0.1f}s. Artefacts -> {C.OUT_DIR}")
    print(f"  violations={kpis['total_violations']:,}  hotspots={kpis['n_hotspots']}  "
          f"critical={kpis['n_critical']}  high={kpis['n_high']}")
    print(f"  risk model: MAE={metrics['mae']} R2={metrics['r2']} "
          f"({metrics['mae_improvement_pct']:+.1f}% vs baseline)")
    print(f"  next:  streamlit run app/dashboard.py")
    print("=" * 64)


if __name__ == "__main__":
    main()
