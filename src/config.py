"""
ParkSense AI - Central configuration.

Single source of truth for paths, the spatial/temporal grid, clustering
parameters and - most importantly - the Congestion Impact Weights that turn a
raw parking violation into a quantified traffic-flow impact.
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_CSV = os.path.join(ROOT, "jan to may police violation_anonymized791b166.csv")

OUT_DIR = os.path.join(ROOT, "outputs")
PROC_DIR = os.path.join(OUT_DIR, "processed")
MAPS_DIR = os.path.join(OUT_DIR, "maps")
FIG_DIR = os.path.join(OUT_DIR, "figures")
MODEL_DIR = os.path.join(OUT_DIR, "models")

for _d in (OUT_DIR, PROC_DIR, MAPS_DIR, FIG_DIR, MODEL_DIR):
    os.makedirs(_d, exist_ok=True)

# Processed artefacts re-used across modules + the dashboard
CLEAN_PARQUET = os.path.join(PROC_DIR, "violations_clean.parquet")
HOTSPOTS_CSV = os.path.join(PROC_DIR, "hotspots.csv")
GRID_CSV = os.path.join(PROC_DIR, "grid_cells.csv")
ENFORCEMENT_CSV = os.path.join(PROC_DIR, "enforcement_plan.csv")
CELL_FEATURES_CSV = os.path.join(PROC_DIR, "cell_features.csv")
TEMPORAL_JSON = os.path.join(PROC_DIR, "temporal_profile.json")
KPI_JSON = os.path.join(PROC_DIR, "kpis.json")
RISK_MODEL_PKL = os.path.join(MODEL_DIR, "risk_lgbm.pkl")
RISK_METRICS_JSON = os.path.join(MODEL_DIR, "risk_metrics.json")

# --------------------------------------------------------------------------- #
# Geography (Bengaluru) & time
# --------------------------------------------------------------------------- #
# Valid bounding box - drops GPS noise / out-of-city points.
LAT_MIN, LAT_MAX = 12.70, 13.40
LON_MIN, LON_MAX = 77.30, 77.90
CITY_CENTER = (12.9716, 77.5946)  # MG Road area, used to centre maps

LOCAL_TZ = "Asia/Kolkata"  # raw timestamps are UTC (+00); analysis needs IST
EARTH_RADIUS_M = 6_371_000.0

# Peak windows (IST) used for temporal-concentration scoring & patrol planning.
MORNING_PEAK = (8, 11)   # 08:00-10:59
EVENING_PEAK = (17, 21)  # 17:00-20:59

# Hour -> coarse bucket used by the spatio-temporal risk model.
HOUR_BUCKETS = {
    "night":        list(range(0, 7)),    # 00-06
    "morning_peak": list(range(7, 11)),   # 07-10
    "midday":       list(range(11, 16)),  # 11-15
    "evening_peak": list(range(16, 21)),  # 16-20
    "late":         list(range(21, 24)),  # 21-23
}

# --------------------------------------------------------------------------- #
# Spatial grids
# --------------------------------------------------------------------------- #
# ~0.001 deg latitude ~= 111 m. Choose cell sizes accordingly.
HOTSPOT_SNAP_DEG = 0.00035   # ~39 m fine grid for DBSCAN aggregation
RISK_CELL_DEG = 0.0025       # ~278 m enforcement grid for the risk model

# DBSCAN (haversine, radians). eps_m converted to radians at runtime.
DBSCAN_EPS_M = 140.0         # neighbourhood radius
DBSCAN_MIN_WEIGHT = 35       # min weighted violations to seed a hotspot core

# --------------------------------------------------------------------------- #
# Congestion Impact Weights  (the heart of "quantify impact on traffic flow")
# --------------------------------------------------------------------------- #
# 0..1 score: how much each violation type chokes a *moving* carriageway or an
# intersection. Parking on a live main road / next to a signal is far worse for
# traffic flow than a number-plate offence. Derived from how the violation
# physically obstructs through-traffic and intersections.
CONGESTION_WEIGHTS: dict[str, float] = {
    "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS": 1.00,
    "PARKING NEAR ROAD CROSSING":                 0.95,
    "PARKING IN A MAIN ROAD":                     0.95,
    "DOUBLE PARKING":                             0.90,
    "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE": 0.85,
    "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC":    0.75,
    "PARKING OTHER THAN BUS STOP":                0.65,
    "WRONG PARKING":                              0.60,
    "NO PARKING":                                 0.55,
    "PARKING ON FOOTPATH":                        0.40,
}

# Violation strings that are genuinely parking / standing related. Everything
# else (defective number plate, no helmet, mobile phone ...) is non-parking and
# kept only for context.
PARKING_VIOLATIONS = set(CONGESTION_WEIGHTS.keys())

# Weight applied to non-parking violations so they never dominate impact.
NON_PARKING_WEIGHT = 0.10

# Validation states to drop as not-a-real-violation.
DROP_VALIDATION = {"rejected", "duplicate"}

# --------------------------------------------------------------------------- #
# Congestion Impact Score (CIS) component weights  (sum = 1.0)
# --------------------------------------------------------------------------- #
CIS_WEIGHTS = {
    "volume":      0.30,  # how many violations (log-scaled)
    "severity":    0.25,  # mean congestion weight of those violations
    "junction":    0.20,  # share occurring at signalised junctions
    "persistence": 0.15,  # active across many distinct days (chronic vs one-off)
    "peak":        0.10,  # share during rush-hour windows
}

CIS_TIERS = [  # (lower_bound, label) - evaluated high -> low
    (75, "Critical"),
    (55, "High"),
    (35, "Medium"),
    (0,  "Low"),
]

# Enforcement planning defaults
PATROL_UNITS = 12            # mobile enforcement teams available per shift
PATROL_SHIFT_HOURS = 3       # length of a deployment window
