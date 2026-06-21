"""
Stage 5 - Spatio-temporal risk model (LightGBM, Tweedie) -- enhanced.

Attacks the "enforcement is reactive" pain point. We learn the **expected
severity-weighted parking load** of every 278 m cell for each
(day-of-week x time-window) - the surface a planner needs to pre-position
patrols.

Feature design follows the spatio-temporal literature:
  * ST-ResNet (Zhang, Zheng & Qi, AAAI 2017) decomposes spatio-temporal signal
    into CLOSENESS + PERIOD + TREND + EXTERNAL factors.
  * Tobler's First Law / spatial autocorrelation (Moran's I, Getis-Ord Gi*):
    "near things are more related" -> spatial-lag (neighbour) features.

So each (cell, dow, bucket) row carries:
  - PERIOD   : cyclical sin/cos of weekday & time-of-day, weekend flag
  - CLOSENESS: cell history, cell x bucket mean, cell x dow mean
  - TREND    : early-vs-late half load change
  - SPATIAL  : neighbour load, neighbour-bucket load, local density,
               Getis-Ord Gi* z-score, distance to city centre
  - CONTEXT  : severity, junction share, parking share, peak share, vehicle /
               violation diversity

Validation is honest: features + target are built on the FIRST 80% of dates and
the model is scored on the LAST 20% (future weeks). Hyper-parameters are tuned by
randomised search with early stopping. Compared to a fair per-cell baseline.
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config as C

BUCKETS = list(C.HOUR_BUCKETS.keys())
BUCKET_IDX = {b: i for i, b in enumerate(BUCKETS)}
# representative centre-hour of each bucket, for cyclical time-of-day encoding
BUCKET_HOUR = {"night": 3.0, "morning_peak": 8.5, "midday": 13.0,
               "evening_peak": 18.5, "late": 22.5}

MIN_CELL = 20          # min violations in a cell to model it
N_SEARCH = 10          # randomised hyper-parameter trials

FEATURES = [
    # spatial position
    "cell_lat", "cell_lon", "dist_center",
    # period (external/time)
    "bucket_idx", "dow", "is_weekend", "dow_sin", "dow_cos", "hour_sin", "hour_cos",
    # closeness (cell behaviour history)
    "cell_total", "cell_hist_load", "cell_sev", "cell_junction",
    "cell_parking_share", "cell_peak_share", "cell_vehicle_div", "cell_viol_div",
    "cell_bucket_mean", "cell_dow_mean",
    # trend
    "cell_trend",
    # spatial autocorrelation
    "neigh_load", "neigh_bucket_load", "local_density", "getis_g",
]
TARGET = "exp_load"


def _neighbour_ids(lat: float, lon: float, step: float) -> list[str]:
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            out.append(f"{round(lat + dy * step, 5)}_{round(lon + dx * step, 5)}")
    return out


def _static_features(df: pd.DataFrame, dates) -> pd.DataFrame:
    """Cell-level (dow/bucket-independent) features over a date window."""
    sub = df[df["date"].isin(set(dates))]
    dts = sorted(set(dates))
    nday = max(1, len(dts))
    stat = (
        sub.groupby("cell_id")
        .agg(cell_lat=("cell_lat", "first"), cell_lon=("cell_lon", "first"),
             cell_total=("congestion_weight", "size"),
             cell_sev=("congestion_weight", "mean"),
             cell_junction=("at_junction", "mean"),
             cell_parking_share=("is_parking", "mean"),
             cell_peak_share=("is_peak", "mean"),
             cell_vehicle_div=("vehicle_type", "nunique"),
             cell_viol_div=("primary_violation", "nunique"),
             load_sum=("congestion_weight", "sum"))
        .reset_index()
    )
    stat["cell_hist_load"] = stat["load_sum"] / (nday * len(BUCKETS))
    stat["dist_center"] = np.sqrt(
        ((stat["cell_lat"] - C.CITY_CENTER[0]) * 111.32) ** 2
        + ((stat["cell_lon"] - C.CITY_CENTER[1]) * 111.32
           * np.cos(np.radians(stat["cell_lat"]))) ** 2
    )

    # TREND: early vs late half rate (ST-ResNet "trend")
    mid = dts[len(dts) // 2]
    n_early = max(1, sum(d <= mid for d in dts))
    n_late = max(1, sum(d > mid for d in dts))
    early = sub[sub["date"] <= mid].groupby("cell_id")["congestion_weight"].sum()
    late = sub[sub["date"] > mid].groupby("cell_id")["congestion_weight"].sum()
    stat["cell_trend"] = (stat["cell_id"].map(late).fillna(0) / n_late
                          - stat["cell_id"].map(early).fillna(0) / n_early)

    # SPATIAL autocorrelation: neighbour load, local density, Getis-Ord Gi*
    step = C.RISK_CELL_DEG
    tot = dict(zip(stat["cell_id"], stat["cell_total"]))
    hist = dict(zip(stat["cell_id"], stat["cell_hist_load"]))
    neigh_load, local_density = [], []
    for r in stat.itertuples(index=False):
        s, hl = r.cell_total, []
        for nid in _neighbour_ids(r.cell_lat, r.cell_lon, step):
            if nid in tot:
                s += tot[nid]
                hl.append(hist[nid])
        neigh_load.append(float(np.mean(hl)) if hl else r.cell_hist_load)
        local_density.append(s)
    stat["neigh_load"] = neigh_load
    stat["local_density"] = local_density
    gmean = stat["cell_total"].mean()
    gstd = stat["cell_total"].std() or 1.0
    stat["getis_g"] = (stat["local_density"] / 9.0 - gmean) / (gstd / 3.0)
    return stat.drop(columns="load_sum")


def _bucket_means(df: pd.DataFrame, dates) -> pd.DataFrame:
    """Per (cell, bucket) expected load + its spatial-lag (neighbour) value."""
    sub = df[df["date"].isin(set(dates))].copy()
    sub["bucket"] = sub["hour_bucket"]
    nday = max(1, len(set(dates)))
    bm = (
        sub.groupby(["cell_id", "bucket"])
        .agg(load_sum=("congestion_weight", "sum"),
             cell_lat=("cell_lat", "first"), cell_lon=("cell_lon", "first"))
        .reset_index()
    )
    bm["cell_bucket_mean"] = bm["load_sum"] / nday
    step = C.RISK_CELL_DEG
    val = {(c, b): v for c, b, v in
           zip(bm["cell_id"], bm["bucket"], bm["cell_bucket_mean"])}
    nb = []
    for r in bm.itertuples(index=False):
        vals = [val[(nid, r.bucket)]
                for nid in _neighbour_ids(r.cell_lat, r.cell_lon, step)
                if (nid, r.bucket) in val]
        nb.append(float(np.mean(vals)) if vals else r.cell_bucket_mean)
    bm["neigh_bucket_load"] = nb
    return bm[["cell_id", "bucket", "cell_bucket_mean", "neigh_bucket_load"]]


def _dow_means(df: pd.DataFrame, dates) -> pd.DataFrame:
    sub = df[df["date"].isin(set(dates))].copy()
    sub["dow"] = pd.to_datetime(sub["date"]).dt.dayofweek
    d = pd.to_datetime(pd.Series(sorted(set(dates))))
    days_per_dow = d.dt.dayofweek.value_counts().to_dict()
    dm = (sub.groupby(["cell_id", "dow"])
          .agg(load_sum=("congestion_weight", "sum")).reset_index())
    dm["n"] = dm["dow"].map(days_per_dow).clip(lower=1)
    dm["cell_dow_mean"] = dm["load_sum"] / dm["n"]
    return dm[["cell_id", "dow", "cell_dow_mean"]]


def _expected_load(df: pd.DataFrame, dates) -> pd.DataFrame:
    """Target: mean severity-weighted load per (cell, dow, bucket)."""
    sub = df[df["date"].isin(set(dates))].copy()
    sub["bucket"] = sub["hour_bucket"]
    sub["dow"] = pd.to_datetime(sub["date"]).dt.dayofweek
    d = pd.to_datetime(pd.Series(sorted(set(dates))))
    days_per_dow = d.dt.dayofweek.value_counts().to_dict()
    g = (sub.groupby(["cell_id", "dow", "bucket"])
         .agg(load_sum=("congestion_weight", "sum")).reset_index())
    g["n"] = g["dow"].map(days_per_dow).clip(lower=1)
    g["exp_load"] = g["load_sum"] / g["n"]
    return g[["cell_id", "dow", "bucket", "exp_load"]]


def _assemble(active_ids, static, bmeans, dmeans, target=None) -> pd.DataFrame:
    """Full (cell x dow x bucket) grid joined with all features (+ target)."""
    cells = static[static["cell_id"].isin(active_ids)][["cell_id"]]
    grid = (cells.assign(k=1)
            .merge(pd.DataFrame({"dow": range(7), "k": 1}), on="k")
            .merge(pd.DataFrame({"bucket": BUCKETS, "k": 1}), on="k")
            .drop(columns="k"))
    grid = (grid.merge(static, on="cell_id", how="left")
            .merge(bmeans, on=["cell_id", "bucket"], how="left")
            .merge(dmeans, on=["cell_id", "dow"], how="left"))
    for col in ("cell_bucket_mean", "neigh_bucket_load", "cell_dow_mean"):
        grid[col] = grid[col].fillna(0.0)
    grid["bucket_idx"] = grid["bucket"].map(BUCKET_IDX)
    grid["is_weekend"] = (grid["dow"] >= 5).astype(int)
    grid["dow_sin"] = np.sin(2 * np.pi * grid["dow"] / 7)
    grid["dow_cos"] = np.cos(2 * np.pi * grid["dow"] / 7)
    hr = grid["bucket"].map(BUCKET_HOUR)
    grid["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    grid["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    if target is not None:
        grid = grid.merge(target, on=["cell_id", "dow", "bucket"], how="left")
        grid["exp_load"] = grid["exp_load"].fillna(0.0)
    return grid


PARAM_GRID = {
    "learning_rate": [0.02, 0.03, 0.05],
    "num_leaves": [31, 63, 127],
    "min_child_samples": [20, 40, 80],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "reg_lambda": [0.0, 1.0, 5.0],
}


def _search(tr: pd.DataFrame, rng, log) -> tuple[dict, int]:
    """Randomised hyper-parameter search with early stopping on a held-out split."""
    idx = rng.permutation(len(tr))
    vcut = int(0.85 * len(tr))
    fit, val = tr.iloc[idx[:vcut]], tr.iloc[idx[vcut:]]
    best, best_params, best_iter = np.inf, None, 400
    for _ in range(N_SEARCH):
        params = {k: rng.choice(v).item() for k, v in PARAM_GRID.items()}
        m = LGBMRegressor(objective="tweedie", tweedie_variance_power=1.2,
                          n_estimators=1500, random_state=42, n_jobs=-1,
                          verbose=-1, **params)
        m.fit(fit[FEATURES], fit[TARGET],
              eval_set=[(val[FEATURES], val[TARGET])], eval_metric="l1",
              callbacks=[early_stopping(50, verbose=False)])
        vp = np.clip(m.predict(val[FEATURES], num_iteration=m.best_iteration_), 0, None)
        vmae = mean_absolute_error(val[TARGET], vp)
        if vmae < best:
            best, best_params, best_iter = vmae, params, (m.best_iteration_ or 400)
    log(f"[risk] tuned {N_SEARCH} configs -> val MAE {best:.3f}, best_iter {best_iter}, "
        f"lr {best_params['learning_rate']}, leaves {best_params['num_leaves']}")
    return best_params, best_iter


def train(df: pd.DataFrame, verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    df = df.copy()
    rng = np.random.default_rng(42)

    dates = np.array(sorted(df["date"].unique()))
    cut = int(len(dates) * 0.8)
    train_dates, test_dates = dates[:cut], dates[cut:]
    cutoff = test_dates[0]

    # features built ONLY on the past (train window) -> no leakage
    static = _static_features(df, train_dates)
    bmeans = _bucket_means(df, train_dates)
    dmeans = _dow_means(df, train_dates)
    active = static[static["cell_total"] >= MIN_CELL]["cell_id"]

    tr = _assemble(active, static, bmeans, dmeans, _expected_load(df, train_dates))
    te = _assemble(active, static, bmeans, dmeans, _expected_load(df, test_dates))
    log(f"[risk] cells={len(active):,}  train rows={len(tr):,}  "
        f"test rows={len(te):,}  feats={len(FEATURES)}  (future cutoff {cutoff})")

    best_params, best_iter = _search(tr, rng, log)
    model = LGBMRegressor(objective="tweedie", tweedie_variance_power=1.2,
                          n_estimators=int(best_iter * 1.1) + 50, random_state=42,
                          n_jobs=-1, verbose=-1, **best_params)
    model.fit(tr[FEATURES], tr[TARGET])

    pred = np.clip(model.predict(te[FEATURES]), 0, None)
    y = te[TARGET].to_numpy()
    mae = float(mean_absolute_error(y, pred))
    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    r2 = float(r2_score(y, pred))

    base_map = tr.groupby("cell_id")[TARGET].mean()
    base = te["cell_id"].map(base_map).fillna(base_map.mean()).to_numpy()
    base_mae = float(mean_absolute_error(y, base))
    base_rmse = float(np.sqrt(mean_squared_error(y, base)))

    order = np.argsort(-pred)
    k = max(1, int(len(y) * 0.10))
    capture = float(y[order[:k]].sum() / y.sum()) if y.sum() else 0.0
    base_order = np.argsort(-base)
    base_capture = float(y[base_order[:k]].sum() / y.sum()) if y.sum() else 0.0

    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)

    metrics = {
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "cutoff_date": str(cutoff), "n_cells": int(len(active)),
        "n_features": len(FEATURES),
        "mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4),
        "baseline_mae": round(base_mae, 4), "baseline_rmse": round(base_rmse, 4),
        "mae_improvement_pct": round(100 * (base_mae - mae) / base_mae, 1) if base_mae else 0,
        "rmse_improvement_pct": round(100 * (base_rmse - rmse) / base_rmse, 1) if base_rmse else 0,
        "load_capture_top10pct": round(capture, 3),
        "baseline_capture_top10pct": round(base_capture, 3),
        "best_params": best_params,
        "feature_importance": {k_: int(v) for k_, v in imp.items()},
    }

    with open(C.RISK_MODEL_PKL, "wb") as f:
        pickle.dump({"model": model, "features": FEATURES, "buckets": BUCKETS}, f)
    with open(C.RISK_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # serving table over ALL dates (latest knowledge), one row per (cell,dow,bucket)
    s_all = _static_features(df, dates)
    b_all = _bucket_means(df, dates)
    d_all = _dow_means(df, dates)
    act_all = s_all[s_all["cell_total"] >= MIN_CELL]["cell_id"]
    serve = _assemble(act_all, s_all, b_all, d_all, target=None)
    keep = ["cell_id", "bucket"] + [c for c in FEATURES if c not in ("cell_id", "bucket")]
    serve[keep].to_csv(C.CELL_FEATURES_CSV, index=False)

    log(f"[risk] MAE {mae:.3f} vs base {base_mae:.3f} (+{metrics['mae_improvement_pct']}%)  "
        f"R2 {r2:.3f}  top-10% capture {capture*100:.0f}% (base {base_capture*100:.0f}%)")
    return metrics


if __name__ == "__main__":
    import data_pipeline as dp

    d = dp.load_clean()
    m = train(d)
    print(json.dumps({k: v for k, v in m.items()
                      if k not in ("feature_importance", "best_params")}, indent=2))
    print("top features:", list(m["feature_importance"])[:8])
