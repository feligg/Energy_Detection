"""Validation without real labels: synthetic anomaly injection + sensitivity.

BDG2 has no confirmed anomaly labels, so precision/recall cannot be measured on
the real series. Instead we INJECT known anomalies into copies of real
buildings, re-run the detectors, and measure how many injected events each
method finds (recall) at three magnitudes. The share of untouched hours that
are flagged shows how noisy each method is.

The injected data keeps the original `expected_kwh` (the model of normal
behaviour), exactly as a detector would see it in operation.
"""
import numpy as np
import pandas as pd

from . import config as C
from . import anomaly_rules, detectors

INJ_TYPES = ["night", "weekend", "sudden", "sustained"]
SCORE_COLS = ["m_zscore", "m_iqr", "m_iforest", "m_residual", "final_anomaly"]


def _pick_buildings(df, n):
    rng = np.random.default_rng(C.RANDOM_STATE)
    ids = sorted(df["building_id"].unique())
    return list(rng.choice(ids, size=min(n, len(ids)), replace=False))


def _event_windows(g, kind, k, rng, taken):
    """Return k lists of row positions (non-overlapping) for one building."""
    hour, off, n = g["hour"].to_numpy(), g["is_off"].to_numpy(), len(g)
    length = {"night": 8, "weekend": 10, "sudden": 6, "sustained": 12}[kind]
    starts = np.where(
        (hour == 22) & (off == 0) if kind == "night" else
        (hour == 8) & (g["is_weekend"].to_numpy() == 1) if kind == "weekend" else
        (hour >= 6) & (hour <= 12) & (off == 0))[0]
    rng.shuffle(starts)
    out = []
    for s in starts:
        if s + length >= n or taken[s:s + length + 24].any():
            continue
        taken[s:s + length + 24] = True
        out.append(np.arange(s, s + length))
        if len(out) == k:
            break
    return out


def inject(df: pd.DataFrame, level: float, seed: int = C.RANDOM_STATE):
    """Return (df_injected, truth) where truth has one row per injected hour."""
    rng = np.random.default_rng(seed)
    d = df.reset_index(drop=True).copy()
    truth = np.zeros(len(d), dtype="int8")
    kind_of = np.full(len(d), "", dtype=object)
    event_of = np.full(len(d), -1, dtype="int64")
    eid = 0
    for _, idx in d.groupby("building_id").groups.items():
        idx = np.asarray(idx)
        g = d.loc[idx]
        taken = g["kwh"].isna().to_numpy().copy()
        med = float(g["kwh"].median())
        wk = (g[g["is_off"] == 0].groupby("hour")["kwh"].median())
        for kind in INJ_TYPES:
            for win in _event_windows(g.reset_index(drop=True), kind, C.VAL_EVENTS_PER_TYPE, rng, taken):
                rows = idx[win]
                base = d.loc[rows, "kwh"].to_numpy()
                if kind == "weekend":          # runs at working-day level * (1 + level)
                    new = wk.reindex(d.loc[rows, "hour"]).to_numpy() * (1 + level)
                    new = np.maximum(new, base)
                elif kind == "night":
                    new = base + level * med * 1.0
                else:
                    new = base + level * med
                d.loc[rows, "kwh"] = new.astype("float32")
                truth[rows] = 1
                kind_of[rows] = kind
                event_of[rows] = eid
                eid += 1
    d["inj"], d["inj_kind"], d["inj_event"] = truth, kind_of, event_of
    return d


def _score(d):
    ev = d[d["inj"] == 1]
    rows = []
    for kind in INJ_TYPES:
        e = ev[ev["inj_kind"] == kind]
        for col in SCORE_COLS:
            hour_recall = float(e[col].mean()) if len(e) else np.nan
            event_recall = float(e.groupby("inj_event")[col].max().mean()) if len(e) else np.nan
            rows.append({"inj_type": kind, "detector": col.replace("m_", ""),
                         "hour_recall": hour_recall, "event_recall": event_recall})
    clean = d[(d["inj"] == 0) & d["kwh"].notna()]
    fp = {c.replace("m_", ""): float(clean[c].mean()) for c in SCORE_COLS}
    return pd.DataFrame(rows), fp


def run_injection_experiment(df: pd.DataFrame) -> pd.DataFrame:
    """df = the fully prepared dataset (calendar + expected_kwh)."""
    ids = _pick_buildings(df, C.VAL_N_BUILDINGS)
    base = df[df["building_id"].isin(ids)].copy()
    out = []
    for level in C.VAL_LEVELS:
        d = inject(base, level)
        d = anomaly_rules.apply_rules(d)
        d = detectors.apply_methods(d)
        tbl, fp = _score(d)
        tbl["level"] = level
        tbl["flag_rate_on_untouched_hours"] = tbl["detector"].map(fp)
        out.append(tbl)
    res = pd.concat(out, ignore_index=True).round(3)
    res.to_csv(C.RESULTS / "validation_injection.csv", index=False)
    return res


def run_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """How much do the headline numbers move when the key thresholds change?"""
    ids = _pick_buildings(df, C.VAL_N_BUILDINGS)
    base = df[df["building_id"].isin(ids)].copy()
    grid = {"RESIDUAL_IQR_K": [2.0, 3.0, 4.0], "ZSCORE_THRESHOLD": [2.5, 3.0, 3.5],
            "IQR_K": [1.0, 1.5, 3.0], "IFOREST_CONTAMINATION": [0.01, 0.02, 0.04]}
    rows = []
    for name, values in grid.items():
        original = getattr(C, name)
        for val in values:
            setattr(C, name, val)
            try:
                d = anomaly_rules.apply_rules(base.copy())
                d = detectors.apply_methods(d)
            finally:
                setattr(C, name, original)
            sv = ((d["kwh"] - d["expected_kwh"]).clip(lower=0) * d["final_anomaly"]).sum()
            rows.append({"parameter": name, "value": val,
                         "final_anomaly_hours": int(d["final_anomaly"].sum()),
                         "potential_saving_kwh": round(float(sv), 0),
                         "avoidable_pct": round(float(sv / d["kwh"].sum() * 100), 3),
                         "is_default": val == original})
    res = pd.DataFrame(rows)
    res.to_csv(C.RESULTS / "validation_sensitivity.csv", index=False)
    return res
