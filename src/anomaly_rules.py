"""Task 2 - rule-based definitions of five abnormal-consumption types.

All rules compare a building with ITS OWN normal behaviour, so a large and a
small building are treated fairly. "Context" = same building + same
hour-of-day + same day-type (weekday / weekend).

Robust z-score = (x - median) / (1.4826 * MAD): like a z-score, but median and
MAD are not distorted by the very anomalies we are looking for.
"""
import numpy as np
import pandas as pd

from . import config as C

CTX = ["building_id", "is_weekend", "hour"]
TYPE_COLS = ["night_high", "weekend_high", "sudden_increase",
             "weather_deviation", "sustained_high"]


def add_context_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Median / MAD / robust-z / mean / std / quartiles inside each context."""
    g = df.groupby(CTX)["kwh"]
    df["ctx_median"] = g.transform("median").astype("float32")
    df["ctx_mean"] = g.transform("mean").astype("float32")
    df["ctx_std"] = g.transform("std").astype("float32")
    df["ctx_q1"] = g.transform(lambda s: s.quantile(0.25)).astype("float32")
    df["ctx_q3"] = g.transform(lambda s: s.quantile(0.75)).astype("float32")
    absdev = (df["kwh"] - df["ctx_median"]).abs()
    mad = absdev.groupby([df[c] for c in CTX]).transform("median")
    scale = np.maximum(1.4826 * mad, 0.02 * df["ctx_median"].abs() + 1e-6)
    df["robust_z"] = ((df["kwh"] - df["ctx_median"]) / scale).astype("float32")
    return df


def rule_night_high(df):
    """1. Unusually high use at night (22:00-05:59) vs the building's own nights."""
    return (df["hour"].isin(C.NIGHT_HOURS) & (df["robust_z"] > C.ROBUST_Z)
            & (df["kwh"] > C.MIN_EXCESS_RATIO * df["ctx_median"]))


def rule_weekend_high(df):
    """2. A weekend (daytime) hour where a building that normally steps down on
    weekends runs at weekday level or above."""
    wk = (df[df["is_weekend"] == 0].groupby(["building_id", "hour"])["kwh"]
          .median().rename("weekday_median").reset_index())
    d = df[["building_id", "hour"]].merge(wk, on=["building_id", "hour"], how="left")
    weekday_med = d["weekday_median"].to_numpy()
    return ((df["is_weekend"] == 1) & ~df["hour"].isin(C.NIGHT_HOURS)
            & (df["ctx_median"] < C.WEEKEND_DROP_RATIO * weekday_med)
            & (df["kwh"] >= C.WEEKEND_AS_WEEKDAY_RATIO * weekday_med)
            & (df["kwh"] > C.MIN_EXCESS_RATIO * df["ctx_median"]))


def rule_sudden_increase(df):
    """3. One-hour jump larger than Q3 + 3*IQR of the building's own hourly
    increases and larger than 30 % of its median load."""
    prev = df.groupby("building_id")["kwh"].shift(1)
    consecutive = df.groupby("building_id")["timestamp"].diff() == pd.Timedelta("1h")
    d = (df["kwh"] - prev).where(consecutive)
    df["delta_1h"] = d.astype("float32")
    grp = d.groupby(df["building_id"])
    q1, q3 = grp.transform(lambda s: s.quantile(.25)), grp.transform(lambda s: s.quantile(.75))
    med = df.groupby("building_id")["kwh"].transform("median")
    return (d > q3 + C.SUDDEN_IQR_K * (q3 - q1)) & (d > C.SUDDEN_MIN_RATIO * med)


def rule_weather_deviation(df):
    """4. Actual use much higher than the expected use for similar weather
    (baseline model or Student 1's prediction)."""
    resid = df["kwh"] - df["expected_kwh"]
    g = resid.groupby(df["building_id"])
    med = g.transform("median")
    mad = (resid - med).abs().groupby(df["building_id"]).transform("median")
    rz = (resid - med) / np.maximum(1.4826 * mad, 0.02 * df["expected_kwh"].abs() + 1e-6)
    df["resid_z"] = rz.astype("float32")
    return ((rz > C.ROBUST_Z) & (df["expected_kwh"] > 0)
            & (df["kwh"] >= C.WEATHER_RATIO * df["expected_kwh"]))


def rule_sustained_high(df):
    """5. At least 6 consecutive hours that are each clearly above normal."""
    high = ((df["robust_z"] > C.SUSTAINED_Z)
            & (df["kwh"] > C.SUSTAINED_RATIO * df["ctx_median"]))
    run_id = (high != high.groupby(df["building_id"]).shift()).cumsum()
    run_len = high.groupby(run_id).transform("size")
    return high & (run_len >= C.SUSTAINED_HOURS)


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    df = add_context_stats(df)
    valid = df["kwh"].notna()
    rules = {"night_high": rule_night_high, "weekend_high": rule_weekend_high,
             "sudden_increase": rule_sudden_increase,
             "weather_deviation": rule_weather_deviation,
             "sustained_high": rule_sustained_high}
    for name, fn in rules.items():
        df[name] = (fn(df).fillna(False) & valid).astype("int8")
    df["n_rule_types"] = df[TYPE_COLS].sum(axis=1).astype("int8")
    return df