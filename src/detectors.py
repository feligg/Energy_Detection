"""Task 3 - anomaly-detection methods.

All methods are ONE-SIDED (they only flag consumption that is too HIGH),
because the goal of the project is to find avoidable energy use.

  1. zscore   - classic (x - mean) / std inside building+season+day-type+hour
  2. iqr      - Tukey fence  x > Q3 + 1.5*IQR inside building+season+weekday+hour
  3. iforest  - Isolation Forest per building on calendar, weather and load features
  4. residual - actual much higher than expected: fence on the building's own
                residuals (actual - expected), expected = baseline or Student 1

Voting. Z-score and IQR both describe "unusual for this time slot", so they are
counted as ONE family ("statistical"). The final decision is a vote of the three
independent families: statistical, iforest, residual.

  final_anomaly : >= 2 families agree   (central estimate)
  final_low     : all 3 families agree  (conservative estimate)
  final_high    : >= 2 of the 4 methods (generous estimate)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from . import config as C

METHODS = ["zscore", "iqr", "iforest", "residual"]
FAMILIES = ["statistical", "iforest", "residual"]


def method_zscore(df):
    std = np.maximum(df["ctx_std"], 0.02 * df["ctx_mean"].abs() + 1e-6)
    z = (df["kwh"] - df["ctx_mean"]) / std
    z = z.where(df["ctx_n"] >= C.MIN_CONTEXT_SAMPLES)
    df["zscore_value"] = z.astype("float32")
    return z > C.ZSCORE_THRESHOLD


def method_iqr(df):
    fence = df["ctx2_q3"] + C.IQR_K * (df["ctx2_q3"] - df["ctx2_q1"])
    return (df["kwh"] > fence) & (df["ctx2_n"] >= C.MIN_CONTEXT_SAMPLES)


def method_iforest(df):
    flags = pd.Series(False, index=df.index)
    score = pd.Series(np.nan, index=df.index, dtype="float64")
    prev24 = df.groupby("building_id")["kwh"].transform(
        lambda s: s.rolling(24, min_periods=12).mean())
    feats = pd.DataFrame({
        "kwh": df["kwh"],
        "ratio_ctx": df["kwh"] / (df["ctx_median"] + 1e-6),
        "hour_sin": np.sin(2 * np.pi * df["hour"] / 24),
        "hour_cos": np.cos(2 * np.pi * df["hour"] / 24),
        "is_off": df["is_off"],
        "temp": df["airTemperature"],
        "delta_1h": df.get("delta_1h", pd.Series(0.0, index=df.index)),
        "ratio_24h": df["kwh"] / (prev24 + 1e-6),
    })
    for _, idx in df.groupby("building_id").groups.items():
        X = feats.loc[idx]
        ok = X.notna().all(axis=1) & df.loc[idx, "kwh"].notna()
        if ok.sum() < 500:
            continue
        model = IsolationForest(n_estimators=C.IFOREST_ESTIMATORS,
                                contamination=C.IFOREST_CONTAMINATION,
                                random_state=C.RANDOM_STATE, n_jobs=2)
        model.fit(X[ok])
        pred = model.predict(X[ok]) == -1
        high_side = df.loc[X.index[ok], "kwh"] > df.loc[X.index[ok], "ctx_median"]
        flags.loc[X.index[ok]] = pred & high_side.to_numpy()
        score.loc[X.index[ok]] = -model.score_samples(X[ok])
    df["iforest_score"] = score.astype("float32")
    return flags


def method_residual(df):
    resid = df["kwh"] - df["expected_kwh"]
    g = resid.groupby(df["building_id"])
    q1, q3 = g.transform(lambda s: s.quantile(.25)), g.transform(lambda s: s.quantile(.75))
    return (resid > q3 + C.RESIDUAL_IQR_K * (q3 - q1)) & (df["expected_kwh"] > 0)


def apply_methods(df: pd.DataFrame) -> pd.DataFrame:
    valid = df["kwh"].notna()
    fns = {"zscore": method_zscore, "iqr": method_iqr,
           "iforest": method_iforest, "residual": method_residual}
    for name, fn in fns.items():
        df[f"m_{name}"] = (fn(df).fillna(False) & valid).astype("int8")
    mcols = [f"m_{m}" for m in METHODS]
    df["n_methods"] = df[mcols].sum(axis=1).astype("int8")
    df["f_statistical"] = (df["m_zscore"] | df["m_iqr"]).astype("int8")
    df["n_families"] = (df["f_statistical"] + df["m_iforest"] + df["m_residual"]).astype("int8")
    df["final_anomaly"] = (df["n_families"] >= C.MIN_FAMILIES_FOR_FINAL).astype("int8")
    df["final_low"] = (df["n_families"] >= C.MIN_FAMILIES_LOW).astype("int8")
    df["final_high"] = (df["n_methods"] >= 2).astype("int8")
    return df
