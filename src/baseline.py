"""Calendar features and the expected ("normal") consumption.

Expected consumption under similar weather = Student 1's predictions when the
file exists, otherwise a weather-aware gradient-boosting baseline trained per
building with blocked cross-fitting: the time axis is cut into CV_BLOCKS
contiguous blocks and every block is predicted by a model that never saw it, so
the model cannot memorise (and therefore hide) the anomalies we want to expose.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config as C

FEATURES = ["hour", "dow", "month", "doy", "is_weekend", "is_break", "airTemperature",
            "dewTemperature", "windSpeed", "cloudCoverage", "temp_24h_mean"]


def add_off_days(df: pd.DataFrame) -> pd.DataFrame:
    """Flag weekdays on which a whole site runs far below its normal load
    (academic breaks, public holidays). Uses the median over the site's
    buildings, so a single abnormal building cannot create a "break"."""
    date = df["timestamp"].dt.normalize()
    daily = (df.assign(date=date).groupby(["site_id", "building_id", "date"])["kwh"]
             .mean().rename("daily").reset_index())
    daily = daily[daily["date"].dt.dayofweek < 5].dropna(subset=["daily"])
    ref = daily.groupby("building_id")["daily"].transform("median")
    daily["rel"] = daily["daily"] / ref.clip(lower=1e-6)
    site = daily.groupby(["site_id", "date"]).agg(site_rel=("rel", "median"),
                                                  n=("rel", "size")).reset_index()
    site["is_break"] = ((site["site_rel"] < C.BREAK_RATIO)
                        & (site["n"] >= C.BREAK_MIN_BUILDINGS)).astype("int8")
    key = pd.DataFrame({"site_id": df["site_id"].to_numpy(), "date": date.to_numpy()})
    key = key.merge(site[["site_id", "date", "is_break"]], on=["site_id", "date"], how="left")
    df["is_break"] = key["is_break"].fillna(0).astype("int8").to_numpy()
    return df


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour.astype("int8")
    df["dow"] = ts.dt.dayofweek.astype("int8")
    df["month"] = ts.dt.month.astype("int8")
    df["doy"] = ts.dt.dayofyear.astype("int16")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    df["season"] = df["month"].map(C.SEASON_OF_MONTH).astype("int8")
    df = add_off_days(df)
    df["is_off"] = ((df["is_weekend"] == 1) | (df["is_break"] == 1)).astype("int8")
    df["temp_24h_mean"] = (df.groupby("building_id")["airTemperature"]
                           .transform(lambda s: s.rolling(24, min_periods=6).mean())
                           .astype("float32"))
    return df


def _fit_predict_building(bid: str, g: pd.DataFrame):
    g = g.reset_index(drop=True)
    n = len(g)
    block = np.minimum((np.arange(n) * C.CV_BLOCKS) // n, C.CV_BLOCKS - 1)
    ok = g["kwh"].notna().to_numpy()
    pred = np.full(n, np.nan, dtype="float32")
    X = g[FEATURES]
    for b in range(C.CV_BLOCKS):
        tr = (block != b) & ok
        te = block == b
        if tr.sum() < 500:
            continue
        # a feature that is constant / all-NaN in the training blocks (e.g. a
        # weather sensor that is missing for a whole site) cannot be binned
        use = [f for f in FEATURES if X.loc[tr, f].nunique(dropna=True) >= 2]
        model = HistGradientBoostingRegressor(**C.BASELINE_PARAMS)
        model.fit(X.loc[tr, use], g.loc[tr, "kwh"])
        pred[te] = model.predict(X.loc[te, use])
    return bid, pred


def add_expected(df: pd.DataFrame, n_jobs: int = 2) -> pd.DataFrame:
    """Adds `expected_kwh` and `expected_source`."""
    groups = [(b, g) for b, g in df.groupby("building_id", sort=False)]
    res = Parallel(n_jobs=n_jobs)(delayed(_fit_predict_building)(b, g[FEATURES + ["kwh"]])
                                  for b, g in groups)
    lookup = dict(res)
    df["expected_kwh"] = np.concatenate([lookup[b] for b, _ in groups])
    df["expected_kwh"] = df["expected_kwh"].clip(lower=0).astype("float32")
    df["expected_source"] = "baseline_model"

    if C.STUDENT1_FILE.exists():
        df = _override_with_student1(df)
    return df


def _override_with_student1(df: pd.DataFrame) -> pd.DataFrame:
    """Student 1 file: columns timestamp, building_id, predicted_kwh (long)."""
    s1 = pd.read_csv(C.STUDENT1_FILE, parse_dates=["timestamp"])
    s1 = s1.rename(columns={"predicted_kwh": "s1_pred"})[["timestamp", "building_id", "s1_pred"]]
    df = df.merge(s1.drop_duplicates(["timestamp", "building_id"]),
                  on=["timestamp", "building_id"], how="left")
    use = df["s1_pred"].notna()
    df["expected_baseline_kwh"] = df["expected_kwh"]
    df.loc[use, "expected_kwh"] = df.loc[use, "s1_pred"].clip(lower=0).astype("float32")
    df.loc[use, "expected_source"] = "student1"
    print(f"[baseline] Student 1 predictions used for {use.mean():.1%} of rows")
    return df.drop(columns="s1_pred")
