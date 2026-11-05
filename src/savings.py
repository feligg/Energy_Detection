"""Task 5 - ESTIMATED energy-saving potential.

    Potential saving = Actual consumption - Expected normal consumption

computed only for hours that are final anomalies and only when positive.
Three scenarios give a range instead of a single number:

    low      all 3 detector families agree          (conservative)
    central  >= 2 families agree                    (main estimate)
    high     >= 2 of the 4 methods agree            (generous)

These are model-based estimates, NOT verified savings from a control system.
"""
import numpy as np
import pandas as pd

from . import config as C
from .anomaly_rules import TYPE_COLS

SCENARIOS = {"low": "final_low", "central": "final_anomaly", "high": "final_high"}


def add_saving_columns(df: pd.DataFrame) -> pd.DataFrame:
    excess = (df["kwh"] - df["expected_kwh"]).clip(lower=0)
    for name, flag in SCENARIOS.items():
        col = "saving_kwh" if name == "central" else f"saving_{name}_kwh"
        df[col] = np.where(df[flag] == 1, excess, 0.0).astype("float32")
    return df


def attention_levels(avoidable_pct: pd.Series) -> pd.Series:
    lo, hi = avoidable_pct.quantile(C.ATTENTION_QUANTILES[0]), avoidable_pct.quantile(C.ATTENTION_QUANTILES[1])
    return pd.Series(np.select([avoidable_pct >= hi, avoidable_pct >= lo], ["high", "medium"], "low"),
                     index=avoidable_pct.index)


def compute_savings(df: pd.DataFrame) -> dict:
    v = add_saving_columns(df[df["kwh"].notna()].copy())
    v["date"] = v["timestamp"].dt.normalize()
    v["month"] = v["timestamp"].dt.to_period("M").dt.to_timestamp()
    sv = ["saving_low_kwh", "saving_kwh", "saving_high_kwh"]

    def agg(keys):
        a = v.groupby(keys).agg(actual_kwh=("kwh", "sum"), expected_kwh=("expected_kwh", "sum"),
                                saving_low_kwh=("saving_low_kwh", "sum"),
                                saving_kwh=("saving_kwh", "sum"),
                                saving_high_kwh=("saving_high_kwh", "sum"),
                                anomaly_hours=("final_anomaly", "sum"))
        a["avoidable_pct"] = (a["saving_kwh"] / a["actual_kwh"] * 100).round(3)
        return a.reset_index()

    daily = agg(["building_id", "date"])
    monthly = agg(["building_id", "month"])

    # ---- per building ranking ------------------------------------------------
    days = v.groupby("building_id")["date"].nunique()
    b = agg(["building_id"]).set_index("building_id")
    b["avg_daily_saving_kwh"] = (b["saving_kwh"] / days).round(2)
    b["avg_monthly_saving_kwh"] = (b["avg_daily_saving_kwh"] * 30.4).round(1)
    meta = v.groupby("building_id")[["site_id", "sub_primaryspaceusage", "sqm"]].first()
    b = b.join(meta)
    b["saving_kwh_per_sqm"] = (b["saving_kwh"] / b["sqm"]).round(3)
    dom = (v[v["final_anomaly"] == 1].groupby("building_id")[TYPE_COLS].sum()
           .idxmax(axis=1).rename("dominant_anomaly_type"))
    b = b.join(dom).fillna({"dominant_anomaly_type": "none"})
    b["attention_rank"] = b["saving_kwh"].rank(ascending=False, method="first").astype(int)
    b["attention_level"] = attention_levels(b["avoidable_pct"])
    b = b.sort_values("attention_rank").reset_index()

    # ---- site and overall summary -------------------------------------------
    site = (b.groupby("site_id")[["actual_kwh", "saving_low_kwh", "saving_kwh", "saving_high_kwh",
                                  "anomaly_hours"]].sum()
            .assign(avoidable_pct=lambda x: (x.saving_kwh / x.actual_kwh * 100).round(2))
            .sort_values("saving_kwh", ascending=False).reset_index())
    by_type = {}
    for t in TYPE_COLS:
        m = (v["final_anomaly"] == 1) & (v[t] == 1)
        by_type[t] = float((v.loc[m, "kwh"] - v.loc[m, "expected_kwh"]).clip(lower=0).sum())
    total = float(v["kwh"].sum())
    overall = {
        "buildings": int(b.shape[0]),
        "total_actual_kwh": total,
        "potential_saving_kwh": {k: float(v[c].sum()) for k, c in
                                 zip(["low", "central", "high"], sv)},
        "avoidable_pct": {k: float(v[c].sum() / total * 100) for k, c in
                          zip(["low", "central", "high"], sv)},
        "mean_daily_saving_kwh_per_building": float(daily["saving_kwh"].mean()),
        "mean_monthly_saving_kwh_per_building": float(b["avg_monthly_saving_kwh"].mean()),
        "saving_kwh_by_rule_type_(overlapping)": by_type,
        "note": "Model-based ESTIMATES, not verified savings from a real control system.",
    }

    daily.to_csv(C.RESULTS / "savings_daily.csv", index=False)
    monthly.to_csv(C.RESULTS / "savings_monthly.csv", index=False)
    b.to_csv(C.RESULTS / "savings_by_building.csv", index=False)
    site.to_csv(C.RESULTS / "savings_by_site.csv", index=False)
    return {"daily": daily, "monthly": monthly, "buildings": b, "sites": site,
            "overall": overall}
