"""Task 5 - ESTIMATED energy-saving potential.

    Potential saving = Actual consumption - Expected normal consumption

computed only for hours that are final anomalies and only when positive.
These are model-based estimates, NOT verified savings from a control system.
"""
import numpy as np
import pandas as pd

from . import config as C
from .anomaly_rules import TYPE_COLS


def compute_savings(df: pd.DataFrame) -> dict:
    v = df[df["kwh"].notna()].copy()
    v["saving_kwh"] = np.where(
        v["final_anomaly"] == 1, (v["kwh"] - v["expected_kwh"]).clip(lower=0), 0.0)
    v["date"] = v["timestamp"].dt.normalize()
    v["month"] = v["timestamp"].dt.to_period("M").dt.to_timestamp()

    def agg(keys):
        a = v.groupby(keys).agg(actual_kwh=("kwh", "sum"), expected_kwh=("expected_kwh", "sum"),
                                saving_kwh=("saving_kwh", "sum"),
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
    b["attention_level"] = pd.cut(
        b["avoidable_pct"], [-1, 1, 3, 100], labels=["low", "medium", "high"]).astype(str)
    b = b.sort_values("attention_rank").reset_index()

    # ---- site and overall summary -------------------------------------------
    site = (b.groupby("site_id")[["actual_kwh", "saving_kwh", "anomaly_hours"]].sum()
            .assign(avoidable_pct=lambda x: (x.saving_kwh / x.actual_kwh * 100).round(2))
            .sort_values("saving_kwh", ascending=False).reset_index())
    by_type = {}
    for t in TYPE_COLS:
        m = (v["final_anomaly"] == 1) & (v[t] == 1)
        by_type[t] = float((v.loc[m, "kwh"] - v.loc[m, "expected_kwh"]).clip(lower=0).sum())
    overall = {
        "buildings": int(b.shape[0]),
        "total_actual_kwh": float(v["kwh"].sum()),
        "total_potential_saving_kwh": float(v["saving_kwh"].sum()),
        "overall_avoidable_pct": float(v["saving_kwh"].sum() / v["kwh"].sum() * 100),
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