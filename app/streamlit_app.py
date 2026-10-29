"""Task 6 - Streamlit dashboard.

Run:  streamlit run app/streamlit_app.py
(first run `python run_pipeline.py` so that outputs/results exists)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RES, CASES = ROOT / "outputs" / "results", ROOT / "outputs" / "cases"

METHODS = ["zscore", "iqr", "iforest", "residual"]
TYPES = ["night_high", "weekend_high", "sudden_increase", "weather_deviation", "sustained_high"]
TYPE_TEXT = {
    "night_high": "Night 22:00-05:59: robust z > 3 vs the building's own nights and >= 1.25x normal level",
    "weekend_high": "Weekend daytime hour at >= 95 % of weekday level in a building that normally drops >= 10 % on weekends",
    "sudden_increase": "1-hour increase > Q3 + 3*IQR of the building's own increases and > 30 % of its median load",
    "weather_deviation": "Actual >= 1.3x expected for similar weather and residual robust z > 3",
    "sustained_high": ">= 6 consecutive hours with robust z > 2.5 and >= 1.2x normal level",
}
COL_TYPE = dict(zip(TYPES, ["#6a4c93", "#1982c4", "#ff9f1c", "#e63946", "#2a9d8f"]))
COL_METHOD = dict(zip(METHODS, ["#e63946", "#ff9f1c", "#6a4c93", "#2a9d8f"]))

st.set_page_config(page_title="Abnormal Energy Consumption", layout="wide")


@st.cache_data(show_spinner="Loading results ...")
def load():
    ts = pd.read_parquet(RES / "timeseries_flags.parquet")
    ts["saving_kwh"] = np.where(ts["final_anomaly"] == 1,
                                (ts["kwh"] - ts["expected_kwh"]).clip(lower=0), 0.0)
    return ts


if not (RES / "timeseries_flags.parquet").exists():
    st.error("No results found. Run `python run_pipeline.py` first.")
    st.stop()

ts = load()
valid = ts[ts["kwh"].notna()]

# ------------------------------------------------------------------ sidebar
st.sidebar.title("Filters")
sites = sorted(valid["site_id"].unique())
site_sel = st.sidebar.multiselect("Site", sites, default=sites)
scope = valid[valid["site_id"].isin(site_sel)]
d0, d1 = scope["timestamp"].min().date(), scope["timestamp"].max().date()
date_rng = st.sidebar.date_input("Date range", (d0, d1), min_value=d0, max_value=d1)
if len(date_rng) == 2:
    scope = scope[(scope["timestamp"].dt.date >= date_rng[0]) & (scope["timestamp"].dt.date <= date_rng[1])]
meth_sel = st.sidebar.multiselect("Detection methods shown", METHODS, default=METHODS)
type_sel = st.sidebar.multiselect("Anomaly types shown", TYPES, default=TYPES)
st.sidebar.caption("Final anomaly = flagged by >= 2 of the 4 methods.")

# ----------------------------------------------------------- building table
bld = (scope.groupby("building_id")
       .agg(site=("site_id", "first"), use=("sub_primaryspaceusage", "first"),
            actual_kwh=("kwh", "sum"), saving_kwh=("saving_kwh", "sum"),
            anomaly_hours=("final_anomaly", "sum"), hours=("kwh", "size"))
       .reset_index())
bld["avoidable_pct"] = (bld["saving_kwh"] / bld["actual_kwh"] * 100).round(2)
bld["share_anomalous_hours_pct"] = (bld["anomaly_hours"] / bld["hours"] * 100).round(2)
fin = scope[scope["final_anomaly"] == 1]
dom = fin.groupby("building_id")[TYPES].sum().idxmax(axis=1) if len(fin) else pd.Series(dtype=str)
bld["main_anomaly_type"] = bld["building_id"].map(dom).fillna("none")
bld["attention"] = pd.cut(bld["avoidable_pct"], [-1, 1, 3, 1e9], labels=["low", "medium", "high"]).astype(str)
bld = bld.sort_values("saving_kwh", ascending=False).reset_index(drop=True)

st.title("Abnormal electricity consumption and energy-saving potential")
st.caption("BDG2 education buildings. Savings are model-based ESTIMATES, "
           "not verified savings from a real control system.")

tabs = st.tabs(["Overview", "Building time series", "Anomaly types", "Method comparison",
                "Saving potential", "Buildings needing attention", "Inspected cases", "Data quality"])

# --------------------------------------------------------------- 1 overview
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Buildings", f"{scope['building_id'].nunique():,}")
    c[1].metric("Hours analysed", f"{len(scope):,}")
    c[2].metric("Final anomaly hours", f"{int(scope['final_anomaly'].sum()):,}",
                f"{scope['final_anomaly'].mean() * 100:.2f} % of hours")
    c[3].metric("Potential saving (kWh)", f"{scope['saving_kwh'].sum():,.0f}")
    c[4].metric("Avoidable share", f"{scope['saving_kwh'].sum() / max(scope['kwh'].sum(), 1) * 100:.2f} %")
    a, b = st.columns(2)
    site_sv = scope.groupby("site_id")[["saving_kwh", "kwh"]].sum().reset_index()
    site_sv["avoidable_pct"] = site_sv["saving_kwh"] / site_sv["kwh"] * 100
    a.plotly_chart(px.bar(site_sv.sort_values("saving_kwh", ascending=False), x="site_id",
                          y="saving_kwh", color="avoidable_pct", color_continuous_scale="Reds",
                          title="Estimated saving potential by site (kWh)"), width="stretch")
    if len(fin):
        heat = (fin.pivot_table(index="dow", columns="hour", values="kwh", aggfunc="size", fill_value=0)
                .reindex(index=range(7), columns=range(24), fill_value=0))
        heat.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        b.plotly_chart(px.imshow(heat, aspect="auto", color_continuous_scale="Reds",
                                 labels=dict(x="hour of day", y="weekday", color="anomaly hours"),
                                 title="When do final anomalies occur?"), width="stretch")
    fig_png = ROOT / "outputs" / "figures" / "method_comparison.png"
    if fig_png.exists():
        st.image(str(fig_png))

# ------------------------------------------------------ 2 building time series
with tabs[1]:
    order = bld["building_id"].tolist()
    if not order:
        st.warning("No data for this filter.")
    else:
        bsel = st.selectbox("Building (sorted by saving potential)", order)
        mode = st.radio("Colour anomalies by", ["Final (>= 2 methods)", "Detection method", "Anomaly type"],
                        horizontal=True)
        g = scope[scope["building_id"] == bsel].sort_values("timestamp")
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=g["timestamp"], y=g["kwh"], mode="lines", name="actual (normal)",
                                   line=dict(color="#1d3557", width=1)))
        fig.add_trace(go.Scattergl(x=g["timestamp"], y=g["expected_kwh"], mode="lines",
                                   name="expected normal", line=dict(color="#8ecae6", width=1, dash="dot")))
        if mode.startswith("Final"):
            a = g[g["final_anomaly"] == 1]
            fig.add_trace(go.Scattergl(x=a["timestamp"], y=a["kwh"], mode="markers", name="final anomaly",
                                       marker=dict(color="#e63946", size=5)))
        elif mode.startswith("Detection"):
            for m in meth_sel:
                a = g[g[f"m_{m}"] == 1]
                fig.add_trace(go.Scattergl(x=a["timestamp"], y=a["kwh"], mode="markers", name=m,
                                           marker=dict(color=COL_METHOD[m], size=5)))
        else:
            for t in type_sel:
                a = g[g[t] == 1]
                fig.add_trace(go.Scattergl(x=a["timestamp"], y=a["kwh"], mode="markers", name=t,
                                           marker=dict(color=COL_TYPE[t], size=5)))
        fig.update_layout(height=480, yaxis_title="kWh per hour", hovermode="x unified",
                          title=f"{bsel}: normal vs abnormal periods", margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
        daily = g.assign(date=g["timestamp"].dt.date).groupby("date")["saving_kwh"].sum().reset_index()
        st.plotly_chart(px.bar(daily, x="date", y="saving_kwh",
                               title="Estimated avoidable consumption per day (kWh)"), width="stretch")

# ---------------------------------------------------------- 3 anomaly types
with tabs[2]:
    rows = [{"type": t, "hours": int(scope[t].sum()),
             "buildings": int(scope.loc[scope[t] == 1, "building_id"].nunique()),
             "rule": TYPE_TEXT[t]} for t in TYPES]
    tt = pd.DataFrame(rows)
    st.dataframe(tt, width="stretch", hide_index=True)
    a, b = st.columns(2)
    a.plotly_chart(px.bar(tt[tt["type"].isin(type_sel)], x="type", y="hours", color="type",
                          color_discrete_map=COL_TYPE, title="Anomaly hours per type"), width="stretch")
    hh = pd.concat([scope[scope[t] == 1].groupby("hour").size().rename(t) for t in type_sel], axis=1) \
        .reindex(range(24)).fillna(0) if type_sel else pd.DataFrame()
    if len(hh):
        b.plotly_chart(px.line(hh, title="Hour of day of each anomaly type",
                               color_discrete_map=COL_TYPE), width="stretch")

# -------------------------------------------------------- 4 method comparison
with tabs[3]:
    cnt = pd.DataFrame({"method": METHODS + ["final"],
                        "hours": [int(scope[f"m_{m}"].sum()) for m in METHODS] + [int(scope["final_anomaly"].sum())]})
    a, b, c3 = st.columns(3)
    a.plotly_chart(px.bar(cnt, x="method", y="hours", title="Hours flagged"), width="stretch")
    jac = pd.DataFrame(index=METHODS, columns=METHODS, dtype=float)
    for m1 in METHODS:
        for m2 in METHODS:
            A, B = scope[f"m_{m1}"] == 1, scope[f"m_{m2}"] == 1
            jac.loc[m1, m2] = (A & B).sum() / max((A | B).sum(), 1)
    b.plotly_chart(px.imshow(jac.round(2), text_auto=True, color_continuous_scale="Blues",
                             zmin=0, zmax=1, title="Agreement (Jaccard)"), width="stretch")
    ag = scope[scope["n_methods"] > 0].groupby("n_methods").size().rename("hours").reset_index()
    c3.plotly_chart(px.bar(ag, x="n_methods", y="hours", title="Hours by number of agreeing methods"),
                    width="stretch")
    per_b = scope.groupby("building_id")[[f"m_{m}" for m in METHODS]].sum()
    per_b.columns = METHODS
    st.subheader("Flagged hours per building and method")
    st.dataframe(per_b.sort_values("iqr", ascending=False), width="stretch")

# --------------------------------------------------------- 5 saving potential
with tabs[4]:
    st.info("Potential saving = actual consumption - expected normal consumption, summed over final "
            "anomaly hours. This is an estimate, not a verified saving.")
    sc = scope.assign(month=scope["timestamp"].dt.to_period("M").dt.to_timestamp(),
                      date=scope["timestamp"].dt.date)
    nb = max(sc["building_id"].nunique(), 1)
    ndays = max(sc["date"].nunique(), 1)
    c = st.columns(4)
    c[0].metric("Total potential saving", f"{sc['saving_kwh'].sum():,.0f} kWh")
    c[1].metric("Avg daily / building", f"{sc['saving_kwh'].sum() / ndays / nb:,.1f} kWh")
    c[2].metric("Avg monthly / building", f"{sc['saving_kwh'].sum() / ndays / nb * 30.4:,.0f} kWh")
    c[3].metric("Avoidable share", f"{sc['saving_kwh'].sum() / max(sc['kwh'].sum(), 1) * 100:.2f} %")
    mon = sc.groupby("month")[["saving_kwh", "kwh"]].sum().reset_index()
    mon["avoidable_pct"] = mon["saving_kwh"] / mon["kwh"] * 100
    a, b = st.columns(2)
    a.plotly_chart(px.bar(mon, x="month", y="saving_kwh", title="Monthly potential saving (kWh)"),
                   width="stretch")
    b.plotly_chart(px.line(mon, x="month", y="avoidable_pct", markers=True,
                           title="Avoidable share of consumption (%)"), width="stretch")
    st.plotly_chart(px.bar(bld.head(15).iloc[::-1], x="saving_kwh", y="building_id", orientation="h",
                           color="avoidable_pct", color_continuous_scale="Reds",
                           title="15 buildings with the greatest saving potential"), width="stretch")
    dly = sc.groupby("date")["saving_kwh"].sum().reset_index()
    st.plotly_chart(px.line(dly, x="date", y="saving_kwh", title="Potential saving per day, all buildings (kWh)"),
                    width="stretch")

# ------------------------------------------------------------ 6 attention list
with tabs[5]:
    lvl = st.multiselect("Attention level", ["high", "medium", "low"], default=["high", "medium"])
    show = bld[bld["attention"].isin(lvl)][
        ["building_id", "site", "use", "attention", "saving_kwh", "avoidable_pct", "anomaly_hours",
         "share_anomalous_hours_pct", "main_anomaly_type"]].round(1)
    st.caption("Level from the avoidable share: low < 1 %, medium 1-3 %, high > 3 %. "
               "The list is sorted by absolute saving potential (kWh).")
    st.dataframe(show, width="stretch", hide_index=True)
    st.download_button("Download list (CSV)", show.to_csv(index=False), "buildings_requiring_attention.csv")

# ----------------------------------------------------------- 7 inspected cases
with tabs[6]:
    f = RES / "representative_cases.csv"
    if f.exists():
        cs = pd.read_csv(f)
        st.dataframe(cs[["case_id", "building_id", "start", "duration_h", "total_excess_kwh", "primary_type",
                         "methods", "mean_temp_c", "possible_reason"]].round(1), width="stretch", hide_index=True)
        cid = st.selectbox("Show case plot", cs["case_id"].tolist())
        img = list(CASES.glob(f"{cid}_*.png"))
        if img:
            st.image(str(img[0]))
        st.caption(cs.loc[cs["case_id"] == cid, "possible_reason"].iloc[0])

# ------------------------------------------------------------- 8 data quality
with tabs[7]:
    q = RES / "data_quality_report.md"
    st.markdown(q.read_text() if q.exists() else "No report found.")
    s = RES / "data_quality_summary.json"
    if s.exists():
        st.json(json.loads(s.read_text()), expanded=False)