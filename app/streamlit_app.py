"""Task 6 - Streamlit dashboard (black and white, kept simple).

Run:  streamlit run app/streamlit_app.py
(first run `python run_pipeline.py` so that outputs/results exists)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C                      # noqa: E402
from src.savings import attention_levels          # noqa: E402

RES, CASES, FIGS = C.RESULTS, C.CASES, C.FIGURES
METHODS = ["zscore", "iqr", "iforest", "residual"]
TYPES = ["night_high", "weekend_high", "sudden_increase", "weather_deviation", "sustained_high"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TYPE_TEXT = {
    "night_high": f"Night 22:00-05:59: robust z > {C.ROBUST_Z} vs the building's own nights "
                  f"and >= {C.MIN_EXCESS_RATIO}x its normal level.",
    "weekend_high": f"Weekend or academic-break daytime hour at >= {C.WEEKEND_AS_WEEKDAY_RATIO:.0%} of "
                    f"working-day level, in a building that normally drops >= "
                    f"{1 - C.WEEKEND_DROP_RATIO:.0%} on such days.",
    "sudden_increase": f"1-hour increase > Q3 + {C.SUDDEN_IQR_K}*IQR of the building's own increases "
                       f"and > {C.SUDDEN_MIN_RATIO:.0%} of its median load.",
    "weather_deviation": f"Actual >= {C.WEATHER_RATIO}x the expected use for similar weather "
                         f"and residual robust z > {C.ROBUST_Z}.",
    "sustained_high": f">= {C.SUSTAINED_HOURS} consecutive hours with robust z > {C.SUSTAINED_Z} "
                      f"and >= {C.SUSTAINED_RATIO}x normal level.",
}
SYMBOL_METHOD = dict(zip(METHODS, ["circle-open", "square-open", "x", "diamond-open"]))
SYMBOL_TYPE = dict(zip(TYPES, ["circle-open", "square-open", "triangle-up-open", "x", "diamond-open"]))
DASH_METHOD = dict(zip(METHODS, ["solid", "dash", "dot", "dashdot"]))
# one fixed colour per method / anomaly type, used in every chart
MCOL = {"zscore": "#1f77b4", "iqr": "#ff7f0e", "iforest": "#2ca02c", "residual": "#9467bd",
        "final": "#d62728"}
TCOL = dict(zip(TYPES, ["#3b4cc0", "#e377c2", "#d62728", "#17becf", "#bcbd22"]))
SAVE_C, LOW_C, HIGH_C = "#2a9d8f", "#e9c46a", "#e76f51"
HEAT = "YlOrRd"

# ----------------------------------------------------------------- styling
BW = go.layout.Template()
BW.layout = go.Layout(
    paper_bgcolor="white", plot_bgcolor="white",
    font=dict(color="black", family="Helvetica, Arial, sans-serif", size=13),
    colorway=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"],
    xaxis=dict(showgrid=False, linecolor="black", ticks="outside", zeroline=False),
    yaxis=dict(gridcolor="#e5e5e5", linecolor="black", ticks="outside", zeroline=False),
    margin=dict(t=50, b=30, l=50, r=20), hovermode="x unified",
    legend=dict(orientation="h", y=-0.2))
GREYS = [[0, "#ffffff"], [1, "#000000"]]


def show(fig, height=380, **kw):
    fig.update_layout(template=BW, height=height, **kw)
    st.plotly_chart(fig, width="stretch")


st.set_page_config(page_title="Abnormal Energy Consumption", layout="wide")
st.markdown("""<style>
h1, h2, h3 {font-weight: 600; letter-spacing: -0.01em;}
[data-testid="stMetricValue"] {font-weight: 600;}
.block-container {padding-top: 2rem; max-width: 1200px;}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading results ...")
def load():
    return pd.read_parquet(RES / "timeseries_flags.parquet")


def read_csv(name):
    f = RES / name
    return pd.read_csv(f) if f.exists() else None


if not (RES / "timeseries_flags.parquet").exists():
    st.error("No results found. Run `python run_pipeline.py` first.")
    st.stop()

ts = load()
valid = ts[ts["kwh"].notna()]

# ------------------------------------------------------------------ sidebar
st.sidebar.header("Filters")
sites = sorted(valid["site_id"].unique())
site_sel = st.sidebar.multiselect("Site", sites, default=sites)
scope = valid[valid["site_id"].isin(site_sel)]
if scope.empty:
    st.warning("Select at least one site.")
    st.stop()
d0, d1 = scope["timestamp"].min().date(), scope["timestamp"].max().date()
date_rng = st.sidebar.date_input("Date range", (d0, d1), min_value=d0, max_value=d1)
if isinstance(date_rng, (tuple, list)) and len(date_rng) == 2:
    scope = scope[(scope["timestamp"].dt.date >= date_rng[0])
                  & (scope["timestamp"].dt.date <= date_rng[1])]
st.sidebar.caption("Final anomaly = flagged by at least 2 of 3 method families "
                   "(statistical, Isolation Forest, residual).")

# ----------------------------------------------------------- building table
bld = (scope.groupby("building_id")
       .agg(site=("site_id", "first"), use=("sub_primaryspaceusage", "first"),
            actual_kwh=("kwh", "sum"), saving_low=("saving_low_kwh", "sum"),
            saving_kwh=("saving_kwh", "sum"), saving_high=("saving_high_kwh", "sum"),
            anomaly_hours=("final_anomaly", "sum"), hours=("kwh", "size"))
       .reset_index())
bld["avoidable_pct"] = (bld["saving_kwh"] / bld["actual_kwh"] * 100).round(2)
bld["anomalous_hours_pct"] = (bld["anomaly_hours"] / bld["hours"] * 100).round(2)
fin = scope[scope["final_anomaly"] == 1]
dom = fin.groupby("building_id")[TYPES].sum().idxmax(axis=1) if len(fin) else pd.Series(dtype=str)
bld["main_type"] = bld["building_id"].map(dom).fillna("none")
bld["attention"] = attention_levels(bld["avoidable_pct"])
bld = bld.sort_values("saving_kwh", ascending=False).reset_index(drop=True)

st.title("Abnormal electricity consumption")
st.caption("BDG2 education buildings. All savings are model-based ESTIMATES, "
           "not verified savings from a real control system.")

tabs = st.tabs(["Overview", "Building", "Methods", "Savings", "Attention list", "Cases"])

# --------------------------------------------------------------- 1 overview
with tabs[0]:
    tot = scope["kwh"].sum()
    c = st.columns(4)
    c[0].metric("Buildings", f"{scope['building_id'].nunique():,}")
    c[1].metric("Anomaly hours", f"{int(scope['final_anomaly'].sum()):,}")
    c[1].caption(f"{scope['final_anomaly'].mean() * 100:.2f} % of all hours")
    c[2].metric("Potential saving", f"{scope['saving_kwh'].sum() / 1000:,.0f} MWh")
    c[2].caption(f"range {scope['saving_low_kwh'].sum() / 1000:,.0f} - "
                 f"{scope['saving_high_kwh'].sum() / 1000:,.0f} MWh (low - high)")
    c[3].metric("Avoidable share", f"{scope['saving_kwh'].sum() / max(tot, 1) * 100:.2f} %")
    a, b = st.columns(2)
    if len(fin):
        heat = (fin.pivot_table(index="dow", columns="hour", values="kwh", aggfunc="size", fill_value=0)
                .reindex(index=range(7), columns=range(24), fill_value=0))
        fig = go.Figure(go.Heatmap(z=heat.values, x=list(range(24)), y=DAYS, colorscale=HEAT,
                                   colorbar=dict(title="hours")))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(title="When do anomalies occur?", xaxis_title="hour of day")
        with a:
            show(fig)
    tcount = pd.DataFrame({"type": TYPES, "hours": [int(scope[t].sum()) for t in TYPES]})
    fig = go.Figure(go.Bar(x=tcount["hours"], y=tcount["type"], orientation="h", marker_color=[TCOL[t] for t in tcount["type"]]))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Anomaly hours per rule type")
    with b:
        show(fig)
    with st.expander("How each anomaly type is defined"):
        st.table(pd.DataFrame({"type": TYPES, "rule": [TYPE_TEXT[t] for t in TYPES]}).set_index("type"))
    with st.expander("Data preparation and quality"):
        q = RES / "data_quality_report.md"
        st.markdown(q.read_text() if q.exists() else "No report found.")

# ------------------------------------------------------------- 2 building
with tabs[1]:
    order = bld["building_id"].tolist()
    bsel = st.selectbox("Building (sorted by saving potential)", order)
    mode = st.radio("Show anomalies by",
                    ["Each method in its own panel", "Final decision", "Detection method (overlay)",
                     "Anomaly type"], horizontal=True)
    st.caption("Panels / overlay = WHICH DETECTION METHOD flagged the hour (Task 3). "
               "Anomaly type = WHAT KIND of abnormal behaviour it is, from the Task 2 rules. "
               "Final decision = hours flagged by at least 2 method families.")
    g = scope[scope["building_id"] == bsel].sort_values("timestamp")
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=g["timestamp"], y=g["kwh"], mode="lines", name="actual",
                               line=dict(color="black", width=1)))
    fig.add_trace(go.Scattergl(x=g["timestamp"], y=g["expected_kwh"], mode="lines",
                               name="expected normal", line=dict(color="#888888", width=1, dash="dot")))
    fig_h = 460
    if mode == "Each method in its own panel":
        panels = [("zscore", "Z-score"), ("iqr", "IQR"), ("iforest", "Isolation Forest"),
                  ("residual", "Residual (actual vs expected / Student 1)"), ("final", "Final decision")]
        fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.04,
                            subplot_titles=[f"{t}: {int((g['final_anomaly'] if k == 'final' else g[f'm_{k}']).sum())} h"
                                            for k, t in panels])
        for i, (k, t) in enumerate(panels, start=1):
            flag = g["final_anomaly"] if k == "final" else g[f"m_{k}"]
            fig.add_trace(go.Scattergl(x=g["timestamp"], y=g["kwh"], mode="lines", showlegend=False,
                                       line=dict(color="#999999", width=1)), row=i, col=1)
            a_ = g[flag == 1]
            fig.add_trace(go.Scattergl(x=a_["timestamp"], y=a_["kwh"], mode="markers", showlegend=False,
                                       marker=dict(color=MCOL[k], size=5)), row=i, col=1)
        fig_h = 150 * len(panels) + 80
    elif mode == "Final decision":
        a_ = g[g["final_anomaly"] == 1]
        fig.add_trace(go.Scattergl(x=a_["timestamp"], y=a_["kwh"], mode="markers", name="final anomaly",
                                   marker=dict(color=MCOL["final"], size=6)))
    elif mode == "Detection method (overlay)":
        for m in METHODS:
            a_ = g[g[f"m_{m}"] == 1]
            fig.add_trace(go.Scattergl(x=a_["timestamp"], y=a_["kwh"], mode="markers", name=m,
                                       marker=dict(color=MCOL[m], size=7, symbol=SYMBOL_METHOD[m].replace("-open", ""))))
    else:
        for t in TYPES:
            a_ = g[g[t] == 1]
            fig.add_trace(go.Scattergl(x=a_["timestamp"], y=a_["kwh"], mode="markers", name=t,
                                       marker=dict(color=TCOL[t], size=7, symbol=SYMBOL_TYPE[t].replace("-open", ""))))
    fig.update_layout(title=f"{bsel}: normal and abnormal periods")
    if mode != "Each method in its own panel":
        fig.update_layout(yaxis_title="kWh per hour")
    show(fig, fig_h)
    daily = g.assign(date=g["timestamp"].dt.date).groupby("date")["saving_kwh"].sum().reset_index()
    fig = go.Figure(go.Bar(x=daily["date"], y=daily["saving_kwh"], marker_color=SAVE_C))
    fig.update_layout(title="Estimated avoidable consumption per day (kWh)")
    show(fig, 260)

# -------------------------------------------------------------- 3 methods
with tabs[2]:
    nvalid = max(len(scope), 1)
    LABEL = {"zscore": "1. Z-score", "iqr": "2. IQR", "iforest": "3. Isolation Forest",
             "residual": "4. Residual vs expected (Student 1 / baseline)"}
    summ = pd.DataFrame([{
        "method": LABEL[m], "flagged_hours": int(scope[f"m_{m}"].sum()),
        "pct_of_hours": round(scope[f"m_{m}"].sum() / nvalid * 100, 2),
        "buildings_flagged": int(scope.loc[scope[f"m_{m}"] == 1, "building_id"].nunique()),
    } for m in METHODS] + [{
        "method": "Final (>= 2 families)", "flagged_hours": int(scope["final_anomaly"].sum()),
        "pct_of_hours": round(scope["final_anomaly"].sum() / nvalid * 100, 2),
        "buildings_flagged": int(scope.loc[scope["final_anomaly"] == 1, "building_id"].nunique())}])
    st.subheader("Detection methods compared")
    st.caption("All charts on this tab compare the four methods side by side "
               "(each has its own colour); 'final' is the combined vote used for the savings.")
    st.dataframe(summ, width="stretch", hide_index=True)
    names = METHODS + ["final"]
    cnt = [int(scope[f"m_{m}"].sum()) for m in METHODS] + [int(scope["final_anomaly"].sum())]
    a, b = st.columns(2)
    fig = go.Figure(go.Bar(x=names, y=cnt, marker_color=[MCOL[n] for n in names]))
    fig.update_layout(title="Hours flagged", hovermode="closest")
    with a:
        show(fig, 330)
    jac = pd.DataFrame(index=METHODS, columns=METHODS, dtype=float)
    for m1 in METHODS:
        for m2 in METHODS:
            A, B = scope[f"m_{m1}"] == 1, scope[f"m_{m2}"] == 1
            jac.loc[m1, m2] = (A & B).sum() / max((A | B).sum(), 1)
    fig = go.Figure(go.Heatmap(z=jac.values.round(2), x=METHODS, y=METHODS, colorscale=HEAT,
                               zmin=0, zmax=1, text=jac.values.round(2), texttemplate="%{text}"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="Agreement between methods (Jaccard)", hovermode="closest")
    with b:
        show(fig, 330)
    hh = pd.concat([scope[scope[f"m_{m}"] == 1].groupby("hour").size().rename(m) for m in METHODS],
                   axis=1).reindex(range(24)).fillna(0)
    fig = go.Figure()
    for m in METHODS:
        fig.add_trace(go.Scatter(x=hh.index, y=hh[m] / max(hh[m].sum(), 1), mode="lines", name=m,
                                 line=dict(color=MCOL[m], width=2, dash=DASH_METHOD[m])))
    fig.update_layout(title="When in the day does each method flag? (share of its flags)",
                      xaxis_title="hour of day")
    show(fig, 320)
    w1, w2 = st.columns(2)
    for col_, key, lab, order_ in [(w1, "dow", "weekday", range(7)), (w2, "month", "month", range(1, 13))]:
        tt = pd.concat([scope[scope[f"m_{m}"] == 1].groupby(key).size().rename(m) for m in METHODS],
                       axis=1).reindex(list(order_)).fillna(0)
        f_ = go.Figure()
        for m in METHODS:
            f_.add_trace(go.Scatter(x=(DAYS if key == "dow" else list(tt.index)), y=tt[m] / max(tt[m].sum(), 1),
                                    mode="lines", name=m,
                                    line=dict(color=MCOL[m], width=2, dash=DASH_METHOD[m])))
        f_.update_layout(title=f"Share of each method's flags by {lab}")
        with col_:
            show(f_, 300)
    per_b = scope.groupby("building_id")[[f"m_{m}" for m in METHODS] + ["final_anomaly"]].sum()
    per_b.columns = METHODS + ["final"]
    st.subheader("Flagged hours per building")
    st.dataframe(per_b.sort_values("final", ascending=False), width="stretch", height=300)
    rc = read_csv("eval_rule_confirmation.csv")
    if rc is not None:
        st.subheader("Are the rule types backed by the detection methods?")
        st.dataframe(rc, width="stretch", hide_index=True)

# --------------------------------------------------------------- 4 savings
with tabs[3]:
    st.caption("Potential saving = actual consumption - expected normal consumption, summed over final "
            "anomaly hours. Low: all 3 families agree. Central: at least 2 agree. "
            "High: at least 2 of the 4 methods agree.")
    sc = scope.assign(month=scope["timestamp"].dt.to_period("M").dt.to_timestamp(),
                      date=scope["timestamp"].dt.date)
    nb, ndays = max(sc["building_id"].nunique(), 1), max(sc["date"].nunique(), 1)
    s_c = sc["saving_kwh"].sum()
    c = st.columns(4)
    c[0].metric("Total (central)", f"{s_c / 1000:,.0f} MWh")
    c[1].metric("Per building per day", f"{s_c / ndays / nb:,.1f} kWh")
    c[2].metric("Per building per month", f"{s_c / ndays / nb * 30.4:,.0f} kWh")
    c[3].metric("Avoidable share", f"{s_c / max(sc['kwh'].sum(), 1) * 100:.2f} %")
    mon = sc.groupby("month")[["saving_low_kwh", "saving_kwh", "saving_high_kwh", "kwh"]].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=mon["month"], y=mon["saving_kwh"], name="central", marker_color=SAVE_C))
    fig.add_trace(go.Scatter(x=mon["month"], y=mon["saving_low_kwh"], name="low", mode="lines+markers",
                             line=dict(color=LOW_C, dash="dot")))
    fig.add_trace(go.Scatter(x=mon["month"], y=mon["saving_high_kwh"], name="high", mode="lines+markers",
                             line=dict(color=HIGH_C, dash="dash")))
    fig.update_layout(title="Monthly potential saving (kWh)")
    show(fig, 340)
    top = bld.head(15).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top["saving_kwh"], y=top["building_id"], orientation="h", marker_color=SAVE_C,
        error_x=dict(type="data", symmetric=False, color=HIGH_C,
                     array=(top["saving_high"] - top["saving_kwh"]).clip(lower=0),
                     arrayminus=(top["saving_kwh"] - top["saving_low"]).clip(lower=0))))
    fig.update_layout(title="15 buildings with the greatest saving potential (bar = central, whisker = low-high)",
                      hovermode="closest")
    show(fig, 480)
    site_sv = (scope.groupby("site_id")[["saving_kwh", "kwh"]].sum().reset_index()
               .assign(avoidable_pct=lambda x: x.saving_kwh / x.kwh * 100)
               .sort_values("saving_kwh", ascending=False))
    fig = go.Figure(go.Bar(x=site_sv["site_id"], y=site_sv["saving_kwh"], marker_color=SAVE_C))
    fig.update_layout(title="Potential saving by site (kWh)", hovermode="closest")
    show(fig, 300)

# ----------------------------------------------------------- 5 attention
with tabs[4]:
    lvl = st.multiselect("Attention level", ["high", "medium", "low"], default=["high", "medium"])
    show_tbl = bld[bld["attention"].isin(lvl)][
        ["building_id", "site", "use", "attention", "saving_kwh", "saving_low", "saving_high",
         "avoidable_pct", "anomaly_hours", "anomalous_hours_pct", "main_type"]].round(1)
    st.caption("Level is relative to the other buildings shown: high = top 10 % by avoidable share, "
               "medium = next 20 %, low = the rest. Sorted by absolute saving potential (kWh).")
    st.dataframe(show_tbl, width="stretch", hide_index=True)
    st.download_button("Download list (CSV)", show_tbl.to_csv(index=False),
                       "buildings_requiring_attention.csv")

# ---------------------------------------------------------------- 6 cases
with tabs[5]:
    f = RES / "representative_cases.csv"
    if f.exists():
        cs = pd.read_csv(f).fillna("")
        cols = [c_ for c_ in ["case_id", "case_group", "building_id", "start", "duration_h",
                              "total_excess_kwh", "primary_type", "methods", "verdict",
                              "reviewer_notes"] if c_ in cs.columns]
        st.caption("Task 4: inspect each case (plot below), then write your verdict and explanation. "
                   "Press Save to store it in representative_cases.csv.")
        edited = st.data_editor(cs[cols], width="stretch", hide_index=True,
                                disabled=[c_ for c_ in cols if c_ not in ("verdict", "reviewer_notes")],
                                column_config={"verdict": st.column_config.SelectboxColumn(
                                    "verdict", options=["", "true anomaly", "plausible", "false alarm"])})
        if st.button("Save verdicts and notes"):
            full = pd.read_csv(f).fillna("")
            for c_ in ("verdict", "reviewer_notes"):
                full[c_] = edited[c_].to_numpy()
            full.to_csv(f, index=False)
            st.success("Saved.")
        done = int((edited["verdict"] != "").sum()) if "verdict" in edited else 0
        st.caption(f"{done} of {len(edited)} cases reviewed (at least 20 are required).")
        cid = st.selectbox("Case plot", cs["case_id"].tolist())
        img = list(CASES.glob(f"{cid}_*.png"))
        if img:
            st.image(str(img[0]))
        st.caption(cs.loc[cs["case_id"] == cid, "possible_reason"].iloc[0])
    inj = read_csv("validation_injection.csv")
    if inj is not None:
        st.subheader("Validation with injected anomalies")
        st.caption("Known anomalies were added to real buildings; the table shows the share of "
                   "injected events each detector finds (event recall).")
        lv = st.selectbox("Injected size (share of normal level)", sorted(inj["level"].unique()), index=1)
        piv = (inj[inj["level"] == lv]
               .pivot(index="inj_type", columns="detector", values="event_recall"))
        st.dataframe(piv.round(2), width="stretch")
        fp = (inj[inj["level"] == lv].drop_duplicates("detector")
              .set_index("detector")["flag_rate_on_untouched_hours"].round(4))
        st.caption("Share of untouched hours flagged: " + ", ".join(f"{k} {v:.2%}" for k, v in fp.items()))
    sens = read_csv("validation_sensitivity.csv")
    if sens is not None:
        st.subheader("Threshold sensitivity")
        st.dataframe(sens, width="stretch", hide_index=True)