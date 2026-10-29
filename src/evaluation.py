"""Task 4 - compare the detection methods and prepare >= 20 cases to inspect.

There are no confirmed anomaly labels in BDG2, so we cannot compute precision
or recall. Instead we measure: how many hours each method flags, WHEN they
flag them, how much the methods agree (Jaccard), and we examine concrete
events one by one (plots + a possible reason).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .anomaly_rules import TYPE_COLS
from .detectors import METHODS

MCOLS = [f"m_{m}" for m in METHODS]


# ------------------------------------------------------------ comparison
def compare_methods(df: pd.DataFrame) -> dict:
    v = df[df["kwh"].notna()]
    n_valid = len(v)
    out = {}

    rows = []
    for m in MCOLS + ["final_anomaly"] + TYPE_COLS:
        flagged = v[m].sum()
        rows.append({"detector": m.replace("m_", ""), "flagged_hours": int(flagged),
                     "pct_of_valid_hours": round(flagged / n_valid * 100, 3),
                     "buildings_with_flags": int(v.loc[v[m] == 1, "building_id"].nunique())})
    out["counts"] = pd.DataFrame(rows)

    # time distribution (share of each method's flags per hour / weekday / month)
    for key, col in [("hour", "hour"), ("weekday", "dow"), ("month", "month")]:
        t = v.groupby(col)[MCOLS + ["final_anomaly"]].sum()
        out[f"time_{key}"] = (t / t.sum()).round(4)

    # consistency between methods: Jaccard overlap
    jac = pd.DataFrame(index=METHODS, columns=METHODS, dtype=float)
    for a in METHODS:
        for b in METHODS:
            A, B = v[f"m_{a}"] == 1, v[f"m_{b}"] == 1
            union = (A | B).sum()
            jac.loc[a, b] = round((A & B).sum() / union, 3) if union else np.nan
    out["jaccard"] = jac

    out["agreement"] = (v[v["n_methods"] > 0].groupby("n_methods").size()
                        .rename("hours").to_frame())
    out["agreement"]["share"] = (out["agreement"]["hours"] / out["agreement"]["hours"].sum()).round(3)

    # how many rule hits are confirmed by >= 2 detection methods?
    r = []
    for t in TYPE_COLS:
        hit = v[v[t] == 1]
        r.append({"rule_type": t, "hours": len(hit),
                  "confirmed_by_>=2_methods_pct": round((hit["n_methods"] >= 2).mean() * 100, 1)
                  if len(hit) else np.nan})
    out["rule_confirmation"] = pd.DataFrame(r)

    for k, tbl in out.items():
        tbl.to_csv(C.RESULTS / f"eval_{k}.csv")
    _plot_comparison(out)
    return out


def _plot_comparison(out):
    cnt = out["counts"].set_index("detector").loc[[m.replace("m_", "") for m in MCOLS] + ["final_anomaly"]]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    cnt["flagged_hours"].plot.bar(ax=ax[0], color="#2a6f97")
    ax[0].set_title("Hours flagged per method"); ax[0].set_ylabel("hours")
    out["time_hour"].plot(ax=ax[1]); ax[1].set_title("When (hour of day) are anomalies flagged?")
    ax[1].set_xlabel("hour"); ax[1].set_ylabel("share of method's flags")
    j = out["jaccard"].astype(float)
    im = ax[2].imshow(j.values, cmap="Blues", vmin=0, vmax=1)
    ax[2].set_xticks(range(len(j))); ax[2].set_xticklabels(j.columns, rotation=30)
    ax[2].set_yticks(range(len(j))); ax[2].set_yticklabels(j.index)
    for (a, b), val in np.ndenumerate(j.values):
        ax[2].text(b, a, f"{val:.2f}", ha="center", va="center")
    ax[2].set_title("Agreement between methods (Jaccard)")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    fig.tight_layout(); fig.savefig(C.FIGURES / "method_comparison.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    out["time_weekday"].plot(ax=ax[0]); ax[0].set_title("By weekday (0 = Monday)")
    out["time_month"].plot(ax=ax[1]); ax[1].set_title("By month")
    fig.tight_layout(); fig.savefig(C.FIGURES / "time_distribution.png", dpi=140); plt.close(fig)


# ---------------------------------------------------------------- events
def build_events(df: pd.DataFrame) -> pd.DataFrame:
    d = df[["building_id", "timestamp", "kwh", "expected_kwh", "airTemperature",
            "final_anomaly", "n_methods"] + MCOLS + TYPE_COLS].copy()
    d["excess"] = (d["kwh"] - d["expected_kwh"]).clip(lower=0)
    flag = d["final_anomaly"] == 1
    run = (flag != flag.groupby(d["building_id"]).shift()).cumsum()
    ev = d[flag].copy()
    ev["event_id"] = run[flag]
    agg = ev.groupby("event_id").agg(
        building_id=("building_id", "first"), start=("timestamp", "min"),
        end=("timestamp", "max"), duration_h=("timestamp", "size"),
        actual_kwh=("kwh", "sum"), expected_kwh=("expected_kwh", "sum"),
        total_excess_kwh=("excess", "sum"), peak_excess_kwh=("excess", "max"),
        mean_temp_c=("airTemperature", "mean"),
        **{t: (t, "sum") for t in TYPE_COLS}, **{m: (m, "sum") for m in MCOLS})
    agg["primary_type"] = agg[TYPE_COLS].idxmax(axis=1).where(agg[TYPE_COLS].sum(axis=1) > 0,
                                                            "unclassified")
    agg["methods"] = agg[MCOLS].gt(0).apply(
        lambda r: "+".join(m.replace("m_", "") for m in MCOLS if r[m]), axis=1)
    agg["weekday"] = agg["start"].dt.day_name()
    agg["start_hour"] = agg["start"].dt.hour
    agg["possible_reason"] = agg.apply(_reason, axis=1)
    return agg.reset_index(drop=True).sort_values("total_excess_kwh", ascending=False)


def _reason(r) -> str:
    t, temp = r["primary_type"], r["mean_temp_c"]
    weather = ("hot weather - cooling plant / chiller running harder or left at occupied setpoint"
               if temp >= 24 else
               "cold weather - electric heating / heat pumps or fans over-running"
               if temp <= 5 else "mild weather - the load is not weather driven")
    return {
        "night_high": "Load stays high at night: HVAC, lighting or lab/IT equipment probably not "
                      "scheduled off (check night set-back schedule) or an unplanned night event.",
        "weekend_high": "Weekend load at weekday level: weekday HVAC/lighting schedule also ran on "
                        "the weekend, or a weekend class / exam / event.",
        "sudden_increase": "Abrupt step in load: equipment (chiller, boiler, pumps, fans) switched "
                           "on, schedule started earlier than usual, or a control override.",
        "weather_deviation": f"Higher than expected for similar weather; {weather}.",
        "sustained_high": f"Elevated load for {r['duration_h']} h: equipment left running or a manual "
                          "override; " + weather + ".",
    }.get(t, f"Several methods agree on unusually high load; {weather}.")


def select_cases(events: pd.DataFrame, n: int = C.N_CASES) -> pd.DataFrame:
    """Top events per anomaly type (different buildings), then fill by excess."""
    ev = events[events["duration_h"] >= 1]
    per_type = max(3, n // 5)
    used = {}                                    # max 2 cases per building

    def take(sub, k):
        rows = []
        for idx, r in sub.iterrows():
            if used.get(r["building_id"], 0) < 2 and len(rows) < k:
                used[r["building_id"]] = used.get(r["building_id"], 0) + 1
                rows.append(idx)
        return sub.loc[rows]

    picks = [take(ev[ev["primary_type"] == t].drop_duplicates("building_id"), per_type)
             for t in TYPE_COLS + ["unclassified"]]
    cases = pd.concat(picks)
    if len(cases) < n:
        rest = ev.drop(cases.index, errors="ignore")
        cases = pd.concat([cases, take(rest, n - len(cases))])
    cases = cases.head(max(n, 20)).reset_index(drop=True)
    cases.insert(0, "case_id", [f"case_{i + 1:02d}" for i in range(len(cases))])
    cases["reviewer_notes"] = ""        # <- the student writes the manual verdict here
    return cases


def plot_cases(df: pd.DataFrame, cases: pd.DataFrame, pad_days: int = 4):
    for old in C.CASES.glob("case_*.png"):       # remove plots of earlier runs
        old.unlink()
    by_b = {b: g.set_index("timestamp") for b, g in df.groupby("building_id")}
    for _, c in cases.iterrows():
        g = by_b[c["building_id"]]
        lo = c["start"] - pd.Timedelta(days=pad_days)
        hi = c["end"] + pd.Timedelta(days=pad_days)
        w = g.loc[lo:hi]
        fig, ax = plt.subplots(figsize=(11, 3.6))
        ax.plot(w.index, w["kwh"], color="#1d3557", lw=1.2, label="actual")
        ax.plot(w.index, w["expected_kwh"], color="#2a9d8f", lw=1.0, ls="--", label="expected")
        a = w[w["final_anomaly"] == 1]
        ax.scatter(a.index, a["kwh"], color="#e63946", s=14, zorder=3, label="final anomaly")
        ax.axvspan(c["start"], c["end"] + pd.Timedelta(hours=1), color="#e63946", alpha=.12)
        ax.set_title(f"{c['case_id']} | {c['building_id']} | {c['primary_type']} | "
                     f"{c['start']:%Y-%m-%d %H:%M} ({c['duration_h']} h, "
                     f"+{c['total_excess_kwh']:.0f} kWh)", fontsize=9)
        ax.set_ylabel("kWh / hour"); ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(C.CASES / f"{c['case_id']}_{c['building_id']}.png", dpi=110)
        plt.close(fig)