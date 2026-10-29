"""Task 1 - understand, inspect, check and clean the BDG2 education data.

Output: one tidy "long" table (one row per building-hour) that is also the
file to share with Student 1 so both students use the same cleaned dataset.
"""
import json

import numpy as np
import pandas as pd

from . import config as C

# Variable dictionary (BDG2 documentation) -> used in the data-quality report
VARIABLES = {
    "timestamp": "Hourly local time of the site (no timezone conversion needed)",
    "electricity (kwh)": "Electricity meter reading, kWh per hour, per building",
    "airTemperature": "Outdoor dry-bulb temperature, degrees Celsius",
    "dewTemperature": "Dew-point temperature, degrees Celsius",
    "windSpeed": "Wind speed, metres per second",
    "cloudCoverage": "Cloud cover, oktas (0 = clear, 8 = overcast)",
    "seaLvlPressure": "Sea-level pressure, millibar (hPa)",
    "sqm": "Gross floor area of the building, square metres",
    "primaryspaceusage": "Main building use (we keep 'Education')",
}


# ------------------------------------------------------------------ loading
def load_raw():
    for f in (C.ELECTRICITY_FILE, C.METADATA_FILE, C.WEATHER_FILE):
        if not f.exists() or f.stat().st_size < 10_000:
            raise FileNotFoundError(
                f"{f} missing or a Git-LFS pointer. Run: python scripts/download_data.py")
    meta = pd.read_csv(C.METADATA_FILE)
    elec = pd.read_csv(C.ELECTRICITY_FILE, index_col="timestamp", parse_dates=True)
    weather = pd.read_csv(C.WEATHER_FILE, parse_dates=["timestamp"])
    return meta, elec, weather


def select_education(meta, elec):
    edu = meta[(meta["primaryspaceusage"] == C.USAGE) & meta["building_id"].isin(elec.columns)]
    return edu.reset_index(drop=True), elec[edu["building_id"].tolist()].copy()


# ------------------------------------------------------- quality inspection
def _mark_runs(mask: pd.DataFrame, n: int) -> pd.DataFrame:
    """True for every row that belongs to >= n consecutive True values."""
    full = mask.astype(int).rolling(n).sum() == n          # window END points
    back = full.iloc[::-1].astype(int).rolling(n, min_periods=1).max().iloc[::-1]
    return back.astype(bool)


def inspect_and_clean(wide: pd.DataFrame):
    """Return (clean wide table, per-building quality table, global summary)."""
    q = pd.DataFrame(index=wide.columns)
    q["n_hours"] = len(wide)
    q["missing_raw_pct"] = wide.isna().mean() * 100
    q["n_negative"] = (wide < 0).sum()
    q["n_zero"] = (wide == 0).sum()

    x = wide.copy()
    x[x < 0] = np.nan                                       # impossible values

    zero_runs = _mark_runs(x == 0, C.ZERO_RUN_HOURS)        # meter outage
    q["n_zero_run_hours"] = zero_runs.sum()
    x = x.mask(zero_runs)

    same = (x.diff() == 0) & (x > 0)                        # stuck meter
    flat = _mark_runs(same, C.FLATLINE_HOURS)
    q["n_flatline_hours"] = flat.sum()
    x = x.mask(flat)

    med = x.median()
    spikes = x > (med * C.SPIKE_MULTIPLIER)                 # impossible spike
    q["n_spike_hours"] = spikes.sum()
    x = x.mask(spikes)

    x = x.interpolate(method="time", limit=C.INTERP_LIMIT_HOURS, limit_area="inside")
    q["missing_clean_pct"] = x.isna().mean() * 100
    q["median_kwh"] = x.median()

    # structural break: the SAME calendar month has a >= 4x different median in
    # the two years (meter replaced / re-scaled, building closed...). A baseline
    # learned from the other year would call the whole month "anomalous".
    m = x.resample("MS").median()
    gm = m.groupby(m.index.month)
    ratio = gm.max() / gm.min().clip(lower=1e-3)
    q["regime_change_months"] = (ratio > C.REGIME_RATIO).sum()

    keep = ((q["missing_clean_pct"] <= C.MAX_MISSING_FRACTION * 100) & (q["median_kwh"] > 0)
            & (q["regime_change_months"] < C.REGIME_MIN_MONTHS))
    q["kept"] = keep

    summary = {
        "duplicate_timestamps_electricity": int(wide.index.duplicated().sum()),
        "duplicate_building_columns": int(wide.columns.duplicated().sum()),
        "regular_hourly_index": bool((wide.index.to_series().diff().dropna() == pd.Timedelta("1h")).all()),
        "buildings_raw": int(wide.shape[1]),
        "buildings_kept": int(keep.sum()),
        "buildings_dropped": int((~keep).sum()),
        "dropped_for_regime_change": int((q["regime_change_months"] >= C.REGIME_MIN_MONTHS).sum()),
        "total_negative_values": int(q["n_negative"].sum()),
        "total_zero_run_hours": int(q["n_zero_run_hours"].sum()),
        "total_flatline_hours": int(q["n_flatline_hours"].sum()),
        "total_spike_hours": int(q["n_spike_hours"].sum()),
        "missing_raw_pct_overall": round(float(wide.isna().mean().mean() * 100), 2),
        "missing_clean_pct_kept": round(float(x.loc[:, keep].isna().mean().mean() * 100), 2),
    }
    return x.loc[:, keep], q, summary


def clean_weather(weather: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Deduplicate, put on a complete hourly grid, fill short gaps per site."""
    dup = int(weather.duplicated(["site_id", "timestamp"]).sum())
    w = weather.groupby(["site_id", "timestamp"], as_index=False)[C.WEATHER_COLS].mean()
    parts = []
    for site, g in w.groupby("site_id"):
        g = g.set_index("timestamp").reindex(index)
        g[C.WEATHER_COLS] = g[C.WEATHER_COLS].interpolate(
            method="time", limit=C.INTERP_LIMIT_HOURS, limit_area="inside")
        g["site_id"] = site
        parts.append(g.rename_axis("timestamp").reset_index())
    out = pd.concat(parts, ignore_index=True)
    out.attrs["duplicate_rows_removed"] = dup
    return out


# --------------------------------------------------------- building sample
def sample_buildings(meta_edu: pd.DataFrame, clean_wide: pd.DataFrame, n: int | None):
    """Pick the n most complete buildings, at most MAX_PER_SITE per site."""
    if n is None or n >= clean_wide.shape[1]:
        return clean_wide.columns.tolist()
    comp = clean_wide.notna().mean().rename("complete")
    m = meta_edu.set_index("building_id").join(comp, how="inner").reset_index()
    m = m.sort_values(["complete", "sqm"], ascending=False)
    m["rank_in_site"] = m.groupby("site_id").cumcount()
    m = m[m["rank_in_site"] < C.MAX_PER_SITE]
    return m.head(n)["building_id"].tolist()


# ------------------------------------------------------------- build / save
def build_clean_dataset(n_buildings: int | None = C.DEFAULT_N_BUILDINGS):
    meta, elec, weather = load_raw()
    meta_edu, wide = select_education(meta, elec)
    clean_wide, quality, summary = inspect_and_clean(wide)
    chosen = sample_buildings(meta_edu, clean_wide, n_buildings)
    clean_wide = clean_wide[chosen]
    summary["buildings_used_in_analysis"] = len(chosen)

    w = clean_weather(weather, clean_wide.index)
    summary["weather_duplicate_rows_removed"] = w.attrs["duplicate_rows_removed"]
    summary["weather_missing_pct"] = {c: round(float(w[c].isna().mean() * 100), 2)
                                      for c in C.WEATHER_COLS}

    long = (clean_wide.rename_axis("timestamp").reset_index()
            .melt(id_vars="timestamp", var_name="building_id", value_name="kwh"))
    info = meta_edu[["building_id", "site_id", "sub_primaryspaceusage", "sqm"]]
    long = long.merge(info, on="building_id", how="left")
    long = long.merge(w, on=["site_id", "timestamp"], how="left")
    long = long.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    for c in ["kwh", "sqm"] + C.WEATHER_COLS:
        long[c] = long[c].astype("float32")

    # ---- persist (these two files are what you share with Student 1) ----
    long.to_parquet(C.PROCESSED / "clean_education_long.parquet", index=False)
    clean_wide.to_csv(C.PROCESSED / "clean_education_wide.csv.gz")
    quality.to_csv(C.RESULTS / "data_quality_per_building.csv")
    with open(C.RESULTS / "data_quality_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    write_quality_markdown(summary, quality)
    return long, quality, summary


def write_quality_markdown(summary: dict, quality: pd.DataFrame):
    lines = ["# Data quality report (Task 1)\n", "## Variables and units\n"]
    lines += [f"- **{k}** - {v}" for k, v in VARIABLES.items()]
    lines += ["\n## Global checks\n"]
    lines += [f"- {k}: `{v}`" for k, v in summary.items()]
    worst = quality.sort_values("missing_raw_pct", ascending=False).head(10)
    lines += ["\n## 10 buildings with most missing raw data\n",
              worst[["missing_raw_pct", "n_zero_run_hours", "n_flatline_hours",
                     "missing_clean_pct", "regime_change_months", "kept"]].round(2).to_markdown()]
    (C.RESULTS / "data_quality_report.md").write_text("\n".join(lines))