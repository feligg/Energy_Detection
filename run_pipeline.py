"""Run the whole project: Task 1 -> Task 5 (Task 6 is the Streamlit app).

    python run_pipeline.py                 # 60 most complete education buildings
    python run_pipeline.py --n 100         # more buildings
    python run_pipeline.py --all           # every education building (slow)
"""
import argparse
import json
import time

import pandas as pd

from src import config as C
from src import anomaly_rules, baseline, data_prep, detectors, evaluation, savings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=C.DEFAULT_N_BUILDINGS)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    step = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

    step("Task 1: loading, inspecting and cleaning data")
    df, quality, summary = data_prep.build_clean_dataset(None if args.all else args.n)
    print(json.dumps(summary, indent=2))

    step("Expected-consumption baseline (weather-aware, blocked cross-fitting)")
    df = baseline.add_calendar(df)
    df = baseline.add_expected(df)

    step("Task 2: rule-based anomaly types")
    df = anomaly_rules.apply_rules(df)

    step("Task 3: Z-score / IQR / Isolation Forest / residual detectors")
    df = detectors.apply_methods(df)

    step("Task 4: comparison, events and representative cases")
    comp = evaluation.compare_methods(df)
    print(comp["counts"].to_string(index=False))
    print(comp["jaccard"].to_string())
    events = evaluation.build_events(df)
    events.to_csv(C.RESULTS / "anomaly_events.csv", index=False)
    cases = evaluation.select_cases(events)
    cases.to_csv(C.RESULTS / "representative_cases.csv", index=False)
    evaluation.plot_cases(df, cases)

    step("Task 5: energy-saving potential")
    sv = savings.compute_savings(df)
    with open(C.RESULTS / "savings_overall.json", "w") as f:
        json.dump(sv["overall"], f, indent=2)
    print(json.dumps(sv["overall"], indent=2))
    print(sv["buildings"].head(10)[["building_id", "site_id", "saving_kwh", "avoidable_pct",
                                    "avg_monthly_saving_kwh", "dominant_anomaly_type"]]
          .to_string(index=False))

    step("Saving results for the Streamlit app")
    keep = ["timestamp", "building_id", "site_id", "sub_primaryspaceusage", "sqm", "kwh",
            "expected_kwh", "airTemperature", "hour", "dow", "month", "is_weekend",
            "robust_z", "iforest_score", "n_methods", "final_anomaly",
            *detectors_cols(), *anomaly_rules.TYPE_COLS]
    out = df[keep].copy()
    for c in out.select_dtypes("float64").columns:
        out[c] = out[c].astype("float32")
    out.to_parquet(C.RESULTS / "timeseries_flags.parquet", index=False)
    step("DONE - now run:  streamlit run app/streamlit_app.py")


def detectors_cols():
    return [f"m_{m}" for m in detectors.METHODS]


if __name__ == "__main__":
    main()
