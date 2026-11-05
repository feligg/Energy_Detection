"""Run the whole project: Task 1 -> Task 5 (Task 6 is the Streamlit app).

    python run_pipeline.py                 # 60 most complete education buildings
    python run_pipeline.py --n 100         # more buildings
    python run_pipeline.py --all           # every education building (slow)
    python run_pipeline.py --skip-validation
"""
import argparse
import json
import time

from src import config as C
from src import anomaly_rules, baseline, data_prep, detectors, evaluation, savings, validation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=C.DEFAULT_N_BUILDINGS)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--skip-validation", action="store_true",
                    help="skip the synthetic-injection and sensitivity experiments")
    args = ap.parse_args()
    t0 = time.time()

    def step(m):
        print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

    step("Task 1: loading, inspecting and cleaning data")
    df, quality, summary = data_prep.build_clean_dataset(None if args.all else args.n)
    print(json.dumps(summary, indent=2))

    step("Calendar (season, academic-break days) + expected-consumption baseline")
    df = baseline.add_calendar(df)
    df = baseline.add_expected(df)
    print(f"  break weekdays flagged: {df['is_break'].mean() * 100:.1f} % of hours")

    step("Task 2: rule-based anomaly types")
    df = anomaly_rules.apply_rules(df)

    step("Task 3: Z-score / IQR / Isolation Forest / residual detectors")
    df = detectors.apply_methods(df)

    step("Task 4: comparison, events and cases to inspect")
    comp = evaluation.compare_methods(df)
    print(comp["counts"].to_string(index=False))
    print(comp["jaccard"].to_string())
    events = evaluation.build_events(df)
    events.to_csv(C.RESULTS / "anomaly_events.csv", index=False)
    cases = evaluation.select_cases(events)
    cases.to_csv(C.RESULTS / "representative_cases.csv", index=False)
    evaluation.plot_cases(df, cases)

    step("Task 5: energy-saving potential (low / central / high)")
    sv = savings.compute_savings(df)
    with open(C.RESULTS / "savings_overall.json", "w") as f:
        json.dump(sv["overall"], f, indent=2)
    print(json.dumps(sv["overall"], indent=2))

    if not args.skip_validation:
        step("Validation: synthetic anomaly injection")
        inj = validation.run_injection_experiment(df)
        print(inj[inj["level"] == C.VAL_LEVELS[1]].pivot(
            index="inj_type", columns="detector", values="event_recall").to_string())
        step("Validation: threshold sensitivity")
        print(validation.run_sensitivity(df).to_string(index=False))

    step("Saving results for the Streamlit app")
    df = savings.add_saving_columns(df)
    keep = ["timestamp", "building_id", "site_id", "sub_primaryspaceusage", "sqm", "kwh",
            "expected_kwh", "airTemperature", "hour", "dow", "month", "is_weekend", "is_off",
            "robust_z", "iforest_score", "n_methods", "n_families", "final_anomaly",
            "final_low", "final_high", "saving_low_kwh", "saving_kwh", "saving_high_kwh",
            *[f"m_{m}" for m in detectors.METHODS], *anomaly_rules.TYPE_COLS]
    out = df[keep].copy()
    for c in out.select_dtypes("float64").columns:
        out[c] = out[c].astype("float32")
    out.to_parquet(C.RESULTS / "timeseries_flags.parquet", index=False)
    step("DONE - now run:  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
