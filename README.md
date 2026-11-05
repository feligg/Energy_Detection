# Abnormal Energy Consumption Detection and Energy-Saving Analysis

Finds when and where electricity use in BDG2 education buildings is abnormally
high, and estimates how much energy could have been avoided.
**All savings are model-based estimates, not verified savings from a real control system.**

## Quick start

```bash
pip install -r requirements.txt
python scripts/download_data.py        # BDG2 files (handles Git LFS), ~195 MB
python run_pipeline.py                 # 60 most complete education buildings, ~3-4 min
streamlit run app/streamlit_app.py     # black-and-white dashboard
pytest -q                              # unit tests on synthetic data
```

Options: `--n 100` (more buildings), `--all` (every building, slow), `--skip-validation`.

## Folder structure

```
run_pipeline.py            Task 1 -> 5 end to end
scripts/download_data.py   downloads metadata, weather, electricity_cleaned
src/config.py              ALL thresholds and parameters
src/data_prep.py           Task 1  load, inspect, clean, build the shared dataset
src/baseline.py            season / academic-break calendar + expected "normal" use
src/anomaly_rules.py       Task 2  five anomaly types
src/detectors.py           Task 3  Z-score, IQR, Isolation Forest, residual + voting
src/evaluation.py          Task 4  comparison, events, cases to inspect + plots
src/savings.py             Task 5  saving potential (low / central / high)
src/validation.py          synthetic-anomaly injection + threshold sensitivity
app/streamlit_app.py       Task 6  dashboard
tests/                     pytest checks
```

Share `data/processed/clean_education_long.parquet` with Student 1 so both use the same cleaned data.
To use Student 1's forecasts put `data/external/student1_predictions.csv`
(columns `timestamp, building_id, predicted_kwh`) in place; they replace the built-in baseline.

## How it works

**Task 1 - data.** Education buildings only. Negative values, zero runs >= 24 h (meter outage),
flat-lines >= 48 h (stuck meter) and spikes > 50x median are removed, gaps <= 6 h interpolated,
buildings with > 20 % missing or a structural break dropped. Weather is de-duplicated and aligned.
Report: `outputs/results/data_quality_report.md`.

**Normal level (context).** A building is compared with itself: same *season*, same *day type*
(working day vs weekend / academic-break day) and same *hour*. Break days are found
automatically: a weekday on which the median building of a site runs below 80 % of its normal
weekday load. This stops summer cooling and term breaks from looking like anomalies.

**Task 2 - five anomaly types** (thresholds in `config.py`)

| type | rule |
|---|---|
| night_high | 22:00-05:59, robust z > 3 and >= 1.25x the normal level |
| weekend_high | weekend/break daytime hour >= 95 % of working-day level in a building that normally drops >= 10 % |
| sudden_increase | 1 h rise > Q3 + 3 IQR of the building's own rises and > 30 % of its median load |
| weather_deviation | actual >= 1.3x expected for similar weather and residual robust z > 3 |
| sustained_high | >= 6 consecutive hours with robust z > 2.5 and >= 1.2x normal |

Robust z = (x - median) / (1.4826 MAD).

**Task 3 - methods.** Z-score (season/day-type/hour context), IQR fence (season/weekday/hour
context), Isolation Forest (per building, calendar + weather + load features) and a residual
detector (actual vs expected, IQR fence on residuals). Z-score and IQR are one *statistical
family*; the final decision is a vote of three families: **statistical, Isolation Forest, residual**.

**Task 4 - evaluation.** Counts, time distribution, Jaccard agreement, rule confirmation, and
25 cases to inspect (`representative_cases.csv`): 20 largest events across the five types plus
5 *borderline* events to test for false alarms. Fill in the `verdict` and `reviewer_notes`
columns yourself after looking at each plot in `outputs/cases/`.

**Validation without labels.** `validation.py` injects known anomalies (night, weekend, sudden,
sustained) at +20 / 40 / 80 % into 12 real buildings and measures recall per method
(`validation_injection.csv`), plus the share of untouched hours flagged. It also varies the key
thresholds (`validation_sensitivity.csv`).

**Task 5 - savings.** Saving = actual - expected on anomaly hours (only when positive), reported
as three scenarios: *low* (all 3 families agree), *central* (>= 2), *high* (>= 2 of 4 methods).
Daily, monthly, % avoidable and a building ranking are in `outputs/results/savings_*.csv`.
Attention level is relative: top 10 % of buildings by avoidable share = high, next 20 % = medium.

## Limits to state in the report

- No confirmed labels exist; injected anomalies are a proxy, not proof of real-world accuracy.
- Expected use comes from a model, so errors in it become errors in the savings.
- The sample favours the most complete buildings (max 8 per site); use `--all` for the final run.
- Rule-type savings overlap (an hour can match several rules) and must not be added together.
