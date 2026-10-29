"""Central configuration: paths, thresholds and parameters.

Every anomaly threshold used in the project lives here, so the rules in
Task 2 and the methods in Task 3 can be explained (and tuned) in one place.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
EXTERNAL = ROOT / "data" / "external"
OUT = ROOT / "outputs"
RESULTS = OUT / "results"
FIGURES = OUT / "figures"
CASES = OUT / "cases"

for _p in (RAW, PROCESSED, EXTERNAL, RESULTS, FIGURES, CASES):
    _p.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------- raw files
ELECTRICITY_FILE = RAW / "electricity_cleaned.csv"
METADATA_FILE = RAW / "metadata.csv"
WEATHER_FILE = RAW / "weather.csv"
# Optional: predictions produced by Student 1 (long format, see README)
STUDENT1_FILE = EXTERNAL / "student1_predictions.csv"

# --------------------------------------------------------- Task 1: cleaning
USAGE = "Education"            # primaryspaceusage selected for the project
ZERO_RUN_HOURS = 24            # >= 24 consecutive zero hours -> meter outage
FLATLINE_HOURS = 48            # >= 48 identical non-zero values -> stuck meter
SPIKE_MULTIPLIER = 50          # > 50 x building median -> impossible spike
INTERP_LIMIT_HOURS = 6         # fill only short gaps (<= 6 h)
MAX_MISSING_FRACTION = 0.20    # drop building if > 20 % missing after cleaning
REGIME_RATIO = 4.0             # same calendar month, other year: median differs >= 4x
REGIME_MIN_MONTHS = 3          # ... in >= 3 months -> structural break, drop building
WEATHER_COLS = ["airTemperature", "dewTemperature", "windSpeed",
                "cloudCoverage", "seaLvlPressure"]

# ------------------------------------------------------ Task 2: rule engine
NIGHT_HOURS = [22, 23, 0, 1, 2, 3, 4, 5]     # local building time
ROBUST_Z = 3.0                 # robust z-score threshold (median / MAD)
MIN_EXCESS_RATIO = 1.25        # must also be >= 1.25 x normal level
WEEKEND_AS_WEEKDAY_RATIO = 0.95   # weekend hour >= 95 % of weekday level
WEEKEND_DROP_RATIO = 0.90      # building normally drops >= 10 % on weekends
SUDDEN_IQR_K = 3.0             # jump > Q3 + 3*IQR of hourly increases
SUDDEN_MIN_RATIO = 0.30        # ... and > 30 % of building median load
WEATHER_RATIO = 1.30           # actual >= 1.3 x expected (similar weather)
SUSTAINED_Z = 2.5              # "high" = robust z > 2.5 vs same hour/day-type
SUSTAINED_RATIO = 1.20
SUSTAINED_HOURS = 6            # ... for >= 6 consecutive hours

# ----------------------------------------------------- Task 3: three methods
ZSCORE_THRESHOLD = 3.0
IQR_K = 1.5
IFOREST_CONTAMINATION = 0.02
IFOREST_ESTIMATORS = 200
RANDOM_STATE = 42
MIN_METHODS_FOR_FINAL = 2      # final anomaly = flagged by >= 2 methods

# ---------------------------------------------------------- baseline model
CV_BLOCKS = 4                  # blocked time-series cross-fitting
BASELINE_PARAMS = dict(loss="absolute_error", max_iter=150, learning_rate=0.08,
                       max_leaf_nodes=31, min_samples_leaf=40,
                       random_state=RANDOM_STATE)

# ----------------------------------------------------------------- sampling
DEFAULT_N_BUILDINGS = 60       # buildings used by default (use --all for all)
MAX_PER_SITE = 8               # keeps the sample spread across sites
N_CASES = 25                   # representative cases to inspect (>= 20)