# Energy_Detection
energy_anomaly_project/
├── README.md
├── requirements.txt
├── run_pipeline.py              # runs Task 1 -> Task 5 end to end
├── scripts/
│   └── download_data.py         # downloads the 3 BDG2 files (handles Git LFS)
├── src/
│   ├── config.py                # ALL thresholds and parameters
│   ├── data_prep.py             # Task 1  load, inspect, clean, build shared dataset
│   ├── baseline.py              # expected "normal" consumption (weather-aware)
│   ├── anomaly_rules.py         # Task 2  5 anomaly types
│   ├── detectors.py             # Task 3  Z-score, IQR, Isolation Forest, residual
│   ├── evaluation.py            # Task 4  comparison, events, >= 20 cases + plots
│   └── savings.py               # Task 5  saving potential
├── app/
│   └── streamlit_app.py         # Task 6  dashboard
├── data/
│   ├── raw/                     # electricity_cleaned.csv, metadata.csv, weather.csv
│   ├── processed/               # clean_education_long.parquet  <- share with Student 1
│   └── external/                # student1_predictions.csv (optional)
└── outputs/
    ├── results/                 # CSV / JSON / parquet results
    ├── figures/                 # method comparison plots
    └── cases/                   # one PNG per inspected case