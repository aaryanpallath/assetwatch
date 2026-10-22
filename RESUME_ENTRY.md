# assetwatch — Predictive Maintenance for a Hard Drive Fleet
Tech: Python, DuckDB, scikit-learn, FastAPI, SQLite, pytest, GitHub Actions
- Built a predictive-maintenance service on 60.8M real daily SMART readings from 348,272 Backblaze data-center drives, streaming 23 GB of CSVs to Parquet with DuckDB and serving risk-ranked work orders through a FastAPI REST API.
- Caught 13.4% of next-30-day failures by replacing just 0.1% of the fleet weekly vs 1.3% for the standard SMART rule (38% vs 12% at 1%), with a time-split test on 2.67M drive-weeks.
- Fixed inflated failure probabilities (42K expected vs 2.5K real) with isotonic calibration and enforced one open work order per drive with a SQLite partial unique index.
