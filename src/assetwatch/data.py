"""Backblaze drive telemetry -> weekly snapshots with features and a 30-day failure label.

Backblaze publishes one CSV per day with a row for every drive in its data centers: serial number, model, capacity,
whether the drive failed that day, and its SMART counters. DuckDB streams the CSVs from disk, so a quarter (about
30 million rows) never has to fit in memory.

For each snapshot date d and each hard drive reporting that day:
  features  the drive's SMART counters on d, how much the error counters grew since d - 7, its age (power-on hours),
            capacity and manufacturer
  label     1 if the drive's failure is recorded in (d, d + 30 days]
Boot SSDs and small drives are excluded: the fleet being maintained is the data drives.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

SMART = [5, 9, 12, 187, 188, 193, 194, 197, 198, 199, 241, 242]
ERROR_COUNTERS = [5, 187, 188, 197, 198]  # the counters Backblaze reports as predictive of failure
GROWTH = [5, 187, 197, 198, 199]  # week-over-week change matters more than the level
HORIZON_DAYS = 30


def connect(memory_limit: str = "3GB") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{memory_limit}'")
    return con


def ingest(raw_dir: str | Path, out: str | Path, con: duckdb.DuckDBPyConnection | None = None) -> Path:
    """All daily CSVs under raw_dir -> one Parquet file with only the columns used, hard drives only."""
    con = con or connect()
    out = Path(out)
    cols = ", ".join(f"TRY_CAST(smart_{s}_raw AS DOUBLE) AS smart_{s}" for s in SMART)
    glob = str(Path(raw_dir) / "**" / "*.csv")
    con.execute(f"""
        COPY (
          SELECT CAST(date AS DATE) AS date, serial_number AS serial, model,
                 TRY_CAST(capacity_bytes AS DOUBLE) / 1e12 AS capacity_tb, CAST(failure AS INTEGER) AS failure, {cols}
          FROM read_csv('{glob}', union_by_name = true, header = true, ignore_errors = true)
          WHERE TRY_CAST(capacity_bytes AS DOUBLE) >= 4e12            -- data drives; boot SSDs are small
            AND model NOT ILIKE '%ssd%' AND model NOT ILIKE 'DELLBOSS%'
        ) TO '{out}' (FORMAT parquet)""")
    return out


def snapshots(telemetry: str | Path, dates: list[str], con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """Feature rows for every drive on each snapshot date, with the 30-day failure label."""
    con = con or connect()
    t = str(telemetry)
    growth = ", ".join(
        f"GREATEST(cur.smart_{s} - COALESCE(prev.smart_{s}, cur.smart_{s}), 0) AS d7_smart_{s}" for s in GROWTH
    )
    level = ", ".join(f"cur.smart_{s}" for s in SMART)
    frames = []
    for d in dates:
        frames.append(
            con.execute(f"""
            WITH failures AS (
              SELECT serial, MIN(date) AS failed_on FROM '{t}' WHERE failure = 1 GROUP BY serial
            )
            SELECT cur.date, cur.serial, cur.model, cur.capacity_tb, {level}, {growth},
                   CASE WHEN f.failed_on > cur.date AND f.failed_on <= cur.date + INTERVAL {HORIZON_DAYS} DAY
                        THEN 1 ELSE 0 END AS fails_30d,
                   f.failed_on
            FROM '{t}' cur
            LEFT JOIN '{t}' prev ON prev.serial = cur.serial AND prev.date = cur.date - INTERVAL 7 DAY
            LEFT JOIN failures f ON f.serial = cur.serial
            WHERE cur.date = DATE '{d}' AND cur.failure = 0""").df()
        )
    df = pd.concat(frames, ignore_index=True)
    df["manufacturer"] = df.model.map(manufacturer)
    df["age_years"] = df.smart_9 / 8766
    return df


def manufacturer(model: str) -> str:
    m = str(model).upper()
    if m.startswith(("ST", "SEAGATE")):
        return "Seagate"
    if m.startswith(("HGST", "HUH", "HUS", "HMS")):
        return "HGST"
    if m.startswith(("WDC", "WUH", "WD")):
        return "WDC"
    if m.startswith("TOSHIBA") or m.startswith("MG"):
        return "Toshiba"
    return "Other"
