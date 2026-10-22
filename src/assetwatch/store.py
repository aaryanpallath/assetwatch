"""SQLite storage for the maintenance service: assets, their latest risk, and work orders.

The database enforces the one rule that matters for planning: a drive can have at most one open work order (a
partial unique index), so running the planner twice never doubles up."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
  serial TEXT PRIMARY KEY, model TEXT NOT NULL, manufacturer TEXT NOT NULL, capacity_tb REAL NOT NULL,
  age_years REAL, last_seen TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS risk (
  serial TEXT PRIMARY KEY REFERENCES assets(serial), scored_on TEXT NOT NULL, probability REAL NOT NULL,
  reason TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS risk_by_probability ON risk(probability DESC);
CREATE TABLE IF NOT EXISTS work_orders (
  id INTEGER PRIMARY KEY AUTOINCREMENT, serial TEXT NOT NULL REFERENCES assets(serial), created_at TEXT NOT NULL,
  priority TEXT NOT NULL CHECK (priority IN ('urgent', 'high', 'planned')), reason TEXT NOT NULL,
  risk_at_creation REAL NOT NULL,
  status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'completed', 'cancelled')), closed_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS one_open_order_per_asset ON work_orders(serial) WHERE status = 'open';
"""


def connect(path: str | Path) -> sqlite3.Connection:
    con = sqlite3.connect(path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    return con
