"""assetwatch ingest | evaluate | load

assetwatch ingest   --raw data/raw --out data/telemetry.parquet
assetwatch evaluate --telemetry data/telemetry.parquet        # results/eval.json, results/budget.png
assetwatch load     --telemetry data/telemetry.parquet --db data/assetwatch.sqlite
ASSETWATCH_DB=data/assetwatch.sqlite uvicorn --factory assetwatch.api:create_app
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import data, model, store

# weekly snapshots: each needs the week before it (for growth) and 30 days after it (for the label) inside the data
TRAIN = [str(date(2025, 10, 8) + timedelta(weeks=i)) for i in range(8)]  # Oct 8 - Nov 26, 2025
TEST = [str(date(2026, 1, 7) + timedelta(weeks=i)) for i in range(8)]  # Jan 7 - Feb 25, 2026


def evaluate(telemetry: str, out_dir: Path) -> dict:
    con = data.connect()
    fit, cal, test = (data.snapshots(telemetry, d, con) for d in (TRAIN[:-2], TRAIN[-2:], TEST))
    clf = model.classifier().fit(fit[model.FEATURES], fit.fails_30d)
    raw = clf.predict_proba(test[model.FEATURES])[:, 1]
    iso = model.calibrator(clf.predict_proba(cal[model.FEATURES])[:, 1], cal.fails_30d.to_numpy())
    train = pd.concat([fit, cal])
    # rank by the raw score (calibration steps create ties); report probabilities from the calibrated one
    scores = {"SMART rule (5, 187, 188, 197, 198 > 0)": model.rule_score(test), "gradient boosting": raw}
    result = {
        "train": {"snapshots": TRAIN, "drive_weeks": len(train), "failures_within_30d": int(train.fails_30d.sum())},
        "test": {
            "snapshots": TEST,
            "drive_weeks": len(test),
            "failures_within_30d": int(test.fails_30d.sum()),
            "drives_per_snapshot": int(len(test) / len(TEST)),
        },
        "rule_as_alert": model.rule_operating_point(test),
        # the model given exactly as many alerts as the rule raises
        "model_at_rule_alert_volume": model.at_budget(
            test, scores["gradient boosting"], model.rule_operating_point(test)["flagged_pct"] / 100
        ),
        "models": model.evaluate(test, scores),
        "calibration": model.calibration(test, iso.predict(raw)),
    }
    out_dir.mkdir(exist_ok=True)
    (out_dir / "eval.json").write_text(json.dumps(result, indent=2) + "\n")
    _plot(result, out_dir / "budget.png")
    return result


def _plot(result: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, r in result["models"].items():
        b = r["budgets"]
        ax.plot([x["budget_pct"] for x in b], [100 * x["recall"] for x in b], "o-", label=name)
    ax.set(
        xlabel="drives replaced each week (% of fleet)",
        ylabel="failures in the next 30 days caught (%)",
        title="What a fixed replacement budget catches (Jan-Feb 2026)",
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def load(telemetry: str, db: str, model_path: Path) -> int:
    """Train on every labeled snapshot, score the fleet on the last day of data, and write it to the service DB."""
    con = data.connect()
    fit, cal = data.snapshots(telemetry, TRAIN + TEST[:-2], con), data.snapshots(telemetry, TEST[-2:], con)
    clf = model.classifier().fit(fit[model.FEATURES], fit.fails_30d)
    iso = model.calibrator(clf.predict_proba(cal[model.FEATURES])[:, 1], cal.fails_30d.to_numpy())
    joblib.dump({"model": clf, "calibrator": iso}, model_path)
    last = con.execute(f"SELECT MAX(date) FROM '{telemetry}'").fetchone()[0]
    fleet = data.snapshots(telemetry, [str(last)], con)
    fleet["probability"] = iso.predict(clf.predict_proba(fleet[model.FEATURES])[:, 1])
    s = store.connect(db)
    with s:
        s.executemany(
            """INSERT INTO assets VALUES (?, ?, ?, ?, ?, ?)
                         ON CONFLICT(serial) DO UPDATE SET model = excluded.model, age_years = excluded.age_years,
                           last_seen = excluded.last_seen""",
            [
                (r.serial, r.model, r.manufacturer, float(r.capacity_tb), _f(r.age_years), str(r.date.date()))
                for r in fleet.itertuples()
            ],
        )
        s.executemany(
            """INSERT INTO risk VALUES (?, ?, ?, ?)
                         ON CONFLICT(serial) DO UPDATE SET scored_on = excluded.scored_on,
                           probability = excluded.probability, reason = excluded.reason""",
            [
                (row.serial, str(row.date.date()), float(row.probability), model.explain(row))
                for _, row in fleet.iterrows()
            ],
        )
    return len(fleet)


def _f(v) -> float | None:
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="assetwatch")
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest")
    i.add_argument("--raw", default="data/raw")
    i.add_argument("--out", default="data/telemetry.parquet")
    e = sub.add_parser("evaluate")
    e.add_argument("--telemetry", default="data/telemetry.parquet")
    e.add_argument("--out", default="results")
    lo = sub.add_parser("load")
    lo.add_argument("--telemetry", default="data/telemetry.parquet")
    lo.add_argument("--db", default="data/assetwatch.sqlite")
    lo.add_argument("--model", default="data/model.joblib")
    a = p.parse_args(argv)
    if a.cmd == "ingest":
        print(data.ingest(a.raw, a.out))
    elif a.cmd == "evaluate":
        print(json.dumps(evaluate(a.telemetry, Path(a.out)), indent=2))
    else:
        print(f"scored {load(a.telemetry, a.db, Path(a.model)):,} drives into {a.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
