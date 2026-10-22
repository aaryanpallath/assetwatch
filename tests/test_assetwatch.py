import csv
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from assetwatch import data, model, store
from assetwatch.api import create_app

D0 = date(2025, 10, 1)


def write_day(dir_, day, rows):
    cols = ["date", "serial_number", "model", "capacity_bytes", "failure"] + [f"smart_{s}_raw" for s in data.SMART]
    with open(dir_ / f"{day}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            smart = r.get("smart", {})
            w.writerow(
                [str(day), r["serial"], r.get("model", "ST16000NM001G"), r.get("cap", 16e12), r.get("failure", 0)]
                + [smart.get(s, 0) for s in data.SMART]
            )


@pytest.fixture
def telemetry(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for k in range(60):
        day = D0 + timedelta(days=k)
        rows = [{"serial": "NEVER"}, {"serial": "BOOT", "model": "CT250MX500SSD1", "cap": 250e9}]
        if k <= 20:  # fails on day 20: 10 days after the snapshot on day 10
            rows.append({"serial": "SOON", "failure": int(k == 20), "smart": {5: 10 + 3 * k, 197: 2 * k}})
        if k <= 50:  # fails on day 50: 40 days after the snapshot
            rows.append({"serial": "LATER", "failure": int(k == 50)})
        if k <= 10:  # fails on the snapshot day itself: not a row to predict for
            rows.append({"serial": "TODAY", "failure": int(k == 10)})
        write_day(raw, day, rows)
    return data.ingest(raw, tmp_path / "t.parquet")


def test_ingest_keeps_data_drives_only(telemetry):
    serials = set(pd.read_parquet(telemetry).serial)
    assert "BOOT" not in serials and {"NEVER", "SOON", "LATER", "TODAY"} <= serials


def test_labels_and_week_over_week_growth(telemetry):
    snap = data.snapshots(telemetry, [str(D0 + timedelta(days=10))]).set_index("serial")
    assert snap.fails_30d.to_dict() == {"NEVER": 0, "SOON": 1, "LATER": 0}  # TODAY failed on the day: excluded
    assert snap.loc["SOON", "smart_5"] == 40 and snap.loc["SOON", "d7_smart_5"] == 21  # (10+30) - (10+9)
    assert snap.loc["SOON", "d7_smart_197"] == 14
    assert snap.loc["NEVER", "manufacturer"] == "Seagate"


def frame(failing: list[int], errors: list[int], dates: int = 1) -> pd.DataFrame:
    rows = []
    for d in range(dates):
        for i, (y, e) in enumerate(zip(failing, errors, strict=True)):
            rows.append(
                {
                    "date": d,
                    "serial": f"S{i}",
                    "fails_30d": y,
                    **{f"smart_{s}": (e if s == 5 else 0) for s in data.SMART},
                }
            )
    return pd.DataFrame(rows)


def test_rule_and_budget_metrics():
    df = frame([1, 0, 1, 0, 0, 0, 0, 0, 0, 0], [50, 3, 0, 0, 0, 0, 0, 0, 0, 0])
    s = model.rule_score(df)
    assert s[0] > s[1] > 0 and s[2] == 0
    r = model.at_budget(df, s, 0.2)  # replace 2 of 10
    assert (r["replaced"], r["caught"], r["failures"]) == (2, 1, 2) and r["recall"] == 0.5
    op = model.rule_operating_point(df)
    assert op == {"flagged_pct": 20.0, "recall": 0.5, "precision": 0.5}


def test_explanations_name_the_counters_that_moved():
    row = pd.Series(
        {
            "smart_9": 54000.0,
            "smart_5": 120.0,
            "d7_smart_5": 40.0,
            "smart_197": 8.0,
            "d7_smart_197": 0.0,
            "age_years": 6.2,
        }
    )
    assert model.explain(row) == "120 reallocated sectors (+40 this week); 8 pending sectors; 6.2 years in service"
    assert model.explain(pd.Series({"smart_9": 9000.0, "age_years": 1.0})).startswith("elevated risk")
    assert model.explain(pd.Series({"smart_9": float("nan")})).startswith("stopped reporting SMART")
    assert model.explain(pd.Series({"smart_9": 1.0, "smart_188": 4.3e9, "age_years": 1.0})) == "command timeouts"


@pytest.fixture
def client(tmp_path):
    db = tmp_path / "s.sqlite"
    con = store.connect(db)
    rng = np.random.default_rng(0)
    with con:
        for i in range(20):
            con.execute(
                "INSERT INTO assets VALUES (?, 'ST16000NM001G', 'Seagate', 16, 3.0, '2026-03-31')", (f"D{i:02d}",)
            )
            con.execute(
                "INSERT INTO risk VALUES (?, '2026-03-31', ?, 'test')", (f"D{i:02d}", float(rng.uniform(0, 0.9)))
            )
    return TestClient(create_app(str(db)))


def test_planning_respects_budget_and_never_double_books(client):
    first = client.post("/work-orders/plan", json={"budget": 3}).json()["created"]
    risks = [o["risk"] for o in first]
    assert len(first) == 3 and risks == sorted(risks, reverse=True)
    top = client.get("/assets", params={"limit": 3}).json()
    assert [a["serial"] for a in top] == [o["serial"] for o in first]
    second = client.post("/work-orders/plan", json={"budget": 3}).json()["created"]
    assert not {o["serial"] for o in first} & {o["serial"] for o in second}
    assert len(client.get("/work-orders", params={"status": "open"}).json()) == 6


def test_work_order_lifecycle_and_validation(client):
    order = client.post("/work-orders/plan", json={"budget": 1}).json()["created"][0]
    assert client.patch(f"/work-orders/{order['id']}", json={"status": "completed"}).json()["status"] == "completed"
    assert client.patch(f"/work-orders/{order['id']}", json={"status": "cancelled"}).status_code == 409
    assert client.patch("/work-orders/999", json={"status": "completed"}).status_code == 404
    assert client.post("/work-orders/plan", json={"budget": 0}).status_code == 422
    assert client.get("/assets/bad serial!").status_code == 422
    assert client.get("/assets/NOPE").status_code == 404
    detail = client.get(f"/assets/{order['serial']}").json()
    assert detail["work_orders"][0]["status"] == "completed"
    s = client.get("/fleet/summary").json()
    assert s["drives"] == 20 and s["open_work_orders"] == 0
