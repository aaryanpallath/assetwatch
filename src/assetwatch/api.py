"""REST API for the maintenance team.

GET   /fleet/summary                 fleet size, expected failures in 30 days, open work orders
GET   /assets?min_risk=&manufacturer=&limit=&offset=   drives ranked by 30-day failure risk
GET   /assets/{serial}               one drive: details, risk and the reason, its work orders
POST  /work-orders/plan              open work orders for the riskiest drives, within a budget
GET   /work-orders?status=           work orders
PATCH /work-orders/{id}              complete or cancel a work order
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Literal

from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, Field

from . import store

SERIAL = r"^[A-Za-z0-9._-]{1,40}$"


class PlanRequest(BaseModel):
    budget: int = Field(ge=1, le=5000, description="how many drives the team can replace")
    min_risk: float = Field(default=0.0, ge=0.0, le=1.0, description="never plan below this risk")


class StatusUpdate(BaseModel):
    status: Literal["completed", "cancelled"]


def priority(p: float) -> str:
    return "urgent" if p >= 0.3 else "high" if p >= 0.1 else "planned"


def create_app(db_path: str | None = None) -> FastAPI:
    con = store.connect(db_path or os.environ.get("ASSETWATCH_DB", "data/assetwatch.sqlite"))
    app = FastAPI(title="assetwatch", description=__doc__)

    def now() -> str:
        return datetime.now(UTC).isoformat(timespec="seconds")

    @app.get("/fleet/summary")
    def summary():
        row = con.execute("""SELECT COUNT(*) AS drives, COALESCE(SUM(r.probability), 0) AS expected_failures_30d,
                                    MAX(r.scored_on) AS scored_on
                             FROM assets a LEFT JOIN risk r USING (serial)""").fetchone()
        by = con.execute("""
            SELECT manufacturer, COUNT(*) AS drives, ROUND(SUM(probability), 1) AS expected_failures_30d
            FROM assets JOIN risk USING (serial) GROUP BY manufacturer ORDER BY drives DESC""").fetchall()
        open_orders = con.execute("SELECT COUNT(*) FROM work_orders WHERE status = 'open'").fetchone()[0]
        return {
            "drives": row["drives"],
            "scored_on": row["scored_on"],
            "expected_failures_30d": round(row["expected_failures_30d"], 1),
            "open_work_orders": open_orders,
            "by_manufacturer": [dict(r) for r in by],
        }

    @app.get("/assets")
    def assets(
        min_risk: float = Query(0.0, ge=0, le=1),
        manufacturer: str | None = Query(None, max_length=20),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ):
        sql = """SELECT a.*, r.probability, r.reason, r.scored_on FROM assets a JOIN risk r USING (serial)
                 WHERE r.probability >= ?"""
        args: list = [min_risk]
        if manufacturer:
            sql += " AND a.manufacturer = ?"
            args.append(manufacturer)
        sql += " ORDER BY r.probability DESC LIMIT ? OFFSET ?"
        return [dict(r) for r in con.execute(sql, [*args, limit, offset]).fetchall()]

    @app.get("/assets/{serial}")
    def asset(serial: str = Path(pattern=SERIAL)):
        row = con.execute(
            """SELECT a.*, r.probability, r.reason, r.scored_on FROM assets a
                             LEFT JOIN risk r USING (serial) WHERE serial = ?""",
            (serial,),
        ).fetchone()
        if row is None:
            raise HTTPException(404, "no such asset")
        orders = con.execute("SELECT * FROM work_orders WHERE serial = ? ORDER BY id DESC", (serial,)).fetchall()
        return {**dict(row), "work_orders": [dict(o) for o in orders]}

    @app.post("/work-orders/plan", status_code=201)
    def plan(req: PlanRequest):
        """Open work orders for the riskiest drives that do not already have one, up to the budget."""
        with con:  # one transaction: either the whole plan is recorded or none of it
            candidates = con.execute(
                """
                SELECT r.serial, r.probability, r.reason FROM risk r
                WHERE r.probability >= ?
                  AND NOT EXISTS (SELECT 1 FROM work_orders w WHERE w.serial = r.serial AND w.status = 'open')
                ORDER BY r.probability DESC LIMIT ?""",
                (req.min_risk, req.budget),
            ).fetchall()
            created = []
            for c in candidates:
                cur = con.execute(
                    """INSERT INTO work_orders (serial, created_at, priority, reason, risk_at_creation)
                                     VALUES (?, ?, ?, ?, ?)""",
                    (c["serial"], now(), priority(c["probability"]), c["reason"], c["probability"]),
                )
                created.append(
                    {
                        "id": cur.lastrowid,
                        "serial": c["serial"],
                        "priority": priority(c["probability"]),
                        "risk": round(c["probability"], 4),
                        "reason": c["reason"],
                    }
                )
        return {"created": created, "expected_failures_prevented": round(sum(c["risk"] for c in created), 2)}

    @app.get("/work-orders")
    def work_orders(
        status: Literal["open", "completed", "cancelled"] | None = None, limit: int = Query(100, ge=1, le=1000)
    ):
        sql, args = "SELECT * FROM work_orders", []
        if status:
            sql += " WHERE status = ?"
            args.append(status)
        return [dict(r) for r in con.execute(sql + " ORDER BY id DESC LIMIT ?", [*args, limit]).fetchall()]

    @app.patch("/work-orders/{order_id}")
    def update(order_id: int, body: StatusUpdate):
        with con:
            row = con.execute("SELECT status FROM work_orders WHERE id = ?", (order_id,)).fetchone()
            if row is None:
                raise HTTPException(404, "no such work order")
            if row["status"] != "open":
                raise HTTPException(409, f"work order is already {row['status']}")
            con.execute("UPDATE work_orders SET status = ?, closed_at = ? WHERE id = ?", (body.status, now(), order_id))
        return dict(con.execute("SELECT * FROM work_orders WHERE id = ?", (order_id,)).fetchone())

    return app
