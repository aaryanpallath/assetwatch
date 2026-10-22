"""Failure risk, and the question a maintenance team actually asks: with a fixed budget of proactive replacements,
which drives should we replace?

Two ways to rank the fleet are compared:
  SMART rule   the standard practice: a drive is suspect if any of SMART 5, 187, 188, 197 or 198 is above zero;
               among suspects, more errors rank higher
  model        gradient-boosted trees on SMART levels, week-over-week growth of the error counters, age, capacity
               and manufacturer, predicting failure within 30 days

Both are scored at the same budgets (for example, replace the riskiest 0.5% of the fleet each week): of the drives
that really failed within 30 days, how many did the budget catch, and what share of replacements were needed?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder

from .data import ERROR_COUNTERS, GROWTH, SMART

NUMERIC = [f"smart_{s}" for s in SMART] + [f"d7_smart_{s}" for s in GROWTH] + ["age_years", "capacity_tb"]
CATEGORICAL = ["manufacturer"]
FEATURES = NUMERIC + CATEGORICAL
BUDGETS = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)


def calibrator(scores: np.ndarray, y: np.ndarray):
    """Map model scores to failure probabilities with isotonic regression fit on held-out weeks. Training with
    balanced class weights helps the ranking but inflates every score, and with early-stopped trees the inflation is
    not a clean odds factor (simply dividing it out under-predicted failures threefold), so it is learned from data."""
    from sklearn.isotonic import IsotonicRegression

    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(scores, y)


def calibration(df: pd.DataFrame, p: np.ndarray) -> dict:
    """Expected vs observed failures overall and among the riskiest drives."""
    y = df.fails_30d.to_numpy()
    order = np.argsort(-p)
    edges = [0, 100, 1000, 10000, len(p)]
    bands = [
        {
            "rank": f"{lo + 1}-{hi}",
            "expected": round(float(p[order[lo:hi]].sum()), 1),
            "observed": int(y[order[lo:hi]].sum()),
        }
        for lo, hi in zip(edges[:-1], edges[1:], strict=True)
    ]
    return {"expected_failures": round(float(p.sum()), 1), "observed_failures": int(y.sum()), "by_risk_rank": bands}


def rule_score(df: pd.DataFrame) -> np.ndarray:
    """The SMART rule as a ranking: suspects first, by total error count; everything else scores 0."""
    errors = sum(df[f"smart_{s}"].fillna(0).clip(lower=0) for s in ERROR_COUNTERS)
    return np.log1p(errors.to_numpy())


def classifier():
    pre = ColumnTransformer(
        [
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ]
    )
    return make_pipeline(
        pre,
        HistGradientBoostingClassifier(
            categorical_features=[0],
            learning_rate=0.05,
            max_iter=600,
            max_leaf_nodes=31,
            min_samples_leaf=100,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=40,
            class_weight="balanced",
            random_state=0,
        ),
    )


def at_budget(df: pd.DataFrame, score: np.ndarray, budget: float) -> dict[str, float]:
    """Replace the top `budget` share of drives on each snapshot date; pool the results over dates."""
    caught = flagged = failing = 0
    for _, idx in df.groupby("date").indices.items():
        y = df.fails_30d.to_numpy()[idx]
        k = max(1, int(round(budget * len(idx))))
        top = idx[np.argsort(-score[idx], kind="stable")[:k]]
        caught += int(df.fails_30d.to_numpy()[top].sum())
        flagged += k
        failing += int(y.sum())
    return {
        "budget_pct": round(100 * budget, 2),
        "replaced": flagged,
        "failures": failing,
        "caught": caught,
        "recall": round(caught / max(failing, 1), 3),
        "precision": round(caught / max(flagged, 1), 3),
    }


def rule_operating_point(df: pd.DataFrame) -> dict[str, float]:
    """The rule as a yes/no alert: how many drives it flags, and how many failures it catches."""
    alert = rule_score(df) > 0
    y = df.fails_30d.to_numpy().astype(bool)
    return {
        "flagged_pct": round(100 * float(alert.mean()), 2),
        "recall": round(float((alert & y).sum() / y.sum()), 3),
        "precision": round(float((alert & y).sum() / max(alert.sum(), 1)), 3),
    }


def evaluate(df: pd.DataFrame, scores: dict[str, np.ndarray]) -> dict:
    y = df.fails_30d.to_numpy()
    return {
        name: {
            "pr_auc": round(float(average_precision_score(y, s)), 4),
            "budgets": [at_budget(df, s, b) for b in BUDGETS],
        }
        for name, s in scores.items()
    }


def explain(row: pd.Series) -> str:
    """A plain-language reason for a work order, from the counters that drive risk."""
    parts = []
    if pd.isna(row.get("smart_9")):
        return "stopped reporting SMART health data, which preceded failure in about half of such cases"
    names = {
        5: "reallocated sectors",
        187: "uncorrectable errors",
        188: "command timeouts",
        197: "pending sectors",
        198: "offline uncorrectable sectors",
        199: "interface CRC errors",
    }
    for s, label in names.items():
        v = row.get(f"smart_{s}")
        g = row.get(f"d7_smart_{s}") if s in GROWTH else None
        if s == 188 and v and v > 1e6:  # Seagate packs several counters into one raw 188 value
            parts.append(label)
            continue
        if v and v > 0:
            parts.append(f"{int(v):,} {label}" + (f" (+{int(g):,} this week)" if g else ""))
    age = row.get("age_years")
    if age and age >= 5:
        parts.append(f"{age:.1f} years in service")
    return "; ".join(parts) or "elevated risk from age, model and usage pattern"
