"""Generate actionable HR retention scorecards from model probabilities."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import MODEL_PATH, SCORECARD_PATH, TARGET
from src.features import add_features

RISK_ACTIONS = {
    "High": "Schedule a stay interview and workload review",
    "Medium": "Review workload, manager support, and compensation",
    "Low": "Continue regular engagement and development check-ins",
}


def _risk_tier(probability: float) -> str:
    if probability < .30:
        return "Low"
    if probability <= .60:
        return "Medium"
    return "High"


def _drivers(row: pd.Series) -> list[str]:
    signals = {
        "high overtime": float(row["overtime_hours"]) / 40,
        "frequent leave": float(row["leaves_taken_last_quarter"]) / 8,
        "unplanned leave pattern": float(row["unplanned_leave_ratio"]),
        "declining performance trend": max(0, -float(row["performance_trend_slope"])),
        "recent manager change": float(row["manager_change_flag"]),
        "short tenure": max(0, (3 - float(row["tenure_years"])) / 3),
    }
    return [name for name, _ in sorted(signals.items(), key=lambda item: item[1], reverse=True) if _ > 0][:3]


def build_scorecard(frame: pd.DataFrame, model_bundle: dict) -> pd.DataFrame:
    """Score employees and add transparent risk tiers, drivers, and actions."""
    engineered = add_features(frame)
    feature_columns = model_bundle.get("feature_columns")
    x = engineered.drop(columns=[column for column in [TARGET, "employee_id", "last_3_performance_ratings", "rating_trend"] if column in engineered])
    if feature_columns:
        x = x[feature_columns]
    probabilities = model_bundle["model"].predict_proba(x)[:, 1]
    scorecard = frame[["employee_id", "department", "job_role", "salary_band"]].copy()
    scorecard["tenure_bucket"] = pd.cut(
        frame["tenure_years"],
        bins=[-0.1, 1, 3, 5, 10, 100],
        labels=["<1 yr", "1-3 yrs", "3-5 yrs", "5-10 yrs", "10+ yrs"],
    ).astype(str).to_numpy()
    scorecard["attrition_probability"] = probabilities
    scorecard["risk_tier"] = scorecard["attrition_probability"].map(_risk_tier)
    scorecard["top_3_risk_drivers"] = engineered.apply(lambda row: ", ".join(_drivers(row)), axis=1)
    scorecard["suggested_retention_action"] = scorecard["risk_tier"].map(RISK_ACTIONS)
    return scorecard.sort_values("attrition_probability", ascending=False).reset_index(drop=True)


def generate_scorecard(frame: pd.DataFrame, model_path: Path = MODEL_PATH, output_path: Path = SCORECARD_PATH) -> pd.DataFrame:
    """Load a persisted model, generate the scorecard, and write its CSV."""
    bundle = __import__("joblib").load(model_path)
    scorecard = build_scorecard(frame, bundle)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scorecard.to_csv(output_path, index=False)
    return scorecard
