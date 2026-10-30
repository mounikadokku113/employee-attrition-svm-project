"""Reusable presentation components for the HR dashboard."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]


def add_derived_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add dashboard-only tenure and risk columns without mutating the input."""
    result = frame.copy()
    if "tenure_bucket" not in result.columns and "tenure_years" in result.columns:
        result["tenure_bucket"] = pd.cut(
            result["tenure_years"],
            bins=[-0.1, 1, 3, 5, 10, 100],
            labels=["<1 yr", "1-3 yrs", "3-5 yrs", "5-10 yrs", "10+ yrs"],
        )
    if "risk_tier" not in result.columns and "attrition_probability" in result.columns:
        result["risk_tier"] = pd.cut(
            result["attrition_probability"],
            bins=[-float("inf"), 0.30, 0.60, float("inf")],
            labels=["Low", "Medium", "High"],
            right=True,
            include_lowest=True,
        ).astype(str)
    return result


def warn_missing_columns(frame: pd.DataFrame, columns: list[str]) -> bool:
    """Warn once for missing dashboard fields and return whether they exist."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        for column in missing:
            st.warning(f"Column {column} not found in data")
        return False
    return True


def load_css() -> None:
    """Inject the shared dashboard stylesheet."""
    st.markdown((ROOT / "app" / "styles.css").read_text(encoding="utf-8"), unsafe_allow_html=True)


def hero_banner(updated: str) -> None:
    """Render the dashboard masthead."""
    st.markdown(
        f"""<div class='hero'><h1>HR Retention Scorecard</h1>
        <p>SVM-based employee attrition prediction for focused, human-reviewed retention conversations.</p>
        <div class='hero-meta'><span>People analytics workspace</span><span>Last updated {updated}</span></div></div>""",
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, help_text: str, icon: str, accent: str) -> None:
    """Render a styled KPI card."""
    st.markdown(
        f"""<div class='card metric-card accent-{accent}'><span class='metric-icon'>{icon}</span>
        <div class='metric-label'>{label}</div><div class='metric-value'>{value}</div>
        <div class='metric-help'>{help_text}</div></div>""",
        unsafe_allow_html=True,
    )


def section_title(title: str, note: str | None = None) -> None:
    """Render a consistent section title and optional helper text."""
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    if note:
        st.markdown(f"<div class='section-note'>{note}</div>", unsafe_allow_html=True)


def risk_badge(value: str) -> str:
    """Return an HTML risk-tier badge."""
    labels = {"Low": "🟢 Low", "Medium": "🟡 Medium", "High": "🔴 High"}
    css = value.lower()
    return f"<span class='badge badge-{css}'>{labels.get(value, value)}</span>"


def profile_card(values: dict[str, object]) -> None:
    """Render a compact employee profile grid."""
    cells = "".join(
        f"<div class='profile-item'><small>{label}</small><strong>{value}</strong></div>"
        for label, value in values.items()
    )
    st.markdown(f"<div class='profile-grid'>{cells}</div>", unsafe_allow_html=True)


def styled_scorecard(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare display columns for the employee scorecard table."""
    display = frame.copy()
    display["attrition_probability"] = display["attrition_probability"].map(lambda value: f"{value:.1%}")
    display["risk_tier"] = display["risk_tier"].map(risk_badge)
    return display.rename(columns={
        "employee_id": "Employee ID", "department": "Department", "job_role": "Job Role",
        "attrition_probability": "Attrition Probability", "risk_tier": "Risk Tier",
        "top_3_risk_drivers": "Top 3 Risk Drivers", "suggested_retention_action": "Suggested Retention Action",
    })
