"""Modern HR Retention Scorecard dashboard."""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import MODEL_PATH, REPORTS_DIR, TARGET
from src.data_loader import load_data, validate_data
from src.features import add_features
from src.scorecard import build_scorecard
from app.components import add_derived_columns, hero_banner, load_css, metric_card, profile_card, section_title, warn_missing_columns

st.set_page_config(layout="wide", page_title="HR Retention Scorecard", page_icon="📊")
load_css()

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}
PALETTE = {"Low": "#12B76A", "Medium": "#F79009", "High": "#F04438"}
INDIGO = "#4F46E5"


def chart_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    """Apply the dashboard chart language and stable sizing."""
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=12, r=12, t=42, b=12),
        font=dict(family="Arial, sans-serif", color="#344054"),
        paper_bgcolor="white", plot_bgcolor="white", legend=dict(orientation="h", y=1.08),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#EAECF0", zeroline=False)
    return fig


@st.cache_resource(show_spinner=False)
def get_model() -> dict:
    """Load the unchanged persisted model bundle once per session."""
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_uploaded_csv(content: bytes) -> pd.DataFrame:
    """Validate uploaded CSV bytes without changing the model pipeline."""
    return validate_data(pd.read_csv(pd.io.common.BytesIO(content)))


@st.cache_data(show_spinner=False)
def load_reports() -> tuple[dict, pd.DataFrame]:
    """Load saved metrics and global feature importance reports."""
    metrics = json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    shap_path = REPORTS_DIR / "shap_importance.csv"
    importance_path = shap_path if shap_path.exists() else REPORTS_DIR / "permutation_importance.csv"
    importance = pd.read_csv(importance_path)
    value_column = "mean_abs_shap" if "mean_abs_shap" in importance.columns else "importance"
    return metrics, importance.assign(value=importance[value_column]).sort_values("value", ascending=False)


def apply_filters(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply sidebar selections to the scorecard."""
    filtered = frame.copy()
    if st.session_state.get("department_filter"):
        filtered = filtered[filtered["department"].isin(st.session_state.department_filter)]
    if st.session_state.get("role_filter"):
        filtered = filtered[filtered["job_role"].isin(st.session_state.role_filter)]
    if st.session_state.get("tier_filter"):
        filtered = filtered[filtered["risk_tier"].isin(st.session_state.tier_filter)]
    low, high = st.session_state.get("probability_filter", (0.0, 1.0))
    filtered = filtered[filtered["attrition_probability"].between(low, high)]
    search = st.session_state.get("employee_search", "").strip().lower()
    if search:
        filtered = filtered[filtered["employee_id"].str.lower().str.contains(search, na=False)]
    return filtered


def kpi_row(filtered: pd.DataFrame, source: pd.DataFrame) -> None:
    """Render the four overview KPI cards."""
    columns = st.columns(4)
    values = [
        ("Total Employees", f"{len(filtered):,}", "Employees in current filter", "👥", "indigo"),
        ("High-Risk Employees", f"{(filtered['risk_tier'] == 'High').sum():,}", "Probability above 60%", "🔴", "red"),
        ("Medium-Risk Employees", f"{(filtered['risk_tier'] == 'Medium').sum():,}", "Probability from 30% to 60%", "🟡", "amber"),
        ("Attrition Rate", f"{source[TARGET].mean():.1%}", "Observed resignations in source", "📈", "green"),
    ]
    for column, item in zip(columns, values):
        with column:
            metric_card(*item)


def overview_tab(filtered: pd.DataFrame, source: pd.DataFrame) -> None:
    """Render executive overview charts."""
    if not warn_missing_columns(filtered, ["risk_tier", "attrition_probability", "department"]):
        return
    kpi_row(filtered, source)
    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1, 1.35])
    with left:
        section_title("Risk distribution", "The current filtered population grouped by decision-support tier.")
        counts = filtered["risk_tier"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
        fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=.68, marker_colors=[PALETTE[tier] for tier in counts.index], textinfo="label+percent", hovertemplate="%{label}: %{value:,}<extra></extra>"))
        fig.update_layout(showlegend=False)
        st.plotly_chart(chart_layout(fig, 320), use_container_width=True, config=PLOTLY_CONFIG)
    with right:
        section_title("Average attrition probability by department", "Sorted highest first; axis starts at zero for honest comparison.")
        department_risk = filtered.groupby("department", as_index=False, observed=True)["attrition_probability"].mean().sort_values("attrition_probability")
        fig = px.bar(department_risk, x="attrition_probability", y="department", orientation="h", text="attrition_probability", color="attrition_probability", color_continuous_scale=[[0, "#A7F3D0"], [.6, "#FCD34D"], [1, "#FCA5A5"]])
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside", hovertemplate="%{y}: %{x:.1%}<extra></extra>", marker_line_width=0)
        maximum = float(department_risk["attrition_probability"].max()) if not department_risk.empty else .6
        fig.update_xaxes(range=[0, max(.7, maximum * 1.15)], tickformat=".0%", title="Average probability")
        fig.update_layout(coloraxis_showscale=False, yaxis_title=None)
        st.plotly_chart(chart_layout(fig, 360), use_container_width=True, config=PLOTLY_CONFIG)


def risk_analysis_tab(filtered: pd.DataFrame, importance: pd.DataFrame) -> None:
    """Render population-level risk diagnostics."""
    if not warn_missing_columns(filtered, ["department", "risk_tier", "attrition_probability", "tenure_bucket", "salary_band"]):
        return
    section_title("Where risk is concentrated", "Use the cross-sections below to prioritize retention capacity.")
    left, right = st.columns(2)
    with left:
        heatmap = pd.crosstab(filtered["department"], filtered["risk_tier"]).reindex(columns=["Low", "Medium", "High"], fill_value=0)
        fig = px.imshow(heatmap, text_auto=True, color_continuous_scale=[[0, "#F8FAFC"], [1, INDIGO]], aspect="auto", labels={"x": "Risk tier", "y": "Department", "color": "Employees"})
        st.plotly_chart(chart_layout(fig, 380), use_container_width=True, config=PLOTLY_CONFIG)
    with right:
        score_hist = px.histogram(filtered, x="attrition_probability", nbins=24, color="risk_tier", color_discrete_map=PALETTE, barmode="stack", labels={"attrition_probability": "Attrition probability"})
        score_hist.update_xaxes(range=[0, 1], tickformat=".0%", dtick=.1)
        score_hist.add_vline(x=.3, line_dash="dash", line_color=PALETTE["Medium"], annotation_text="30%", annotation_position="top")
        score_hist.add_vline(x=.6, line_dash="dash", line_color=PALETTE["High"], annotation_text="60%", annotation_position="top")
        st.plotly_chart(chart_layout(score_hist, 380), use_container_width=True, config=PLOTLY_CONFIG)
    left, right = st.columns(2)
    with left:
        tenure = filtered.groupby("tenure_bucket", as_index=False, observed=True)["attrition_probability"].mean()
        tenure["tenure_bucket"] = pd.Categorical(tenure["tenure_bucket"], ["<1 yr", "1-3 yrs", "3-5 yrs", "5-10 yrs", "10+ yrs"])
        fig = px.bar(tenure.sort_values("tenure_bucket"), x="tenure_bucket", y="attrition_probability", color="attrition_probability", color_continuous_scale="Blues", text="attrition_probability")
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        maximum = float(tenure["attrition_probability"].max()) if not tenure.empty else .6
        fig.update_yaxes(range=[0, max(.7, maximum * 1.2)], tickformat=".0%", title="Average probability")
        fig.update_layout(coloraxis_showscale=False, xaxis_title=None, title="By tenure bucket")
        st.plotly_chart(chart_layout(fig, 330), use_container_width=True, config=PLOTLY_CONFIG)
    with right:
        salary = filtered.groupby("salary_band", as_index=False, observed=True)["attrition_probability"].mean()
        fig = px.bar(salary, x="salary_band", y="attrition_probability", color="attrition_probability", color_continuous_scale="Oranges", text="attrition_probability")
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        maximum = float(salary["attrition_probability"].max()) if not salary.empty else .6
        fig.update_yaxes(range=[0, max(.7, maximum * 1.2)], tickformat=".0%", title="Average probability")
        fig.update_layout(coloraxis_showscale=False, xaxis_title=None, title="By salary band")
        st.plotly_chart(chart_layout(fig, 330), use_container_width=True, config=PLOTLY_CONFIG)
    section_title("Global attrition drivers", "SHAP summary is preferred; permutation importance is used as a fallback.")
    top = importance.head(10).sort_values("value")
    fig = px.bar(top, x="value", y="feature", orientation="h", text="value", color="value", color_continuous_scale="Purples")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", hovertemplate="%{y}: %{x:.3f}<extra></extra>")
    fig.update_layout(coloraxis_showscale=False, xaxis_title="Importance", yaxis_title=None)
    st.plotly_chart(chart_layout(fig, 420), use_container_width=True, config=PLOTLY_CONFIG)


def scorecard_tab(filtered: pd.DataFrame) -> None:
    """Render searchable and downloadable employee table."""
    if not warn_missing_columns(filtered, ["employee_id", "department", "job_role", "attrition_probability", "risk_tier", "top_3_risk_drivers", "suggested_retention_action"]):
        return
    section_title("Employee scorecard", "Sort any column, search by employee ID, and download the filtered view.")
    st.text_input("Search employee ID", key="employee_search", placeholder="e.g. E00142")
    table = apply_filters(filtered)
    display = table[["employee_id", "department", "job_role", "attrition_probability", "risk_tier", "top_3_risk_drivers", "suggested_retention_action"]].copy()
    display["risk_tier"] = display["risk_tier"].map({"Low": "🟢 Low", "Medium": "🟡 Medium", "High": "🔴 High"})
    st.dataframe(display, hide_index=True, use_container_width=True, height=530, column_config={
        "employee_id": st.column_config.TextColumn("Employee ID", width="small"),
        "attrition_probability": st.column_config.ProgressColumn("Attrition Probability", format="%.1f%%", min_value=0, max_value=1),
        "risk_tier": st.column_config.TextColumn("Risk Tier", width="small"),
        "top_3_risk_drivers": st.column_config.TextColumn("Top 3 Risk Drivers", width="large"),
        "suggested_retention_action": st.column_config.TextColumn("Suggested Retention Action", width="large"),
    })
    st.download_button("Download filtered CSV", table.to_csv(index=False), "retention_scorecard_filtered.csv", "text/csv")


def drilldown_tab(filtered: pd.DataFrame, source: pd.DataFrame) -> None:
    """Render an individual employee risk view."""
    if not warn_missing_columns(filtered, ["employee_id", "risk_tier", "attrition_probability", "suggested_retention_action", "top_3_risk_drivers"]):
        return
    if not warn_missing_columns(source, ["employee_id", "department", "job_role", "tenure_years", "salary_band", "overtime_hours", "leaves_taken_last_quarter", "unplanned_leave_ratio"]):
        return
    section_title("Employee drill-down", "A compact context view for a human-reviewed retention conversation.")
    if filtered.empty:
        st.info("No employees match the current filters.")
        return
    selected_id = st.selectbox("Select employee", filtered["employee_id"].tolist(), key="drilldown_employee")
    score = filtered[filtered["employee_id"] == selected_id].iloc[0]
    raw = source[source["employee_id"] == selected_id].iloc[0]
    left, right = st.columns([.8, 1.2])
    with left:
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=float(score["attrition_probability"] * 100), number={"suffix": "%"}, title={"text": "Attrition probability"}, gauge={"axis": {"range": [0, 100], "ticksuffix": "%"}, "bar": {"color": PALETTE[score["risk_tier"]]}, "steps": [{"range": [0, 30], "color": "#ECFDF3"}, {"range": [30, 60], "color": "#FFFAEB"}, {"range": [60, 100], "color": "#FEF3F2"}], "threshold": {"line": {"color": INDIGO, "width": 4}, "value": 60}}))
        st.plotly_chart(chart_layout(gauge, 300), use_container_width=True, config=PLOTLY_CONFIG)
    with right:
        profile_card({"Risk tier": score["risk_tier"], "Department": raw["department"], "Role": raw["job_role"], "Tenure": f"{raw['tenure_years']:.1f} years", "Salary band": raw["salary_band"], "Overtime": f"{raw['overtime_hours']:.1f} hours", "Leaves": int(raw["leaves_taken_last_quarter"])})
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div class='callout'><strong>Recommended action</strong><br>{score['suggested_retention_action']}</div>", unsafe_allow_html=True)
    radar_fields = {"Tenure": "tenure_years", "Overtime": "overtime_hours", "Leaves": "leaves_taken_last_quarter", "Unplanned leave": "unplanned_leave_ratio", "Probability": "attrition_probability"}
    employee_values = [float(raw[field]) if field in raw else float(score[field]) for field in radar_fields.values()]
    department_rows = source[source["department"] == raw["department"]]
    department_engineered = add_features(department_rows)
    department_score = filtered[filtered["department"] == raw["department"]]
    department_values = [float(department_engineered[field].mean()) if field in department_engineered else float(department_score["attrition_probability"].mean()) for field in radar_fields.values()]
    maxima = np.maximum(np.maximum(employee_values, department_values), 1e-6)
    employee_scaled = np.array(employee_values) / maxima
    department_scaled = np.array(department_values) / maxima
    categories = list(radar_fields)
    categories += [categories[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=(*employee_scaled, employee_scaled[0]), theta=categories, fill="toself", name="Employee", line_color=INDIGO))
    fig.add_trace(go.Scatterpolar(r=(*department_scaled, department_scaled[0]), theta=categories, fill="toself", name="Department average", line_color="#98A2B3", opacity=.6))
    fig.update_layout(polar={"radialaxis": {"visible": True, "range": [0, 1]}}, title="Employee compared with department average")
    st.plotly_chart(chart_layout(fig, 380), use_container_width=True, config=PLOTLY_CONFIG)
    st.markdown(f"**Top drivers:** {score['top_3_risk_drivers']}")


def model_performance_tab(scorecard: pd.DataFrame, source: pd.DataFrame, bundle: dict, metrics: dict) -> None:
    """Render saved model comparison and SVM diagnostics."""
    if not warn_missing_columns(scorecard, ["employee_id", "attrition_probability"]):
        return
    if not warn_missing_columns(source, ["employee_id", TARGET]):
        return
    section_title("Model performance", "Metrics are from the saved training run; curves below use the loaded synthetic source for an interactive view.")
    rows = [{"Model": name.replace("_", " ").title(), "ROC-AUC": values["roc_auc"], "PR-AUC": values["pr_auc"], "Recall": values["recall"], "Precision": values["precision"], "F1": values["f1"]} for name, values in metrics["scores"].items()]
    comparison = pd.DataFrame(rows)
    st.dataframe(comparison.style.highlight_max(subset=["ROC-AUC", "PR-AUC", "Recall", "Precision", "F1"], color="#E0E7FF").format({column: "{:.3f}" for column in comparison.columns[1:]}), hide_index=True, use_container_width=True)
    scored = source[["employee_id", TARGET]].merge(scorecard[["employee_id", "attrition_probability"]], on="employee_id", how="inner")
    fpr, tpr, _ = roc_curve(scored[TARGET], scored["attrition_probability"])
    precision, recall, _ = precision_recall_curve(scored[TARGET], scored["attrition_probability"])
    threshold = float(bundle.get("threshold", .5))
    matrix = confusion_matrix(scored[TARGET], (scored["attrition_probability"] >= threshold).astype(int))
    left, right = st.columns(2)
    with left:
        fig = go.Figure(go.Heatmap(z=matrix, x=["Stay", "Resign"], y=["Stay", "Resign"], text=matrix, texttemplate="%{text}", colorscale="Blues", showscale=False))
        fig.update_layout(title=f"SVM confusion matrix at {threshold:.0%} threshold", xaxis_title="Predicted", yaxis_title="Actual")
        st.plotly_chart(chart_layout(fig, 350), use_container_width=True, config=PLOTLY_CONFIG)
    with right:
        fig = go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines", line={"color": INDIGO, "width": 3}, name="SVM ROC"))
        fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line={"dash": "dash", "color": "#98A2B3"})
        fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate", xaxis_range=[0, 1], yaxis_range=[0, 1])
        st.plotly_chart(chart_layout(fig, 350), use_container_width=True, config=PLOTLY_CONFIG)
    fig = go.Figure(go.Scatter(x=recall, y=precision, mode="lines", line={"color": "#F79009", "width": 3}, name="SVM PR"))
    fig.update_layout(title="Precision-recall curve", xaxis_title="Recall", yaxis_title="Precision", xaxis_range=[0, 1], yaxis_range=[0, 1])
    st.plotly_chart(chart_layout(fig, 330), use_container_width=True, config=PLOTLY_CONFIG)
    st.info("Data note: this dataset is synthetic. Results demonstrate the pipeline, not real-world accuracy.")


hero_banner(date.today().strftime("%B %d, %Y"))
try:
    uploaded = st.sidebar.file_uploader("Upload employee CSV", type="csv", help="Use the same schema as the training data.")
    source = load_uploaded_csv(uploaded.getvalue()) if uploaded else load_data()
    source = add_derived_columns(source)
    bundle = get_model()
    scorecard = build_scorecard(source, bundle)
    profile_columns = ["tenure_years", "overtime_hours", "leaves_taken_last_quarter", "unplanned_leave_ratio", "days_since_last_leave", "manager_change_flag"]
    available_profile_columns = [column for column in profile_columns if column in source.columns]
    scorecard = scorecard.merge(source[["employee_id", *available_profile_columns]], on="employee_id", how="left")
    scorecard = add_derived_columns(scorecard)
    metrics, importance = load_reports()
except FileNotFoundError:
    st.error("Train the model first with: python -m src.train")
    st.stop()
except Exception as error:
    st.error(f"Could not load this file: {error}")
    st.stop()

with st.sidebar:
    st.markdown("### Filter workspace")
    options = {
        "department_filter": sorted(scorecard["department"].dropna().unique().tolist()),
        "role_filter": sorted(scorecard["job_role"].dropna().unique().tolist()),
        "tier_filter": ["High", "Medium", "Low"],
    }
    for key, values in options.items():
        if key not in st.session_state:
            st.session_state[key] = values
    if "probability_filter" not in st.session_state:
        st.session_state["probability_filter"] = (0.0, 1.0)
    if st.button("Reset filters", use_container_width=True):
        for key, values in options.items():
            st.session_state[key] = values
        st.session_state["probability_filter"] = (0.0, 1.0)
        st.session_state["employee_search"] = ""
        st.rerun()
    st.multiselect("Department", options["department_filter"], key="department_filter")
    st.multiselect("Job role", options["role_filter"], key="role_filter")
    st.multiselect("Risk tier", options["tier_filter"], key="tier_filter")
    st.slider("Attrition probability", 0.0, 1.0, key="probability_filter", format="%.0f%%")
    st.caption(f"{len(source):,} records loaded · {source['department'].nunique()} departments")

filtered = apply_filters(scorecard)
tabs = st.tabs(["Overview", "Risk Analysis", "Employee Scorecard", "Employee Drill-down", "Model Performance"])
with tabs[0]:
    overview_tab(filtered, source)
with tabs[1]:
    risk_analysis_tab(filtered, importance)
with tabs[2]:
    scorecard_tab(filtered)
with tabs[3]:
    drilldown_tab(filtered, source)
with tabs[4]:
    model_performance_tab(scorecard, source, bundle, metrics)
