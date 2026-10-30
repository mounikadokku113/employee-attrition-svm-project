"""Project configuration and reproducibility settings."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RAW_DATA_PATH = DATA_RAW_DIR / "employee_attrition.csv"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT_DIR / "models"
SCORECARD_PATH = REPORTS_DIR / "retention_scorecard.csv"
MODEL_PATH = MODELS_DIR / "attrition_model.joblib"
RANDOM_STATE = 42
TARGET = "resigned_within_quarter"

REQUIRED_COLUMNS = [
    "employee_id", "age", "department", "job_role", "tenure_years",
    "salary_band", "last_3_performance_ratings", "rating_trend",
    "promotions_last_2y", "overtime_hours", "leaves_taken_last_quarter",
    "unplanned_leave_ratio", "days_since_last_leave", "manager_change_flag",
    TARGET,
]

NUMERIC_FEATURES = [
    "age", "tenure_years", "promotions_last_2y", "overtime_hours",
    "leaves_taken_last_quarter", "unplanned_leave_ratio",
    "days_since_last_leave", "manager_change_flag", "leave_frequency",
    "performance_trend_slope", "overtime_to_leave_ratio",
]
CATEGORICAL_FEATURES = ["department", "job_role", "salary_band", "tenure_bucket"]


def ensure_directories() -> None:
    """Create project output directories when they do not exist."""
    for directory in (DATA_RAW_DIR, DATA_PROCESSED_DIR, FIGURES_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
