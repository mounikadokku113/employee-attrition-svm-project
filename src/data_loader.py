"""Load, validate, and generate employee attrition data."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RAW_DATA_PATH, REQUIRED_COLUMNS, TARGET, ensure_directories

LOGGER = logging.getLogger(__name__)


def _parse_ratings(value: object) -> list[float]:
    """Parse ratings represented as a list, tuple, or comma-separated string."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(item) for item in value]
    return [float(item.strip()) for item in str(value).replace(";", ",").split(",")]


def generate_synthetic_data(n_employees: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Generate realistic, intentionally imbalanced HR records."""
    rng = np.random.default_rng(random_state)
    departments = np.array(["Engineering", "Sales", "Operations", "HR", "Finance", "Customer Success"])
    roles = np.array(["Analyst", "Manager", "Specialist", "Engineer", "Associate", "Director"])
    salary_bands = np.array(["A", "B", "C", "D"])
    department = rng.choice(departments, n_employees, p=[.27, .20, .20, .08, .13, .12])
    role = rng.choice(roles, n_employees, p=[.22, .16, .20, .22, .14, .06])
    tenure = np.round(rng.gamma(2.2, 2.0, n_employees).clip(.1, 18), 1)
    overtime = np.round(rng.gamma(2.0, 8, n_employees), 1)
    leaves = rng.poisson(2.5, n_employees).clip(0, 12)
    unplanned = np.round((rng.beta(2, 8, n_employees) + (leaves > 5) * .12).clip(0, 1), 3)
    ratings = np.clip(np.round(rng.normal(3.35, .55, (n_employees, 3)), 1), 1, 5)
    ratings[:, 2] = np.clip(ratings[:, 1] + rng.normal(0, .35, n_employees), 1, 5)
    rating_trend = np.round(ratings[:, 2] - ratings[:, 0], 2)
    promotions = rng.binomial(2, np.minimum(.15 + tenure / 40, .7), n_employees)
    days_since_leave = rng.integers(0, 181, n_employees)
    manager_change = rng.binomial(1, .18, n_employees)
    salary = rng.choice(salary_bands, n_employees, p=[.25, .35, .28, .12])
    age = rng.integers(21, 61, n_employees)
    logit = (-3.0 + .045 * overtime + .20 * leaves + 1.2 * unplanned
             - .24 * tenure - .32 * promotions - .45 * rating_trend
             + .55 * manager_change + (salary == "A") * .35)
    probability = 1 / (1 + np.exp(-logit))
    target = rng.binomial(1, probability)
    # Calibrate the generated prevalence to approximately 15% without losing signal.
    positive_cutoff = np.quantile(probability, .85)
    target = (probability >= positive_cutoff).astype(int)
    frame = pd.DataFrame({
        "employee_id": [f"E{index:05d}" for index in range(1, n_employees + 1)],
        "age": age, "department": department, "job_role": role, "tenure_years": tenure,
        "salary_band": salary, "last_3_performance_ratings": [",".join(map(str, row)) for row in ratings],
        "rating_trend": rating_trend, "promotions_last_2y": promotions,
        "overtime_hours": overtime, "leaves_taken_last_quarter": leaves,
        "unplanned_leave_ratio": unplanned, "days_since_last_leave": days_since_leave,
        "manager_change_flag": manager_change, TARGET: target,
    })
    return frame


def validate_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate required schema, target values, nulls, and duplicate IDs."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Input data is empty")
    if not frame[TARGET].dropna().isin([0, 1]).all():
        raise ValueError(f"{TARGET} must contain only 0 and 1")
    duplicate_ids = int(frame["employee_id"].duplicated().sum())
    if duplicate_ids:
        LOGGER.warning("Found %d duplicate employee IDs", duplicate_ids)
    nulls = int(frame.isna().sum().sum())
    if nulls:
        LOGGER.warning("Found %d missing values; preprocessing will impute them", nulls)
    return frame.copy()


def load_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load a supplied CSV or generate and persist a reproducible synthetic dataset."""
    ensure_directories()
    path = Path(csv_path) if csv_path else None
    if path and path.exists():
        LOGGER.info("Loading employee data from %s", path)
        return validate_data(pd.read_csv(path))
    generated_path = RAW_DATA_PATH
    if not generated_path.exists():
        LOGGER.info("No CSV found; generating synthetic employee data")
        generate_synthetic_data().to_csv(generated_path, index=False)
    return validate_data(pd.read_csv(generated_path))
