"""Feature engineering for employee attrition."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add leave, performance, tenure, and workload features without mutating input."""
    result = frame.copy()
    ratings = result["last_3_performance_ratings"].map(
        lambda value: [float(item) for item in str(value).replace(";", ",").split(",")]
    )
    result["performance_trend_slope"] = ratings.map(
        lambda values: float(np.polyfit(np.arange(len(values)), values, 1)[0])
    )
    result["leave_frequency"] = result["leaves_taken_last_quarter"] / 3.0
    result["overtime_to_leave_ratio"] = result["overtime_hours"] / (result["leaves_taken_last_quarter"] + 1)
    result["tenure_bucket"] = pd.cut(
        result["tenure_years"], bins=[-np.inf, 1, 3, 7, np.inf],
        labels=["New", "Early", "Established", "Veteran"],
    ).astype(str)
    return result
