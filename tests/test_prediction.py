from pathlib import Path

from src.config import MODEL_PATH
from src.data_loader import generate_synthetic_data
from src.features import add_features
from src.scorecard import build_scorecard
from src.preprocessing import build_pipeline
from sklearn.linear_model import LogisticRegression


def test_scorecard_shape_and_probability_range():
    frame = generate_synthetic_data(30)
    engineered = add_features(frame)
    x = engineered.drop(columns=["resigned_within_quarter", "employee_id", "last_3_performance_ratings", "rating_trend"])
    pipeline = build_pipeline(LogisticRegression(max_iter=500))
    pipeline.fit(x, frame["resigned_within_quarter"])
    scorecard = build_scorecard(frame, {"model": pipeline, "feature_columns": list(x.columns)})
    assert len(scorecard) == 30
    assert scorecard["attrition_probability"].between(0, 1).all()
    assert set(scorecard["risk_tier"]).issubset({"Low", "Medium", "High"})
