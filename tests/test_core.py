import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.data_loader import generate_synthetic_data
from src.features import add_features
from src.preprocessing import build_pipeline


def test_feature_engineering_adds_expected_columns():
    frame = add_features(generate_synthetic_data(20))
    assert {"leave_frequency", "performance_trend_slope", "tenure_bucket", "overtime_to_leave_ratio"}.issubset(frame.columns)
    assert frame["performance_trend_slope"].notna().all()


def test_preprocessing_and_prediction_shape():
    frame = add_features(generate_synthetic_data(40))
    features = frame.drop(columns=["resigned_within_quarter", "employee_id", "last_3_performance_ratings", "rating_trend"])
    model = build_pipeline(LogisticRegression(max_iter=500), use_smote=True)
    model.fit(features, frame["resigned_within_quarter"])
    assert model.predict_proba(features).shape == (40, 2)
