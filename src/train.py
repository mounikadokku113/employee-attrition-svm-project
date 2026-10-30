"""Train, compare, evaluate, and persist the attrition models."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.svm import SVC

from src.config import FIGURES_DIR, MODEL_PATH, RANDOM_STATE, TARGET, ensure_directories
from src.data_loader import load_data
from src.evaluate import evaluate_model, find_recall_threshold, permutation_importance_frame, shap_sample
from src.features import add_features
from src.preprocessing import build_pipeline
from src.scorecard import generate_scorecard

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)
EXCLUDED_COLUMNS = [TARGET, "employee_id", "last_3_performance_ratings", "rating_trend"]


def train(csv_path: str | Path | None = None) -> dict:
    """Run the reproducible model selection and artifact generation workflow."""
    ensure_directories()
    frame = add_features(load_data(csv_path))
    x = frame.drop(columns=EXCLUDED_COLUMNS)
    y = frame[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = {
        "logistic_regression": build_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        "random_forest": build_pipeline(RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
    }
    scores = {}
    for name, estimator in candidates.items():
        estimator.fit(x_train, y_train)
        scores[name] = evaluate_model(estimator, x_test, y_test, FIGURES_DIR / name)
    search = GridSearchCV(
        build_pipeline(SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
        {"model__kernel": ["linear", "rbf"], "model__C": [.5, 1, 2], "model__gamma": ["scale", "auto"]},
        scoring="roc_auc", cv=cv, n_jobs=-1, refit=True,
    )
    search.fit(x_train, y_train)
    model = search.best_estimator_
    threshold, threshold_table = find_recall_threshold(model, x_train, y_train)
    scores["svm"] = evaluate_model(model, x_test, y_test, FIGURES_DIR / "svm", threshold)
    scores["svm"]["threshold"] = threshold
    importance = permutation_importance_frame(model, x_test, y_test)
    importance.to_csv(Path(FIGURES_DIR).parent / "permutation_importance.csv", index=False)
    shap_sample(model, x_test, Path(FIGURES_DIR).parent / "shap_importance.csv")
    joblib.dump({"model": model, "threshold": threshold, "feature_columns": list(x.columns), "best_params": search.best_params_}, MODEL_PATH)
    generate_scorecard(frame, MODEL_PATH, Path(FIGURES_DIR).parent / "retention_scorecard.csv")
    metrics = {"best_params": search.best_params_, "cv_best_score": float(search.best_score_), "scores": scores}
    (Path(FIGURES_DIR).parent / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    threshold_table.to_csv(Path(FIGURES_DIR).parent / "threshold_tradeoff.csv", index=False)
    LOGGER.info("Best SVM parameters: %s", search.best_params_)
    return {"frame": frame, "model": model, "metrics": metrics, "threshold": threshold}


if __name__ == "__main__":
    train()
