"""Leakage-safe preprocessing and model pipeline construction."""
from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE


def build_preprocessor() -> ColumnTransformer:
    """Build a transformer that imputes, scales, and one-hot encodes features."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ], remainder="drop")


def build_pipeline(estimator, use_smote: bool = False) -> ImbPipeline:
    """Build an imbalanced-learn pipeline with optional training-only SMOTE."""
    steps = [("preprocessor", build_preprocessor())]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("model", estimator))
    return ImbPipeline(steps)
