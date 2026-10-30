"""Model evaluation, threshold tuning, and explainability plots."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve, precision_recall_curve)

LOGGER = logging.getLogger(__name__)


def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series, figure_dir: Path,
                   threshold: float = 0.5) -> dict[str, float]:
    """Compute classification metrics and save confusion, ROC, and PR plots."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "accuracy": float((predictions == y_test).mean()),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
    }
    LOGGER.info("Metrics at threshold %.2f: %s", threshold, metrics)
    LOGGER.info("Classification report:\n%s", classification_report(y_test, predictions, zero_division=0))
    matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(5, 4)); sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion matrix (threshold={threshold:.2f})"); plt.tight_layout()
    plt.savefig(figure_dir / "confusion_matrix.png", dpi=140); plt.close()
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    plt.figure(figsize=(6, 4)); plt.plot(fpr, tpr, label=f"ROC-AUC {metrics['roc_auc']:.3f}"); plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend(); plt.tight_layout()
    plt.savefig(figure_dir / "roc_curve.png", dpi=140); plt.close()
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    plt.figure(figsize=(6, 4)); plt.plot(recall, precision, label=f"PR-AUC {metrics['pr_auc']:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.tight_layout()
    plt.savefig(figure_dir / "pr_curve.png", dpi=140); plt.close()
    return metrics


def find_recall_threshold(model, x_validation: pd.DataFrame, y_validation: pd.Series,
                          minimum_recall: float = .80) -> tuple[float, pd.DataFrame]:
    """Select the highest precision threshold meeting a minimum positive recall."""
    probabilities = model.predict_proba(x_validation)[:, 1]
    rows = []
    for threshold in np.linspace(.10, .90, 81):
        predictions = (probabilities >= threshold).astype(int)
        rows.append({"threshold": threshold, "recall": recall_score(y_validation, predictions, zero_division=0),
                     "precision": precision_score(y_validation, predictions, zero_division=0),
                     "f1": f1_score(y_validation, predictions, zero_division=0)})
    table = pd.DataFrame(rows)
    eligible = table[table["recall"] >= minimum_recall]
    selected = eligible.sort_values(["precision", "threshold"], ascending=[False, False]).iloc[0] if not eligible.empty else table.iloc[table["f1"].idxmax()]
    return float(selected["threshold"]), table


def permutation_importance_frame(model, x_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Return feature-level permutation importance for the full pipeline."""
    result = permutation_importance(model, x_test, y_test, scoring="roc_auc", n_repeats=5, random_state=42, n_jobs=-1)
    return pd.DataFrame({"feature": x_test.columns, "importance": result.importances_mean}).sort_values("importance", ascending=False)


def shap_sample(model, x_sample: pd.DataFrame, output_path: Path) -> None:
    """Compute Kernel SHAP values for a small sample when SHAP is available."""
    try:
        import shap
        background = x_sample.head(min(20, len(x_sample)))
        explain = x_sample.head(min(40, len(x_sample)))
        predict = lambda values: model.predict_proba(pd.DataFrame(values, columns=x_sample.columns))[:, 1]
        explainer = shap.KernelExplainer(predict, background)
        values = explainer.shap_values(explain, nsamples=100)
        values = values[0] if isinstance(values, list) else values
        importance = pd.DataFrame({"feature": explain.columns, "mean_abs_shap": np.abs(values).mean(axis=0)})
        importance.sort_values("mean_abs_shap", ascending=False).to_csv(output_path, index=False)
    except Exception as error:
        LOGGER.warning("SHAP export skipped: %s", error)
