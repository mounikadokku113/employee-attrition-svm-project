"""Batch inference entry point for new employee records."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.config import MODEL_PATH, SCORECARD_PATH
from src.data_loader import load_data
from src.scorecard import generate_scorecard


def predict(input_path: str | Path, model_path: Path = MODEL_PATH, output_path: Path = SCORECARD_PATH):
    """Generate a retention scorecard from a CSV input."""
    return generate_scorecard(load_data(input_path), model_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score employee attrition risk")
    parser.add_argument("input", help="Path to an employee CSV")
    parser.add_argument("--output", default=str(SCORECARD_PATH))
    parser.add_argument("--model", default=str(MODEL_PATH))
    args = parser.parse_args()
    result = predict(args.input, Path(args.model), Path(args.output))
    print(result.head(10).to_string(index=False))
