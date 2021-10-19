import json
import fire
import pandas as pd
import joblib
from pathlib import Path
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)


TARGET_NAME = "Survived"


def evaluate(
    processed_valid_data_path: str,
    model_path: str,
    metrics_path: str,
    classification_report_path: str,
    confusion_matrix_path: str,
) -> None:
    valid = pd.read_csv(processed_valid_data_path, index_col=0)
    X_valid = valid.drop(TARGET_NAME, axis=1)
    y_valid = valid[TARGET_NAME]

    model = joblib.load(model_path)
    y_pred = model.predict(X_valid)

    accuracy = accuracy_score(y_valid, y_pred)
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).write_text(
        json.dumps(
            {
                "metrics": [
                    {"name": "accuracy", "numberValue": accuracy, "format": "RAW"}
                ]
            }
        )
    )

    report = classification_report(y_valid, y_pred)
    Path(classification_report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(classification_report_path).write_text(report)

    confusion_matrix = ConfusionMatrixDisplay.from_predictions(y_valid, y_pred)
    Path(confusion_matrix_path).parent.mkdir(parents=True, exist_ok=True)
    confusion_matrix.figure_.savefig(confusion_matrix_path)


if __name__ == "__main__":
    fire.Fire(evaluate)
