from pathlib import Path
import json
import pickle

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_data"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "metrics"
DATA_DIR = PROJECT_ROOT / "data"


def evaluate_model():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    X_test = pd.read_csv(
        PROCESSED_DIR / "X_test_scaled.csv"
    )

    y_test = pd.read_csv(
        PROCESSED_DIR / "y_test.csv"
    ).iloc[:, 0]

    with open(MODELS_DIR / "model.pkl", "rb") as file:
        model = pickle.load(file)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    scores = {
        "mse": float(mse),
        "mae": float(mae),
        "r2": float(r2),
    }

    predictions_df = pd.DataFrame(
        {
            "actual": y_test,
            "prediction": predictions,
        }
    )

    predictions_df.to_csv(
        DATA_DIR / "predictions.csv",
        index=False,
    )

    with open(METRICS_DIR / "scores.json", "w") as file:
        json.dump(scores, file, indent=4)

    print("Model evaluation completed.")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2:  {r2:.4f}")
    print(f"Predictions saved to: {DATA_DIR / 'predictions.csv'}")
    print(f"Scores saved to: {METRICS_DIR / 'scores.json'}")


if __name__ == "__main__":
    evaluate_model()