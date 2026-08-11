from pathlib import Path
import pickle

import pandas as pd
from sklearn.svm import SVR


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_data"
MODELS_DIR = PROJECT_ROOT / "models"


def train_model():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(
        PROCESSED_DIR / "X_train_scaled.csv"
    )

    y_train = pd.read_csv(
        PROCESSED_DIR / "y_train.csv"
    ).iloc[:, 0]

    with open(MODELS_DIR / "best_params.pkl", "rb") as file:
        best_params = pickle.load(file)

    print(f"Training model with parameters: {best_params}")

    model = SVR(**best_params)

    model.fit(X_train, y_train)

    with open(MODELS_DIR / "model.pkl", "wb") as file:
        pickle.dump(model, file)

    print("Model training completed.")
    print(f"Model saved to: {MODELS_DIR / 'model.pkl'}")


if __name__ == "__main__":
    train_model()