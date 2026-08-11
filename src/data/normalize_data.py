from pathlib import Path
import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_data"
MODELS_DIR = PROJECT_ROOT / "models"


def normalize_data():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")

    scaler = StandardScaler()

    # Fit ONLY on training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply same transformation to test data
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
    )

    X_train_scaled.to_csv(
        PROCESSED_DIR / "X_train_scaled.csv",
        index=False,
    )

    X_test_scaled.to_csv(
        PROCESSED_DIR / "X_test_scaled.csv",
        index=False,
    )

    with open(MODELS_DIR / "scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    print("Data normalization completed.")
    print(f"X_train_scaled: {X_train_scaled.shape}")
    print(f"X_test_scaled:  {X_test_scaled.shape}")


if __name__ == "__main__":
    normalize_data()