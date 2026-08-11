from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw_data" / "raw.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_data"

TARGET_COLUMN = "silica_concentrate"
TEST_SIZE = 0.2
RANDOM_STATE = 42

def split_data():
    # Create output directory if it does not exist
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading data from: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)

    # Check that target exists
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in dataset."
        )

    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[[TARGET_COLUMN]]

    # Remove non-numeric features such as datetime/string columns
    non_numeric_columns = X.select_dtypes(exclude="number").columns.tolist()

    if non_numeric_columns:
        print(f"Removing non-numeric columns: {non_numeric_columns}")
        X = X.select_dtypes(include="number")

    print(f"Features used for training: {list(X.columns)}")

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # Save datasets
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    print("\nData split completed successfully.")
    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test:  {y_test.shape}")

    print(f"\nFiles saved to: {PROCESSED_DIR}")


if __name__ == "__main__":
    split_data()