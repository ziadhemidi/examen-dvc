from pathlib import Path
import pickle

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed_data"
MODELS_DIR = PROJECT_ROOT / "models"

CV_FOLDS = 5


def run_grid_search():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(
        PROCESSED_DIR / "X_train_scaled.csv"
    )

    y_train = pd.read_csv(
        PROCESSED_DIR / "y_train.csv"
    ).iloc[:, 0]

    model = SVR()

    param_grid = {
        "kernel": ["rbf", "linear"],
        "C": [0.1, 1, 10, 100],
        "epsilon": [0.01, 0.1, 0.2],
        "gamma": ["scale", "auto"],
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=CV_FOLDS,
        n_jobs=-1,
        verbose=2,
    )

    print("Starting GridSearchCV...")

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_

    with open(MODELS_DIR / "best_params.pkl", "wb") as file:
        pickle.dump(best_params, file)

    print("Grid search completed.")
    print(f"Best parameters: {best_params}")
    print(
        f"Best CV MSE: {-grid_search.best_score_:.4f}"
    )


if __name__ == "__main__":
    run_grid_search()