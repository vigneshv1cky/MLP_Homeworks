import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, LeaveOneOut, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer


def run_grid_search_with_seeds(X, y, param_grid, seeds, scoring="accuracy"):
    """
    Performs grid search with LOOCV on a RandomForestClassifier using a pipeline that includes
    scaling (and optionally normalization or other transformations) for different random seeds.

    Parameters:
    - X: Features array.
    - y: Target array.
    - param_grid: Dictionary of hyperparameters for the classifier.
      (e.g., {'n_estimators': [10, 50, 100], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]})
    - seeds: List of integer seeds for the classifier's random_state.
    - scoring: Scoring metric for grid search (default is 'accuracy').

    Returns:
    - results: Dictionary mapping each seed to its best parameters, best LOOCV score, and test set score.
    """
    results = {}

    # Split the data into training and testing sets.
    # Here, we use a fixed random state for reproducibility.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define LOOCV for the training set.
    loo = LeaveOneOut()

    for seed in seeds:
        print(f"Random state: {seed}")

        # Build a pipeline.
        # You can uncomment the 'normalizer' step if normalization is desired.
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                # ('normalizer', Normalizer()),  # Uncomment this line to add normalization
                ("clf", RandomForestClassifier(random_state=seed)),
            ]
        )

        # Prepare parameter grid with proper prefix for pipeline steps (parameters for the classifier).
        pipeline_param_grid = {
            f"clf__{key}": value for key, value in param_grid.items()
        }

        # Set up GridSearchCV with LOOCV.
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=pipeline_param_grid,
            cv=loo,
            scoring=scoring,
            n_jobs=-1,
        )

        # Fit grid search on the training data.
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_cv_score = grid_search.best_score_

        print("Best parameters:", best_params)
        print("Best LOOCV accuracy score on training set:", best_cv_score)

        # Evaluate the best model on the test set.
        test_score = grid_search.score(X_test, y_test)
        print("Test set accuracy score:", test_score)
        print("=" * 50)

        # Store the results.
        results[seed] = {
            "best_params": best_params,
            "best_cv_score": best_cv_score,
            "test_score": test_score,
        }

    return results


# Example usage:
if __name__ == "__main__":
    # Load the iris dataset.
    data = load_iris()
    X, y = data.data, data.target

    # Define the hyperparameter grid for RandomForestClassifier.
    param_grid = {
        "n_estimators": [10, 50, 100],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
    }

    # List of different random seeds.
    seeds = np.random.randint(0, 100, size=10)

    # Run the grid search with LOOCV for each seed.
    results = run_grid_search_with_seeds(X, y, param_grid, seeds)
