# -------------------------------------------------
# Imports and Configuration
# -------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Default plot configurations
FIGURESIZE = (10, 6)
FONTSIZE = 10
plt.rcParams["figure.figsize"] = FIGURESIZE
plt.rcParams["font.size"] = FONTSIZE + 2
plt.rcParams["xtick.labelsize"] = FONTSIZE
plt.rcParams["ytick.labelsize"] = FONTSIZE

# -------------------------------------------------
# Load Dataset and Prepare Data
# -------------------------------------------------

fname = "/mlp/datasets/mesonet_1994_2000.pkl"

with open(fname, "rb") as fp:
    dat = pkl.load(fp)

# Extract train-test data and metadata
X_train = pd.DataFrame(dat["ins_training"], columns=dat["feature_names"])
X_test = pd.DataFrame(dat["ins_testing"], columns=dat["feature_names"])
y_train = pd.DataFrame(dat["outs_training"], columns=["Rainfall"])
y_test = pd.DataFrame(dat["outs_testing"], columns=["Rainfall"])
feature_names = dat["feature_names"]

# -------------------------------------------------
# Exploratory Data Analysis
# -------------------------------------------------

# Histogram of Rainfall
plt.figure(figsize=(10, 6))
plt.hist(y_train["Rainfall"], bins=30, edgecolor="black")
plt.xlabel("Rainfall (inches)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Histogram of Rainfall (inches)", fontsize=14)
plt.ylim(0, 1000)
plt.grid(linestyle="--", alpha=0.7)
plt.show()

# Histogram of Maximum Temperature
plt.figure(figsize=(10, 6))
plt.hist(X_train["TMAX"], bins=30, edgecolor="black")
plt.xlabel("MAX Temperature (degrees Fahrenheit)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Histogram of MAX Temperature (degrees Fahrenheit)", fontsize=14)
plt.ylim(0, 1000)
plt.grid(linestyle="--", alpha=0.7)
plt.show()

# -------------------------------------------------
# Train Initial Decision Tree Regressor
# -------------------------------------------------

# Define DecisionTreeRegressor
tree_model = DecisionTreeRegressor(
    criterion="squared_error",
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
)

# Fit and evaluate model
tree_model.fit(X_train, y_train)
train_accuracy = tree_model.score(X_train, y_train)
test_accuracy = tree_model.score(X_test, y_test)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")

# Feature importance for Decision Tree
feature_importances = tree_model.feature_importances_
importance_df_dt = pd.DataFrame(
    {"Feature": feature_names, "Importance": feature_importances}
).sort_values(by="Importance", ascending=False)
print(importance_df_dt)

# -------------------------------------------------
# Define Hyperparameter Tuning Function
# -------------------------------------------------


def perform_experiment(
    model,
    param_list,
    param_name,
    ins_training,
    outs_training,
    ins_testing,
    outs_testing,
    out_file=None,
    cv=5,
    scoring="explained_variance",
    feature_names=None,
):
    """
    Performs hyperparameter tuning and evaluates the model.

    :param model: Regressor model to tune
    :param param_list: List of parameter values to test
    :param param_name: Name of the parameter to tune
    :param ins_training: Training feature data
    :param outs_training: Training target data
    :param ins_testing: Testing feature data
    :param outs_testing: Testing target data
    :param out_file: DOT file to export the model (optional)
    :param cv: Number of cross-validation folds
    :param scoring: Scoring metric for evaluation
    :param feature_names: List of feature names for export (optional)
    """
    param_grid = {param_name: param_list}
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
    )
    grid_search.fit(ins_training, outs_training)

    # Plot training and validation accuracy
    mean_test_scores = grid_search.cv_results_["mean_test_score"]
    mean_train_scores = grid_search.cv_results_["mean_train_score"]
    plt.plot(param_list, mean_test_scores, marker="o", label="Validation Accuracy")
    plt.plot(param_list, mean_train_scores, marker="x", label="Training Accuracy")
    plt.xlabel(param_name)
    plt.ylabel("Mean Validation Accuracy")
    plt.title(f"Accuracy vs {param_name}")
    plt.legend()
    plt.show()

    # Evaluate best model
    best_params = grid_search.best_params_
    print("Best Hyper-Parameters:", best_params)

    model.set_params(**best_params)
    model.fit(ins_training, outs_training)

    train_accuracy = model.score(ins_training, outs_training)
    test_accuracy = model.score(ins_testing, outs_testing)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")


# -------------------------------------------------
# Hyperparameter Tuning for Decision Tree Regressor
# -------------------------------------------------

max_depths = [2, 3, 5, 7, 9, 11]
perform_experiment(
    tree_model, max_depths, "max_depth", X_train, y_train, X_test, y_test
)

# -------------------------------------------------
# Train and Evaluate Random Forest Regressor
# -------------------------------------------------

n_estimators = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190]
rf_model = RandomForestRegressor(max_depth=4, max_features=0.25, random_state=42)
perform_experiment(
    rf_model,
    n_estimators,
    "n_estimators",
    X_train,
    y_train.values.ravel(),
    X_test,
    y_test.values.ravel(),
)

# -------------------------------------------------
# Train and Evaluate Gradient Boosting Regressor
# -------------------------------------------------

gb_model = GradientBoostingRegressor(max_depth=4, max_features=0.25, random_state=42)
perform_experiment(
    gb_model,
    n_estimators,
    "n_estimators",
    X_train,
    y_train.values.ravel(),
    X_test,
    y_test.values.ravel(),
)

# -------------------------------------------------
# Feature Importance Comparisons
# -------------------------------------------------

# Random Forest Feature Importance
feature_importances_rf = rf_model.feature_importances_
importance_df_rf = pd.DataFrame(
    {"Feature": feature_names, "Importance": feature_importances_rf}
).sort_values(by="Importance", ascending=False)
print(importance_df_rf)

# Gradient Boosting Feature Importance
feature_importances_gb = gb_model.feature_importances_
importance_df_gb = pd.DataFrame(
    {"Feature": feature_names, "Importance": feature_importances_gb}
).sort_values(by="Importance", ascending=False)
print(importance_df_gb)


# -------------------------------------------------
# Define Hyperparameter Tuning Function - Customized After MLP Course Completion
# -------------------------------------------------


def perform_experiment(
    model,
    param_grid,
    ins_training,
    outs_training,
    ins_testing,
    outs_testing,
    out_file=None,
    cv=5,
    scoring="explained_variance",
    feature_names=None,
):
    """
    Performs hyperparameter tuning and evaluates the model.

    :param model: Regressor model to tune
    :param param_grid: Dictionary of hyperparameters and their values
    :param ins_training: Training feature data
    :param outs_training: Training target data
    :param ins_testing: Testing feature data
    :param outs_testing: Testing target data
    :param out_file: DOT file to export the model (optional)
    :param cv: Number of cross-validation folds
    :param scoring: Scoring metric for evaluation
    :param feature_names: List of feature names for export (optional)
    """
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
    )
    grid_search.fit(ins_training, outs_training)

    # Plot training and validation accuracy
    mean_test_scores = grid_search.cv_results_["mean_test_score"]
    mean_train_scores = grid_search.cv_results_["mean_train_score"]

    for param_name in param_grid.keys():
        plt.plot(
            [params[param_name] for params in grid_search.cv_results_["params"]],
            mean_test_scores,
            marker="o",
            label=f"Validation Accuracy ({param_name})",
        )
        plt.plot(
            [params[param_name] for params in grid_search.cv_results_["params"]],
            mean_train_scores,
            marker="x",
            label=f"Training Accuracy ({param_name})",
        )
    plt.xlabel("Parameter Values")
    plt.ylabel("Mean Validation Accuracy")
    plt.title("Accuracy vs Parameter Tuning")
    plt.legend()
    plt.show()

    # Evaluate best model
    best_params = grid_search.best_params_
    print("Best Hyper-Parameters:", best_params)

    model.set_params(**best_params)
    model.fit(ins_training, outs_training)

    train_accuracy = model.score(ins_training, outs_training)
    test_accuracy = model.score(ins_testing, outs_testing)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")


# -------------------------------------------------
