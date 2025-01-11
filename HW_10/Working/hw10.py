# PROVIDED: Execute cell

import pandas as pd
import numpy as np
import copy
import re
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from IPython import display


##################
# Default parameters
FIGURESIZE = (10, 6)
FONTSIZE = 10

plt.rcParams["figure.figsize"] = FIGURESIZE
plt.rcParams["font.size"] = FONTSIZE + 2
plt.rcParams["xtick.labelsize"] = FONTSIZE
plt.rcParams["ytick.labelsize"] = FONTSIZE

# PROVIDED: Execute cell
fname = "/Users/vignesh/PycharmProjects/MLP_Homeworks/HW_10/Dataset/hw010_skel.pkl"
with open(fname, "rb") as fp:
    dat = pkl.load(fp)

# TODO: Extract the elements you need from the dat variable

X_train = dat["ins_training"]
y_train = dat["outs_training"]
X_test = dat["ins_testing"]
y_test = dat["outs_testing"]
feature_names = dat["feature_names"]

X_train = pd.DataFrame(X_train, columns=feature_names)
X_test = pd.DataFrame(X_test, columns=feature_names)

y_train = pd.DataFrame(y_train, columns=["Rainfall"])
y_test = pd.DataFrame(y_test, columns=["Rainfall"])


plt.figure(figsize=(10, 6))
plt.hist(y_train["Rainfall"], bins=30, edgecolor="black")
plt.xlabel("Rainfall (inches)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Histogram of Rainfall (inches)", fontsize=14)
plt.ylim(0, 1000)
plt.grid(linestyle="--", alpha=0.7)
plt.show()


tree_model = DecisionTreeRegressor(
    criterion="squared_error",
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
)

tree_model.fit(X_train, y_train)

train_accuracy = tree_model.score(X_train, y_train)
print("Training Accuracy:", train_accuracy)

test_accuracy = tree_model.score(X_test, y_test)
print("Testing Accuracy:", test_accuracy)

feature_importances = tree_model.feature_importances_

importance_df = pd.DataFrame(
    {"Feature": feature_names, "Importance": feature_importances}
)

importance_df = importance_df.sort_values(by="Importance", ascending=False)
importance_df

# Provided
# Hyper-parameter values that we will search over

# MAX_DEPTH
max_depths = [1, 2, 3, 4, 5, 6, 7, 8]

# TODO: copy your implementation of perform_experiment here.  Make the noted modifications


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
    :param model: a decision tree model that already has the criterion set
    :param param_list: a python list of hyper-parameter values to try
    :param param_name: the name of the hyper-parameter (e.g., as used in DecisionTreeRegressor)
    :param ins_training: Training set inputs
    :param outs_training: Training set class labels
    :param ins_testing: Testing set inputs
    :param outs_testing: Testing set class labels
    :param out_file: Name of the output dot file (None = don't generate this file)
    :param cv: Number of folds
    :param scoring: Scoring function to use
    :param feature_names: Names of the features in the same order as in the "ins"
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

    mean_test_scores = grid_search.cv_results_["mean_test_score"]
    mean_train_scores = grid_search.cv_results_["mean_train_score"]
    params = grid_search.cv_results_["params"]

    plt.plot(param_list, mean_test_scores, marker="o", label="Validation Accuracy")
    plt.plot(param_list, mean_train_scores, marker="x", label="Training Accuracy")
    plt.xlabel(param_name)
    plt.ylabel("Mean Validation Accuracy")
    plt.title(f"Accuracy vs {param_name}")
    plt.legend()
    plt.show()

    best_params = grid_search.best_params_
    print("Best Hyper-Parameters:", best_params)

    model.set_params(**best_params)
    model.fit(ins_training, outs_training)

    train_accuracy = model.score(ins_training, outs_training)
    test_accuracy = model.score(ins_testing, outs_testing)
    print("Training Accuracy:", train_accuracy)
    print("Testing Accuracy:", test_accuracy)

    if out_file is not None:
        export_graphviz(
            model, out_file=out_file, feature_names=feature_names, filled=True
        )


perform_experiment(
    tree_model, max_depths, "max_depth", X_train, y_train, X_test, y_test
)


# Feature Importance
feature_importances = tree_model.feature_importances_
importance_df = pd.DataFrame(
    {"Feature": feature_names, "Importance": feature_importances}
)
importance_df_dt = importance_df.sort_values(by="Importance", ascending=False)
importance_df_dt

# PROVIDED list
n_estimators = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190]
rf_model = RandomForestRegressor(max_depth=4, max_features=0.25, random_state=42)
rf_model

perform_experiment(
    model=rf_model,
    param_list=n_estimators,
    param_name="n_estimators",
    ins_training=X_train,
    outs_training=y_train.values.ravel(),
    ins_testing=X_test,
    outs_testing=y_test.values.ravel(),
    cv=6,
    scoring="explained_variance",
    feature_names=None,
)
