# -------------------------------------------------
# Imports and Default Configurations
# -------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from IPython import display

# Configure plot aesthetics
FIGURESIZE = (10, 6)
FONTSIZE = 10
plt.rcParams["figure.figsize"] = FIGURESIZE
plt.rcParams["font.size"] = FONTSIZE + 2
plt.rcParams["xtick.labelsize"] = FONTSIZE
plt.rcParams["ytick.labelsize"] = FONTSIZE

# -------------------------------------------------
# Load Dataset and Extract Variables
# -------------------------------------------------

fname = "/Users/vignesh/PycharmProjects/MLP_Homeworks/HW9/Dataset/neuron_data3.pkl"

with open(fname, "rb") as fp:
    dat = pkl.load(fp)

# Extract data
X_train = pd.DataFrame(dat["ins_training"], columns=dat["feature_names"])
y_train = dat["outs_training"]
X_test = pd.DataFrame(dat["ins_testing"], columns=dat["feature_names"])
y_test = dat["outs_testing"]
feature_names = dat["feature_names"]
feature_mins = dat["feature_mins"]
feature_maxes = dat["feature_maxes"]
feature_units = dat["feature_units"]
class_names = dat["class_names"]

# -------------------------------------------------
# Exploratory Data Analysis
# -------------------------------------------------

# Plot histogram of Excitatory Postsynaptic Potential
X_train["ExcitatoryPostsynapticPotential"].hist(bins=50, figsize=(12, 4))
plt.xlabel("Excitatory Postsynaptic Potential (mV)")
plt.ylabel("Frequency")
plt.title("Histogram of Excitatory Postsynaptic Potential")
plt.show()

# -------------------------------------------------
# Train Initial Decision Tree Classifier
# -------------------------------------------------

# Initialize and train DecisionTreeClassifier
tree_model = DecisionTreeClassifier(max_depth=10, random_state=42)
tree_model.fit(X_train, y_train)

# Evaluate training and testing accuracy
train_accuracy = tree_model.score(X_train, y_train)
test_accuracy = tree_model.score(X_test, y_test)
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")

# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(tree_model, feature_names=feature_names, class_names=class_names, filled=True)
plt.show()

# -------------------------------------------------
# Define Hyperparameter Tuning Experiment Function
# -------------------------------------------------


def perform_experiment(
    model,
    param_list,
    param_name,
    ins_training,
    outs_training,
    ins_testing,
    outs_testing,
    out_file="tree_model.dot",
    cv=30,
    scoring="accuracy",
):
    """
    Perform hyperparameter tuning and plot results.

    :param model: A DecisionTreeClassifier instance.
    :param param_list: List of hyperparameter values to test.
    :param param_name: Name of the hyperparameter to tune.
    :param ins_training: Training feature data.
    :param outs_training: Training labels.
    :param ins_testing: Testing feature data.
    :param outs_testing: Testing labels.
    :param out_file: Name of the DOT file for exporting the decision tree.
    :param cv: Number of cross-validation folds.
    :param scoring: Scoring metric for evaluation.
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

    # Plot training and validation accuracy
    plt.plot(param_list, mean_test_scores, marker="o", label="Validation Accuracy")
    plt.plot(param_list, mean_train_scores, marker="x", label="Training Accuracy")
    plt.xlabel(param_name)
    plt.ylabel("Mean Accuracy")
    plt.title(f"Accuracy vs {param_name}")
    plt.legend()
    plt.show()

    # Best hyperparameters and final model evaluation
    best_params = grid_search.best_params_
    print("Best Hyperparameters:", best_params)

    model.set_params(**best_params)
    model.fit(ins_training, outs_training)

    train_accuracy = model.score(ins_training, outs_training)
    test_accuracy = model.score(ins_testing, outs_testing)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")

    # Export and visualize the decision tree
    # export_graphviz(
    #     model,
    #     out_file=out_file,
    #     feature_names=feature_names,
    #     class_names=class_names,
    #     filled=True,
    # )
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, class_names=class_names, filled=True)
    plt.show()


# -------------------------------------------------
# Perform Experiments with Different Parameters
# -------------------------------------------------

# Experiment with max_leaf_nodes and 'gini' criterion
max_nodes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
perform_experiment(
    DecisionTreeClassifier(criterion="gini"),
    max_nodes,
    "max_leaf_nodes",
    X_train,
    y_train,
    X_test,
    y_test,
)

# Experiment with min_samples_leaf and 'gini' criterion
min_samples_leaf = [4, 5, 6, 7, 9, 11, 13, 15, 17]
perform_experiment(
    DecisionTreeClassifier(criterion="gini"),
    min_samples_leaf,
    "min_samples_leaf",
    X_train,
    y_train,
    X_test,
    y_test,
)

# Experiment with max_leaf_nodes and 'entropy' criterion
perform_experiment(
    DecisionTreeClassifier(criterion="entropy"),
    max_nodes,
    "max_leaf_nodes",
    X_train,
    y_train,
    X_test,
    y_test,
)

# Experiment with min_samples_leaf and 'entropy' criterion
perform_experiment(
    DecisionTreeClassifier(criterion="entropy"),
    min_samples_leaf,
    "min_samples_leaf",
    X_train,
    y_train,
    X_test,
    y_test,
)
