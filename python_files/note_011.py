# -------------------------------------------------
# Imports and Configuration
# -------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, KernelPCA
from pipeline_components import DataSampleDropper, DataFrameSelector, DataSampleSwapper
from visualize import *
from metrics_plots import *

# Default plotting configuration
plt.rcParams["figure.figsize"] = (6, 5)
plt.rcParams["font.size"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.style.use("ggplot")

# -------------------------------------------------
# Load and Prepare Dataset
# -------------------------------------------------

filename = "../HW_11/Dataset/heart_arrhythmia.csv"
heart = pd.read_csv(filename, delimiter=",", nrows=None)
nRows, nCols = heart.shape
print(f"There are {nRows} rows and {nCols} columns")

# -------------------------------------------------
# Data Exploration
# -------------------------------------------------

# Diagnosis Histogram
d = heart["diagnosis"].values
plt.hist(d, bins=20, edgecolor="black")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.title("Histogram of Diagnosis")
plt.show()

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------


def compute_rmse(x, y):
    return np.sqrt(np.nanmean((x - y) ** 2))


def predict_and_score(model, X, y):
    preds = model.predict(X)
    f1 = f1_score(y, preds)
    score = model.score(X, y)
    return preds, score, f1


# -------------------------------------------------
# Preprocessing Pipelines
# -------------------------------------------------

# Features and Columns
feature_names_initial = heart.columns.drop(["J"])
feature_names = heart.columns.drop(["diagnosis", "J"])

# Preprocessing pipeline for data cleaning
pipe_pre = Pipeline(
    [
        ("removeAttribs", DataFrameSelector(feature_names_initial)),
        ("Cleanup", DataSampleSwapper((("?", np.nan),))),
        ("NaNrowDropper", DataSampleDropper()),
    ]
)

# Input pipeline for feature selection and scaling
pipe_X = Pipeline(
    [
        ("pipe_pre", pipe_pre),
        ("selectAttribs", DataFrameSelector(feature_names)),
        ("scaler", RobustScaler()),
    ]
)

# Output pipeline for extracting diagnosis column
pipe_y = Pipeline(
    [
        ("pipe_pre", pipe_pre),
        ("selectAttribs", DataFrameSelector(["diagnosis"])),
    ]
)

# Transform data
X = pipe_X.fit_transform(heart)
y = pipe_y.fit_transform(heart).values.ravel()
y = y != 1  # Convert to binary: 0 = Normal, 1 = Abnormal

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

target_names = ["Normal", "Abnormal"]

# -------------------------------------------------
# Benchmark Logistic Regression Model
# -------------------------------------------------

benchmark_model = LogisticRegression(penalty=None, max_iter=100000, tol=1e-3)
benchmark_model.fit(Xtrain, ytrain)

train_preds, train_accuracy, train_f1 = predict_and_score(
    benchmark_model, Xtrain, ytrain
)
test_preds, test_accuracy, test_f1 = predict_and_score(benchmark_model, Xtest, ytest)

print(f"Training F1 Score: {train_f1:.4f}, Training Accuracy: {train_accuracy:.4f}")
print(f"Test F1 Score: {test_f1:.4f}, Test Accuracy: {test_accuracy:.4f}")

# -------------------------------------------------
# Principal Component Analysis (PCA)
# -------------------------------------------------

# PCA Variance Analysis
pca = PCA()
pca.fit(Xtrain)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

# Plot cumulative variance
plt.plot(cumulative_variance, marker="o")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Variance Explained by Principal Components")
plt.axhline(y=0.75, color="cadetblue", linestyle="--", label="75% Variance Threshold")
plt.axhline(y=0.90, color="cyan", linestyle="--", label="90% Variance Threshold")
plt.axhline(y=0.95, color="g", linestyle="--", label="95% Variance Threshold")
plt.axhline(y=0.99, color="black", linestyle="--", label="99% Variance Threshold")
plt.grid(True)
plt.show()

# -------------------------------------------------
# PCA and Logistic Regression Pipeline
# -------------------------------------------------

pcs_90_variance = np.argmax(cumulative_variance >= 0.90) + 1
print(f"For 90% of the variance, {pcs_90_variance} principal components are required.")

pipeline_pca_logreg = Pipeline(
    [
        ("pca", PCA(n_components=pcs_90_variance, whiten=True)),
        ("logreg", LogisticRegression(penalty=None, tol=1e-3, max_iter=10000)),
    ]
)

pipeline_pca_logreg.fit(Xtrain, ytrain)

train_preds_pca, train_accuracy_pca, train_f1_pca = predict_and_score(
    pipeline_pca_logreg, Xtrain, ytrain
)
test_preds_pca, test_accuracy_pca, test_f1_pca = predict_and_score(
    pipeline_pca_logreg, Xtest, ytest
)

print(
    f"Training F1 Score: {train_f1_pca:.4f}, Training Accuracy: {train_accuracy_pca:.4f}"
)
print(f"Test F1 Score: {test_f1_pca:.4f}, Test Accuracy: {test_accuracy_pca:.4f}")

# -------------------------------------------------
# Grid Search for Optimal Number of PCA Components
# -------------------------------------------------

components = np.arange(1, 120, dtype=np.int16)
param_grid = {"pca__n_components": components}

grid_search = GridSearchCV(
    pipeline_pca_logreg,
    param_grid=param_grid,
    cv=10,
    scoring="accuracy",
    return_train_score=True,
)

grid_search.fit(Xtrain, ytrain)
results_df = pd.DataFrame(grid_search.cv_results_)

# -------------------------------------------------
# Display Validation Performance
# -------------------------------------------------


def display_validation_performance(
    df, key, ylabel="Accuracy", title="Performance vs Number of PCs"
):
    plt.figure()
    df = df.sort_values(by=key)
    plt.plot(df[key], df["mean_train_score"], label="Mean Train Score", marker=".")
    plt.plot(df[key], df["mean_test_score"], label="Mean Validation Score", marker=".")
    plt.xlabel(key)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


display_validation_performance(
    results_df, key="param_pca__n_components", ylabel="Accuracy"
)

# -------------------------------------------------
# Model Evaluation and Results
# -------------------------------------------------


def display_model_results(best_model, Xtrain, ytrain, Xtest, ytest, target_names):
    train_preds = best_model.predict(Xtrain)
    train_probs = best_model.predict_proba(Xtrain)[:, 1]
    train_accuracy = best_model.score(Xtrain, ytrain)
    train_f1 = f1_score(ytrain, train_preds)

    train_cm = confusion_matrix(ytrain, train_preds)
    print("Training Results:")
    confusion_mtx_colormap(train_cm, target_names, target_names, cbarlabel="Counts")
    train_roc, train_prc, _, _ = ks_roc_prc_plot(ytrain, train_probs)
    print(f"Training F1 Score: {train_f1:.4f}, Training Accuracy: {train_accuracy:.4f}")
    plt.show()

    test_preds = best_model.predict(Xtest)
    test_probs = best_model.predict_proba(Xtest)[:, 1]
    test_accuracy = best_model.score(Xtest, ytest)
    test_f1 = f1_score(ytest, test_preds)

    test_cm = confusion_matrix(ytest, test_preds)
    print("\nTesting Results:")
    confusion_mtx_colormap(test_cm, target_names, target_names, cbarlabel="Counts")
    test_roc, test_prc, _, _ = ks_roc_prc_plot(ytest, test_probs)
    print(f"Test F1 Score: {test_f1:.4f}, Test Accuracy: {test_accuracy:.4f}")
    plt.show()


best_estimator = grid_search.best_estimator_
display_model_results(best_estimator, Xtrain, ytrain, Xtest, ytest, target_names)
