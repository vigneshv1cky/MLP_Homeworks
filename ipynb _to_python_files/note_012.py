# -------------------------------------------------
# Imports and Configuration
# -------------------------------------------------

import sys
import pandas as pd
import numpy as np
import os
import time as timelib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.manifold import Isomap
import pickle as pkl

# Default figure parameters
plt.rcParams["figure.figsize"] = (6, 5)
plt.rcParams["font.size"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.style.use("ggplot")

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

fname = "../HW12/Dataset/hw12_dataset_v2.pkl"
with open(fname, "rb") as fp:
    dat = pkl.load(fp)

Xtrain = dat["Xtrain"]
ytrain = dat["ytrain"]
Xtest = dat["Xtest"]
ytest = dat["ytest"]

# -------------------------------------------------
# Initial Linear Regression Model
# -------------------------------------------------

lr_model = LinearRegression()
lr_model.fit(Xtrain, ytrain)

ytrain_pred = lr_model.predict(Xtrain)
ytest_pred = lr_model.predict(Xtest)

train_rmse = np.sqrt(mean_squared_error(ytrain, ytrain_pred))
test_rmse = np.sqrt(mean_squared_error(ytest, ytest_pred))

# Plot predicted vs true values for Linear Regression
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(ytrain, ytrain_pred, alpha=0.6)
plt.title("Training Data: Predicted vs True")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.plot(
    [min(ytrain), max(ytrain)],
    [min(ytrain), max(ytrain)],
    color="green",
    linestyle="--",
)

plt.subplot(1, 2, 2)
plt.scatter(ytest, ytest_pred, alpha=0.6)
plt.title("Testing Data: Predicted vs True")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.plot(
    [min(ytest), max(ytest)], [min(ytest), max(ytest)], color="green", linestyle="--"
)

plt.tight_layout()
plt.show()

print(f"Training RMSE: {train_rmse:.4f}")
print(f"Testing RMSE: {test_rmse:.4f}")

# -------------------------------------------------
# Linear Regression with Isomap Embedding
# -------------------------------------------------

pipeline = Pipeline(
    [
        ("embedding", Isomap(n_neighbors=10, n_components=8)),
        ("regressor", LinearRegression()),
    ]
)
pipeline.fit(Xtrain, ytrain)

ytrain_pred = pipeline.predict(Xtrain)
ytest_pred = pipeline.predict(Xtest)

train_rmse = np.sqrt(mean_squared_error(ytrain, ytrain_pred))
test_rmse = np.sqrt(mean_squared_error(ytest, ytest_pred))

# Plot predicted vs true values for Isomap + Linear Regression
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(ytrain, ytrain_pred, alpha=0.6)
plt.title("Training Data: Predicted vs True")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.plot(
    [min(ytrain), max(ytrain)],
    [min(ytrain), max(ytrain)],
    color="green",
    linestyle="--",
)

plt.subplot(1, 2, 2)
plt.scatter(ytest, ytest_pred, alpha=0.6)
plt.title("Testing Data: Predicted vs True")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.plot(
    [min(ytest), max(ytest)], [min(ytest), max(ytest)], color="green", linestyle="--"
)

plt.tight_layout()
plt.show()

print(f"Training RMSE: {train_rmse:.4f}")
print(f"Testing RMSE: {test_rmse:.4f}")

# -------------------------------------------------
# Grid Search for Isomap + Linear Regression
# -------------------------------------------------

pipeline = Pipeline([("embedding", Isomap()), ("regressor", LinearRegression())])
param_grid = {"embedding__n_neighbors": [20, 25], "embedding__n_components": [10, 20]}

grid_search = GridSearchCV(
    pipeline, param_grid, scoring="neg_mean_squared_error", cv=5, verbose=1
)
grid_search.fit(Xtrain, ytrain)

best_pipeline = grid_search.best_estimator_

ytrain_pred = best_pipeline.predict(Xtrain)
ytest_pred = best_pipeline.predict(Xtest)

train_rmse = np.sqrt(mean_squared_error(ytrain, ytrain_pred))
test_rmse = np.sqrt(mean_squared_error(ytest, ytest_pred))

# Plot predicted vs true values for Grid Search Best Pipeline
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(ytrain, ytrain_pred, alpha=0.6)
plt.title("Training Data: Predicted vs True")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.plot(
    [min(ytrain), max(ytrain)],
    [min(ytrain), max(ytrain)],
    color="green",
    linestyle="--",
)

plt.subplot(1, 2, 2)
plt.scatter(ytest, ytest_pred, alpha=0.6)
plt.title("Testing Data: Predicted vs True")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.plot(
    [min(ytest), max(ytest)], [min(ytest), max(ytest)], color="green", linestyle="--"
)

plt.tight_layout()
plt.show()

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Testing RMSE: {test_rmse:.4f}")

# -------------------------------------------------
# Isomap + Polynomial Regression
# -------------------------------------------------

pipeline = Pipeline(
    [
        ("embedding", Isomap(n_neighbors=20, n_components=25)),
        ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
        ("regressor", LinearRegression()),
    ]
)
pipeline.fit(Xtrain, ytrain)

ytrain_pred = pipeline.predict(Xtrain)
ytest_pred = pipeline.predict(Xtest)

train_rmse = np.sqrt(mean_squared_error(ytrain, ytrain_pred))
test_rmse = np.sqrt(mean_squared_error(ytest, ytest_pred))

# Plot predicted vs true values for Isomap + Polynomial Regression
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(ytrain, ytrain_pred, alpha=0.6)
plt.title("Training Data: Predicted vs True")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.plot(
    [min(ytrain), max(ytrain)],
    [min(ytrain), max(ytrain)],
    color="green",
    linestyle="--",
)

plt.subplot(1, 2, 2)
plt.scatter(ytest, ytest_pred, alpha=0.6)
plt.title("Testing Data: Predicted vs True")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.plot(
    [min(ytest), max(ytest)], [min(ytest), max(ytest)], color="green", linestyle="--"
)

plt.tight_layout()
plt.show()

print(f"Training RMSE: {train_rmse:.4f}")
print(f"Testing RMSE: {test_rmse:.4f}")

# -------------------------------------------------
# Grid Search for Isomap + Polynomial Regression
# -------------------------------------------------

pipeline = Pipeline(
    [
        ("embedding", Isomap()),
        ("polynomial", PolynomialFeatures()),
        ("regressor", LinearRegression()),
    ]
)
param_grid = {
    "embedding__n_neighbors": [20, 25],
    "embedding__n_components": [15, 20, 25],
    "polynomial__degree": [2],
}

grid_search = GridSearchCV(
    pipeline, param_grid, scoring="neg_mean_squared_error", cv=5, verbose=1
)
grid_search.fit(Xtrain, ytrain)

best_pipeline = grid_search.best_estimator_

ytrain_pred = best_pipeline.predict(Xtrain)
ytest_pred = best_pipeline.predict(Xtest)

train_rmse = np.sqrt(mean_squared_error(ytrain, ytrain_pred))
test_rmse = np.sqrt(mean_squared_error(ytest, ytest_pred))

# Plot predicted vs true values for Grid Search Best Pipeline with Polynomial Features
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(ytrain, ytrain_pred, alpha=0.6)
plt.title("Training Data: Predicted vs True")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.plot(
    [min(ytrain), max(ytrain)],
    [min(ytrain), max(ytrain)],
    color="green",
    linestyle="--",
)

plt.subplot(1, 2, 2)
plt.scatter(ytest, ytest_pred, alpha=0.6)
plt.title("Testing Data: Predicted vs True")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.plot(
    [min(ytest), max(ytest)], [min(ytest), max(ytest)], color="green", linestyle="--"
)

plt.tight_layout()
plt.show()

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Testing RMSE: {test_rmse:.4f}")

# -------------------------------------------------
