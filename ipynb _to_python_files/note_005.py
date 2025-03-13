# ----------------------------
# Regression
# Dint Use Gridsearch CV
# ----------------------------

import pickle as pkl
import pandas as pd
import numpy as np
import os, time
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso

# Load the BMI data from all the folds
fname = "/Users/vignesh/PycharmProjects/MLP_Homeworks/hw5/dataset/bmi_dataset_local.pkl"

with open(fname, "rb") as f:
    bmi = pkl.load(f)
    theta_folds = bmi["theta"]
    dtheta_folds = bmi["dtheta"]
    ddtheta_folds = bmi["ddtheta"]
    torque_folds = bmi["torque"]
    time_folds = bmi["time"]
    MI_folds = bmi["MI"]

# Extract fold indices for the training, validation and testing sets
trainset_fold_inds = [19]
validationset_fold_inds = [0, 1, 2, 3]
testset_fold_inds = [4, 5, 6, 7]

# Data to predict: Joint position
predict_folds = theta_folds
predict_index = 1


def extract_data_set(folds, data):
    output = [np.concatenate([d[f] for f in folds]) for d in data]
    return tuple(output)


# Combine the folds into singular numpy arrays for each of the sets
timetrain, Xtrain, ytrain = extract_data_set(
    trainset_fold_inds, [time_folds, MI_folds, predict_folds]
)
ytrain = np.squeeze(ytrain[:, predict_index])

timeval, Xval, yval = extract_data_set(
    validationset_fold_inds, [time_folds, MI_folds, predict_folds]
)
yval = np.squeeze(yval[:, predict_index])

timetest, Xtest, ytest = extract_data_set(
    testset_fold_inds, [time_folds, MI_folds, predict_folds]
)
ytest = np.squeeze(ytest[:, predict_index])

# Construct and train a model using the training set
model_lnr = Pipeline(
    [("scalar", StandardScaler()), ("linear regression", LinearRegression())]
)

model_lnr.fit(Xtrain, ytrain)


# Evaluate performance on training and validation sets
def predict_score_eval(model, X, y):
    preds = model.predict(X)
    mse = np.sum(np.square(y - preds), axis=0) / y.shape[0]
    var = np.var(y, axis=0)
    fvaf = 1 - mse / var
    rmse = np.sqrt(mse)
    return mse, rmse * 180 / np.pi, fvaf


mse_train, rmse_train_deg, fvaf_train = predict_score_eval(model_lnr, Xtrain, ytrain)
mse_val, rmse_val_deg, fvaf_val = predict_score_eval(model_lnr, Xval, yval)

# Create a Lasso model Pipeline
model_regularized = Pipeline(
    [("scalar", StandardScaler()), ("lasso regression", Lasso())]
)

alphas = np.logspace(-7, 0, base=10, num=28)


def hyperparameter_loop(model, alphas):
    rmse_train = np.zeros((len(alphas),))
    rmse_valid = np.zeros((len(alphas),))
    fvaf_train = np.zeros((len(alphas),))
    fvaf_valid = np.zeros((len(alphas),))

    for i, a in enumerate(alphas):
        model_tmp = Pipeline(
            [("scalar", StandardScaler()), ("lasso regression", Lasso(alpha=a))]
        )

        model_tmp.fit(Xtrain, ytrain)

        _, rmse_deg_train, fvaf_train[i] = predict_score_eval(model_tmp, Xtrain, ytrain)
        _, rmse_deg_valid, fvaf_valid[i] = predict_score_eval(model_tmp, Xval, yval)

        rmse_train[i] = rmse_deg_train
        rmse_valid[i] = rmse_deg_valid

    return rmse_train, fvaf_train, rmse_valid, fvaf_valid


rmse_train_lasso, fvaf_train_lasso, rmse_val_lasso, fvaf_val_lasso = (
    hyperparameter_loop(model_regularized, alphas)
)

# Plotting FVAF vs Alpha
plt.figure(figsize=(10, 6))
plt.plot(alphas, fvaf_train_lasso, label="Train FVAF", marker="o")
plt.plot(alphas, fvaf_val_lasso, label="Validation FVAF", marker="o")
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("FVAF")
plt.title("FVAF vs Alpha")
plt.legend()
plt.grid(True)
plt.show()

# Plotting RMSE vs Alpha
plt.figure(figsize=(14, 8))
plt.plot(alphas, rmse_train_lasso, label="Train RMSE (degrees)", marker="o")
plt.plot(alphas, rmse_val_lasso, label="Validation RMSE (degrees)", marker="o")
plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("RMSE (degrees)")
plt.title("RMSE vs Alpha")
plt.legend()
plt.grid(True)
plt.show()

# Best alpha based on validation set performance
best_alpha_idx = np.argmin(rmse_val_lasso)
best_alpha_value = alphas[best_alpha_idx]

# Set the regularized model alpha to the best value and fit the model to the training data
model_regularized.set_params(**{"lasso regression__alpha": best_alpha_value})
model_regularized.fit(Xtrain, ytrain)

# Predictions for test data and evaluation metrics
predtest_regularized = model_regularized.predict(Xtest)
mse_test_regularized, rmse_test_deg_regularized, fvaf_test_regularized = (
    predict_score_eval(model_regularized, Xtest, ytest)
)

print(
    f"Regularized Model Test FVAF: {fvaf_test_regularized}, RMSE: {rmse_test_deg_regularized}"
)

# Predictions using Linear Regression for test data and evaluation metrics
preds_lnr_test = model_lnr.predict(Xtest)
mse_test_linear, rmse_test_deg_linear, fvaf_test_linear = predict_score_eval(
    model_lnr, Xtest, ytest
)

print(f"Linear Regression Test FVAF: {fvaf_test_linear}, RMSE: {rmse_test_deg_linear}")

# Plotting results for comparison
plt.figure(figsize=(12, 8))
plt.plot(
    timetest, ytest * 180 / np.pi, label="Ground Truth", linestyle="-", color="blue"
)
plt.plot(
    timetest,
    predtest_regularized * 180 / np.pi,
    label="Regularized Model Predictions (Lasso)",
    linestyle="-",
    color="r",
)
plt.plot(
    timetest,
    preds_lnr_test * 180 / np.pi,
    label="Linear Model Predictions",
    linestyle="-",
    color="green",
)
plt.xlim([700, 720])
plt.xlabel("Time")
plt.ylabel("Joint Position (Degrees)")
plt.title(
    "Ground Truth vs. Regularized Model Predictions vs. Linear Model Predictions (Time Period: 700-720)"
)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.grid(True)
plt.show()
