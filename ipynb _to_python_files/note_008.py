# -------------------------------------------------
# Imports and Configuration
# -------------------------------------------------

import pandas as pd
import numpy as np
import scipy.stats as stats
import os, re, fnmatch
import pathlib, itertools
import time as timelib
import matplotlib.pyplot as plt
import pickle as pkl

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import roc_curve, auc, f1_score, recall_score
from sklearn.svm import SVC
import joblib
import pdb
import itertools

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Default figure parameters
plt.rcParams["figure.figsize"] = (6, 5)
plt.rcParams["font.size"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["figure.constrained_layout.use"] = False
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

# -------------------------------------------------
# Load Dataset and Initialize Variables
# -------------------------------------------------

fname = "/Users/vignesh/PycharmProjects/MLP_Homeworks/HW8/Dataset/hw08_skel.pkl"

with open(fname, "rb") as f:
    dat = pkl.load(f)

ins_train = dat["ins_train"]
outs_train = dat["outs_train"]
ins_val = dat["ins_val"]
outs_val = dat["outs_val"]


# -------------------------------------------------
# Function Definition: confusion_mtx_colormap
# -------------------------------------------------


# Generate a color map plot for a confusion matrix
def confusion_mtx_colormap(mtx, xnames, ynames, cbarlabel=""):
    """
    Generate a figure that plots a colormap of a matrix
    PARAMS:
        mtx: matrix of values
        xnames: list of x tick names
        ynames: list of the y tick names
        cbarlabel: label for the color bar
    RETURNS:
        fig, ax: the corresponding handles for the figure and axis
    """
    nxvars = mtx.shape[1]
    nyvars = mtx.shape[0]

    # create the figure and plot the correlation matrix
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(mtx, cmap="summer")
    if not cbarlabel == "":
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Specify the row and column ticks and labels for the figure
    ax.set_xticks(range(nxvars))
    ax.set_yticks(range(nyvars))
    ax.set_xticklabels(xnames)
    ax.set_yticklabels(ynames)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("Actual Labels")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    lbl = np.array([["TN", "FP"], ["FN", "TP"]])
    for i in range(nyvars):
        for j in range(nxvars):
            text = ax.text(
                j,
                i,
                "%s = %d" % (lbl[i, j], mtx[i, j]),
                ha="center",
                va="center",
                color="k",
            )

    return fig, ax


# -------------------------------------------------
# Function Definition: ks_roc_plot
# -------------------------------------------------


# Compute the ROC Curve and generate the KS plot
def ks_roc_plot(targets, scores, FIGWIDTH=10, FIGHEIGHT=4, FONTSIZE=14):
    """
    Generate a figure that plots the ROC Curve and the distributions
    of the TPR and FPR over a set of thresholds.
    PARAMS:
        targets: list of true target labels
        scores: list of prediction scores
    """
    # Compute ROC
    fpr, tpr, thresholds = roc_curve(targets, scores)
    auc_roc = auc(fpr, tpr)

    # Compute positve fraction
    pos = np.where(targets)[0]
    npos = len(pos)
    pos_frac = npos / targets.shape[0]

    # Generate KS plot
    fig, ax = plt.subplots(1, 2, figsize=(FIGWIDTH, FIGHEIGHT))
    axs = ax.ravel()

    ax[0].plot(thresholds, tpr, color="b")
    ax[0].plot(thresholds, fpr, color="r")
    ax[0].plot(thresholds, tpr - fpr, color="g")
    ax[0].set_xlim([0, 1])
    ax[0].invert_xaxis()
    ax[0].set(xlabel="threshold", ylabel="fraction")
    ax[0].legend(["TPR", "FPR", "Difference"], fontsize=FONTSIZE)

    # Generate ROC Curve plot
    ax[1].plot(fpr, tpr, color="b")
    ax[1].plot([0, 1], [0, 1], "r--")
    ax[1].set(xlabel="FPR", ylabel="TPR")
    ax[1].set_aspect("equal", "box")
    auc_text = ax[1].text(
        0.05, 0.95, "AUC = %.4f" % auc_roc, color="k", fontsize=FONTSIZE
    )


# -------------------------------------------------
# Imports
# -------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# Initial SVM Model and Accuracy Calculation - Linear Kernel
# ---------------------------------------------------------------------------------------------------------------------------------------------------

# Define and fit the SVM model
model = SVC(kernel="linear", C=1.0, probability=True)
model.fit(ins_train, outs_train)

# Predict labels for training and validation sets
preds_train = model.predict(ins_train)
pred_val = model.predict(ins_val)

# Calculate and print accuracy for both sets
train_accuracy = accuracy_score(outs_train, preds_train)
val_accuracy = accuracy_score(outs_val, pred_val)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# -------------------------------------------------
# Hyperparameter Tuning for C
# -------------------------------------------------

# Initialize variables for tracking the best C and accuracies
best_C = None
best_accuracy = 0
C_values = []
train_accuracies = []
val_accuracies = []

# Iterate over multiple C values to find the optimal one
for C_value in np.arange(0.001, 0.1, 0.001):
    model = SVC(kernel="linear", C=C_value, probability=True)
    model.fit(ins_train, outs_train)

    # Predict on training and validation sets
    preds_train = model.predict(ins_train)
    preds_val = model.predict(ins_val)

    # Calculate accuracies
    train_accuracy = accuracy_score(outs_train, preds_train)
    val_accuracy = accuracy_score(outs_val, pred_val)

    # Store results
    C_values.append(C_value)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    # Update best C if the current validation accuracy is better
    if val_accuracy > best_accuracy:
        best_C = C_value
        best_accuracy = val_accuracy

print(f"Best C value: {best_C}, with Validation Accuracy: {best_accuracy:.4f}")

# Plot training and validation accuracies
plt.plot(C_values, train_accuracies, label="Training Accuracy")
plt.plot(C_values, val_accuracies, label="Validation Accuracy")
plt.xlabel("C value")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy for Different C Values")
plt.legend()
plt.show()

# -------------------------------------------------
# Evaluate Final Model with Best C
# -------------------------------------------------

model = SVC(kernel="linear", C=best_C, probability=True)
model.fit(ins_train, outs_train)

# Predict on training and validation sets
preds_train = model.predict(ins_train)
pred_val = model.predict(ins_val)

# Calculate and print accuracies
train_accuracy = accuracy_score(outs_train, preds_train)
val_accuracy = accuracy_score(outs_val, pred_val)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# -------------------------------------------------
# Probability and Confusion Matrix Analysis
# -------------------------------------------------

# Predict probabilities for training and validation sets
train_probabilities = model.predict_proba(ins_train)
val_probabilities = model.predict_proba(ins_val)

targetnames = ["Negative", "Positive"]

# Training set confusion matrix and visualization
confusion_mtx = confusion_matrix(outs_train, preds_train)
confusion_mtx_colormap(
    confusion_mtx, targetnames, targetnames, "Training Confusion Matrix"
)
plt.show()

plt.hist(train_probabilities[:, 1], bins=30, edgecolor="black", alpha=0.7)
plt.xlabel("Probability of Class 1")
plt.ylabel("Frequency")
plt.title("Histogram of Class 1 Probabilities (Training Set)")
plt.show()

ks_roc_plot(outs_train, train_probabilities[:, 1])
plt.suptitle("TPR/FPR and ROC Curve for Training Set")
plt.show()

# Validation set confusion matrix and visualization
val_confusion_mtx = confusion_matrix(outs_val, preds_val)
confusion_mtx_colormap(
    val_confusion_mtx, targetnames, targetnames, "Validation Confusion Matrix"
)
plt.show()

plt.hist(val_probabilities[:, 1], bins=20, edgecolor="black", alpha=0.7)
plt.xlabel("Probability of Class 1")
plt.ylabel("Frequency")
plt.title("Histogram of Class 1 Probabilities (Validation Set)")
plt.show()

ks_roc_plot(outs_val, val_probabilities[:, 1])
plt.suptitle("TPR/FPR and ROC Curve for Validation Set")
plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# Initial SVM Model and Accuracy Calculation - Polynomial Kernel
# ---------------------------------------------------------------------------------------------------------------------------------------------------

# Define and fit the initial SVM model
model = SVC(kernel="poly", C=1.0, probability=True)
model.fit(ins_train, outs_train)

# Predict labels for training and validation sets
preds_train = model.predict(ins_train)
pred_val = model.predict(ins_val)

# Calculate and print accuracy for both sets
train_accuracy = accuracy_score(outs_train, preds_train)
val_accuracy = accuracy_score(outs_val, pred_val)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# -------------------------------------------------
# Hyperparameter Tuning for C
# -------------------------------------------------

# Initialize variables to track the best C and corresponding accuracies
best_C = None
best_accuracy = 0
C_values = []
train_accuracies = []
val_accuracies = []

# Iterate over multiple C values to tune the regularization parameter
for C_value in np.arange(0.001, 4, 0.05):
    model = SVC(kernel="poly", C=C_value, probability=True)
    model.fit(ins_train, outs_train)

    # Predict on training and validation sets
    preds_train = model.predict(ins_train)
    pred_val = model.predict(ins_val)

    # Compute accuracy for both sets
    train_accuracy = accuracy_score(outs_train, preds_train)
    val_accuracy = accuracy_score(outs_val, pred_val)

    # Store results
    C_values.append(C_value)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    # Update best C if validation accuracy improves
    if val_accuracy > best_accuracy:
        best_C = C_value
        best_accuracy = val_accuracy

print(f"Best C value: {best_C}, with Validation Accuracy: {best_accuracy:.4f}")

# Plot training and validation accuracy for each C value
plt.plot(C_values, train_accuracies, label="Training Accuracy")
plt.plot(C_values, val_accuracies, label="Validation Accuracy")
plt.xlabel("C value")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy for Different C Values")
plt.legend()
plt.show()

# -------------------------------------------------
# Evaluate Final Model with Best C
# -------------------------------------------------

# Define and fit the final SVM model with the best C value
model = SVC(kernel="poly", C=best_C, probability=True)
model.fit(ins_train, outs_train)

# Predict on training and validation sets
preds_train = model.predict(ins_train)
pred_val = model.predict(ins_val)

# Calculate and print accuracy for both sets
train_accuracy = accuracy_score(outs_train, preds_train)
val_accuracy = accuracy_score(outs_val, pred_val)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# -------------------------------------------------
# Confusion Matrix and Probability Analysis
# -------------------------------------------------

# Predict probabilities for training and validation sets
train_probabilities = model.predict_proba(ins_train)
val_probabilities = model.predict_proba(ins_val)

# Target class labels
targetnames = ["Negative", "Positive"]

# Training set confusion matrix and visualization
confusion_mtx = confusion_matrix(outs_train, preds_train)
confusion_mtx_colormap(
    confusion_mtx, targetnames, targetnames, "Training Confusion Matrix"
)
plt.show()

# Histogram of class 1 probabilities for training set
plt.hist(train_probabilities[:, 1], bins=30, edgecolor="black", alpha=0.7)
plt.xlabel("Probability of Class 1")
plt.ylabel("Frequency")
plt.title("Histogram of Class 1 Probabilities (Training Set)")
plt.show()

# TPR/FPR and ROC Curve for the training set
ks_roc_plot(outs_train, train_probabilities[:, 1])
plt.suptitle("TPR/FPR and ROC Curve for Training Set")
plt.show()

# Validation set confusion matrix and visualization
val_confusion_mtx = confusion_matrix(outs_val, preds_val)
confusion_mtx_colormap(
    val_confusion_mtx, targetnames, targetnames, "Validation Confusion Matrix"
)
plt.show()

# Histogram of class 1 probabilities for validation set
plt.hist(val_probabilities[:, 1], bins=20, edgecolor="black", alpha=0.7)
plt.xlabel("Probability of Class 1")
plt.ylabel("Frequency")
plt.title("Histogram of Class 1 Probabilities (Validation Set)")
plt.show()

# TPR/FPR and ROC Curve for the validation set
ks_roc_plot(outs_val, val_probabilities[:, 1])
plt.suptitle("TPR/FPR and ROC Curve for Validation Set")
plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# Initial SVM Model and Accuracy Calculation - Radial Bias Kernel
# ---------------------------------------------------------------------------------------------------------------------------------------------------

# Define and fit the initial SVM model with RBF kernel
model = SVC(kernel="rbf", C=1.0, probability=True)
model.fit(ins_train, outs_train)

# Predict labels for training and validation sets
preds_train = model.predict(ins_train)
pred_val = model.predict(ins_val)

# Calculate and print accuracy for both sets
train_accuracy = accuracy_score(outs_train, preds_train)
val_accuracy = accuracy_score(outs_val, pred_val)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# -------------------------------------------------
# Hyperparameter Tuning for C
# -------------------------------------------------

# Initialize variables for tracking the best C and corresponding accuracies
best_C = None
best_accuracy = 0
C_values = []
train_accuracies = []
val_accuracies = []

# Iterate over multiple C values to find the optimal regularization parameter
for C_value in np.arange(0.001, 4, 0.01):
    model = SVC(kernel="rbf", C=C_value, probability=True)
    model.fit(ins_train, outs_train)

    # Predict on training and validation sets
    preds_train = model.predict(ins_train)
    pred_val = model.predict(ins_val)

    # Compute accuracies for both sets
    train_accuracy = accuracy_score(outs_train, preds_train)
    val_accuracy = accuracy_score(outs_val, pred_val)

    # Store the C value and corresponding accuracies
    C_values.append(C_value)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    # Update best C if validation accuracy improves
    if val_accuracy > best_accuracy:
        best_C = C_value
        best_accuracy = val_accuracy

print(f"Best C value: {best_C}, with Validation Accuracy: {best_accuracy:.4f}")

# Plot training and validation accuracy for each C value
plt.plot(C_values, train_accuracies, label="Training Accuracy")
plt.plot(C_values, val_accuracies, label="Validation Accuracy")
plt.xlabel("C value")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy for Different C Values")
plt.legend()
plt.show()

# -------------------------------------------------
# Evaluate Final Model with Best C
# -------------------------------------------------

# Define and fit the final SVM model with the best C value
model = SVC(kernel="rbf", C=best_C, probability=True)
model.fit(ins_train, outs_train)

# Predict on training and validation sets
preds_train = model.predict(ins_train)
pred_val = model.predict(ins_val)

# Calculate and print accuracies
train_accuracy = accuracy_score(outs_train, preds_train)
val_accuracy = accuracy_score(outs_val, pred_val)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

# -------------------------------------------------
# Confusion Matrix and Probability Analysis
# -------------------------------------------------

# Predict probabilities for training and validation sets
train_probabilities = model.predict_proba(ins_train)
val_probabilities = model.predict_proba(ins_val)

# Define target class names
targetnames = ["Negative", "Positive"]

# Training set confusion matrix
confusion_mtx = confusion_matrix(outs_train, preds_train)
confusion_mtx_colormap(
    confusion_mtx, targetnames, targetnames, "Training Confusion Matrix"
)
plt.show()

# Histogram of class 1 probabilities for training set
plt.hist(train_probabilities[:, 1], bins=30, edgecolor="black", alpha=0.7)
plt.xlabel("Probability of Class 1")
plt.ylabel("Frequency")
plt.title("Histogram of Class 1 Probabilities (Training Set)")
plt.show()

# TPR/FPR and ROC Curve for the training set
ks_roc_plot(outs_train, train_probabilities[:, 1])
plt.suptitle("TPR/FPR and ROC Curve for Training Set")
plt.show()

# Validation set confusion matrix
val_confusion_mtx = confusion_matrix(outs_val, preds_val)
confusion_mtx_colormap(
    val_confusion_mtx, targetnames, targetnames, "Validation Confusion Matrix"
)
plt.show()

# Histogram of class 1 probabilities for validation set
plt.hist(val_probabilities[:, 1], bins=20, edgecolor="black", alpha=0.7)
plt.xlabel("Probability of Class 1")
plt.ylabel("Frequency")
plt.title("Histogram of Class 1 Probabilities (Validation Set)")
plt.show()

# TPR/FPR and ROC Curve for the validation set
ks_roc_plot(outs_val, val_probabilities[:, 1])
plt.suptitle("TPR/FPR and ROC Curve for Validation Set")
plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------------------------
