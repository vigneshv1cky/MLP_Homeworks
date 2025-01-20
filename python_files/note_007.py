# -------------------------------------------------
# Imports and Configuration
# -------------------------------------------------

import pandas as pd
import numpy as np
import pickle as pkl
import scipy.stats as stats
import os, re, fnmatch
import pathlib, itertools, time
import matplotlib.pyplot as plt
import joblib
import random

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import explained_variance_score
from sklearn import metrics
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from matplotlib import cm
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Configure default Matplotlib parameters
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["font.size"] = 12
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 14

# -------------------------------------------------
# Load Dataset and Initialize Variables
# -------------------------------------------------

fname = "/Users/vignesh/PycharmProjects/MLP_Homeworks/hw7/Dataset/bmi_dataset_local.pkl"

# Load the BMI dataset
with open(fname, "rb") as f:
    bmi = pkl.load(f)

    MI_folds = bmi["MI"]
    theta_folds = bmi["theta"]
    dtheta_folds = bmi["dtheta"]
    ddtheta_folds = bmi["ddtheta"]
    torque_folds = bmi["torque"]
    time_folds = bmi["time"]

# Number of folds
nfolds = len(MI_folds)
print(f"Number of folds: {nfolds}")

# -------------------------------------------------
# Generate Fold Indices
# -------------------------------------------------

folds_idx = [None] * nfolds
unified_idx = 0

for i, fold in enumerate(time_folds):
    # Create list of indices for each fold
    folds_idx[i] = list(range(unified_idx, unified_idx + fold.shape[0]))
    unified_idx += fold.shape[0]

# -------------------------------------------------
# Concatenate Data from All Folds
# -------------------------------------------------


def concat_folds(folds):
    """Concatenate all folds along the first axis."""
    return np.concatenate(folds, axis=0)


# Full dataset variables
MI = concat_folds(MI_folds)
theta = concat_folds(theta_folds)
dtheta = concat_folds(dtheta_folds)
ddtheta = concat_folds(ddtheta_folds)
torque = concat_folds(torque_folds)
time = concat_folds(time_folds)

# Sizes of the entire dataset
print(
    f"Dataset sizes: MI={MI.shape}, theta={theta.shape}, dtheta={dtheta.shape}, "
    f"ddtheta={ddtheta.shape}, torque={torque.shape}, time={time.shape}"
)

# Starting indices of each fold in the full dataset
start_indices = [folds_idx[i][0] for i in range(nfolds)]
print(f"Starting indices of each fold: {start_indices}")

# -------------------------------------------------
# Standardize MI Data
# -------------------------------------------------

scaler = StandardScaler()
MI_clean = scaler.fit_transform(MI)  # Standardize the MI data

# Verify the scaling
print(f"Mean of standardized MI: {np.mean(MI_clean):.2f}")
print(f"Standard deviation of standardized MI: {np.std(MI_clean):.2f}")


# -------------------------------------------------
# Function Definition: generate_paramsets
# -------------------------------------------------


def generate_paramsets(param_lists):
    """
    Construct the Cartesian product of the parameters
    PARAMS:
        params_lists: dict of lists of values to try for each parameter.
                      keys of the dict are the names of the hyper-parameters.
                      values are lists of possible values to try for the
                      corresponding hyper-parameter
    RETURNS: a list of dicts of hyper-parameter sets.  These make up the
    Cartesian product of the possible hyper-parameters
    """
    keys, values = zip(*param_lists.items())

    # Determines Cartesian product of parameter values
    combos = itertools.product(*values)

    # Constructs list of dictionaries
    combos_dicts = [dict(zip(keys, vals)) for vals in combos]
    return list(combos_dicts)


# -------------------------------------------------
# Function Definition: predict_score_eval
# -------------------------------------------------


def predict_score_eval(model, X, y, convert_deg=False):
    """
    Compute the model predictions and corresponding evaluation metrics.

    Parameters:
        model (object): The trained model used to make predictions.
        X (ndarray): Feature data (M x N).
        y (ndarray): Desired output (M x k).
        convert_deg (bool, optional): If True, converts RMSE from radians to degrees. Default is False.

    Returns:
        dict: A dictionary containing the following keys:
            - "mse" (ndarray): Mean Squared Error for each column (1 x k).
            - "rmse" (ndarray): Root Mean Squared Error for each column (1 x k).
            - "fvaf" (ndarray): Fraction of Variance Accounted For (1 x k).
            - "preds" (ndarray): Predictions made by the model (M x k).
    """
    preds = model.predict(X)

    mse = np.mean(np.square(y - preds), axis=0)
    var = np.var(y, axis=0)
    fvaf = 1 - (mse / var)
    rmse = np.sqrt(mse)

    if convert_deg:
        rmse = rmse * (180 / np.pi)

    results = {
        "mse": np.reshape(mse, (1, -1)),
        "rmse": np.reshape(rmse, (1, -1)),
        "fvaf": np.reshape(fvaf, (1, -1)),
        "preds": preds,
    }

    return results


# -------------------------------------------------
# Function Definition: KFoldHolisticCrossValidation
# -------------------------------------------------


class KFoldHolisticCrossValidation:
    """
    A class to perform cross-validation for a single hyperparameter set.
    """

    def __init__(self, model, eval_func, rotation_skip=1):
        """
        Initialize the cross-validation class.

        :param model: The Scikit-Learn model to be trained
        :param eval_func: A function to evaluate the model. Takes inputs, targets, and predictions.
        :param rotation_skip: Number of rotations to skip. Default is 1.
        """
        self.model = model
        self.eval_func = eval_func
        self.rotation_skip = rotation_skip

    def perform_cross_validation(self, X, y, folds_idx, trainsize):
        """
        Perform cross-validation.

        :param X: Features
        :param y: Targets
        :param folds_idx: List of fold indices
        :param trainsize: Number of folds for training
        :return: results, summary
        """
        nfolds = len(folds_idx)
        if trainsize > nfolds - 2:
            raise ValueError(
                f"trainsize ({trainsize}) cannot be more than nfolds ({nfolds}) - 2"
            )

        results = {"train": None, "val": None, "test": None}
        summary = {"train": {}, "val": {}, "test": {}}

        for rotation in range(0, nfolds, self.rotation_skip):
            Xtrain, ytrain, Xval, yval, Xtest, ytest = self.get_data(
                X, y, folds_idx, nfolds, rotation, trainsize
            )

            print(f"Rotation: {rotation}, Train examples: {ytrain.shape}")
            self.model.fit(Xtrain, ytrain)

            res_train = self.eval_func(self.model, Xtrain, ytrain)
            res_val = self.eval_func(self.model, Xval, yval)
            res_test = self.eval_func(self.model, Xtest, ytest)

            if results["train"] is None:
                results = {"train": res_train, "val": res_val, "test": res_test}
            else:
                for metric in res_train.keys():
                    for key, res in zip(
                        ["train", "val", "test"], [res_train, res_val, res_test]
                    ):
                        results[key][metric] = np.append(
                            results[key][metric], res[metric], axis=0
                        )

        for metric in results["train"].keys():
            for stat_set in ["train", "val", "test"]:
                summary[stat_set][f"{metric}_mean"] = np.mean(
                    results[stat_set][metric], axis=0
                ).reshape(1, -1)
                summary[stat_set][f"{metric}_std"] = np.std(
                    results[stat_set][metric], axis=0
                ).reshape(1, -1)

        return results, summary

    def get_data(self, X, y, folds_idx, nfolds, rotation, trainsize):
        """
        Retrieve data splits for training, validation, and testing.

        :param X: Features
        :param y: Targets
        :param folds_idx: List of fold indices
        :param nfolds: Total number of folds
        :param rotation: Current rotation index
        :param trainsize: Number of training folds
        :return: Xtrain, ytrain, Xval, yval, Xtest, ytest
        """
        trainfolds = (np.arange(trainsize) + rotation) % nfolds
        valfold = (nfolds - 2 + rotation) % nfolds
        testfold = (valfold + 1) % nfolds

        train_idx = [idx for i in trainfolds for idx in folds_idx[i]]
        val_idx = folds_idx[valfold]
        test_idx = folds_idx[testfold]

        return (
            X[train_idx],
            y[train_idx],
            X[val_idx],
            y[val_idx],
            X[test_idx],
            y[test_idx],
        )


# -------------------------------------------------
# Function Definition: CrossValidationGridSearch
# -------------------------------------------------


class CrossValidationGridSearch:
    """
    Class for performing grid search over train sizes and hyperparameter sets.
    """

    def __init__(
        self,
        model,
        paramsets,
        eval_func,
        opt_metric,
        maximize_opt_metric=False,
        trainsizes=[1],
        rotation_skip=1,
        submodel_name=None,
    ):
        """
        Initialize the grid search class.

        :param model: Model to be trained
        :param paramsets: List of hyperparameter sets (dictionaries)
        :param eval_func: Function to evaluate the model
        :param opt_metric: Metric to optimize
        :param maximize_opt_metric: Whether to maximize the metric (default: False)
        :param trainsizes: List of training set sizes (in folds)
        :param rotation_skip: Number of rotations to skip
        :param submodel_name: Name of pipeline submodel for parameter tuning
        """
        self.model = model
        self.paramsets = paramsets
        self.eval_func = eval_func
        self.opt_metric = f"{opt_metric}_mean"
        self.maximize_opt_metric = maximize_opt_metric
        self.trainsizes = trainsizes
        self.rotation_skip = rotation_skip
        self.submodel_name = submodel_name

        self.results = []
        self.results_partial = {}
        self.report_by_size = None
        self.best_param_inds = None
        self.id = random.randint(0, 10000001)

    def load_checkpoint(self, fname):
        """Load checkpoint data."""
        if not os.path.exists(fname):
            raise ValueError(f"File {fname} does not exist")

        with open(fname, "rb") as f:
            self.results = pkl.load(f)
            self.results_partial = pkl.load(f)
            self.id = pkl.load(f)

    def dump_checkpoint(self, fname):
        """Save checkpoint data."""
        with open(fname, "wb") as f:
            pkl.dump(self.results, f)
            pkl.dump(self.results_partial, f)
            pkl.dump(self.id, f)

    def reset_results(self):
        """Reset internal results."""
        self.results = []
        self.results_partial = {}

    def cross_validation_gridsearch(self, X, y, folds_idx, checkpoint_fname=None):
        """
        Perform grid search with cross-validation.

        :param X: Features
        :param y: Targets
        :param folds_idx: List of fold indices
        :param checkpoint_fname: File for saving checkpoints
        """
        cross_val = KFoldHolisticCrossValidation(
            self.model, self.eval_func, rotation_skip=self.rotation_skip
        )

        if checkpoint_fname and os.path.exists(checkpoint_fname):
            self.load_checkpoint(checkpoint_fname)

        for params in self.paramsets:
            if params in [r["params"] for r in self.results]:
                print(f"Already evaluated: {params}")
                continue

            print(f"Evaluating: {params}")

            if self.submodel_name:
                self.model[self.submodel_name].set_params(**params)
            else:
                self.model.set_params(**params)

            param_results = []
            param_summary = None

            for size in self.trainsizes:
                if size in self.results_partial:
                    result = self.results_partial[size]["result"]
                    summary = self.results_partial[size]["summary"]
                else:
                    result, summary = cross_val.perform_cross_validation(
                        X, y, folds_idx, size
                    )
                    self.results_partial[size] = {"result": result, "summary": summary}

                    if checkpoint_fname:
                        self.dump_checkpoint(checkpoint_fname)

                param_results.append(result)
                param_summary = self._merge_summaries(param_summary, summary)

            self.results.append(
                {"params": params, "results": param_results, "summary": param_summary}
            )

            self.results_partial = {}
            if checkpoint_fname:
                self.dump_checkpoint(checkpoint_fname)

    def _merge_summaries(self, summary1, summary2):
        """Helper to merge summaries."""
        if summary1 is None:
            return summary2

        for metric in summary2["train"].keys():
            for stat_set in ["train", "val", "test"]:
                summary1[stat_set][metric] = np.append(
                    summary1[stat_set][metric], summary2[stat_set][metric], axis=0
                )
        return summary1

    def get_reports_all(self):
        """Generate reports and determine best parameters."""
        self.report_by_size = self.get_reports()
        self.best_param_inds = self.get_best_params()
        return {
            "report_by_size": self.report_by_size,
            "best_param_inds": self.best_param_inds,
        }

    def get_reports(self):
        """
        Generate mean validation summary reports for each train size.

        :return: report_by_size as a 3D array (sizes x metrics x params).
        """
        sizes = np.reshape(self.trainsizes, (1, -1))
        nsizes = sizes.shape[1]
        nparams = len(self.results)

        metrics = list(self.results[0]["summary"]["val"].keys())
        report_by_size = np.empty((nsizes, len(metrics) + 2, nparams), dtype=object)

        for p, paramset_result in enumerate(self.results):
            params = paramset_result["params"]
            res_val = paramset_result["summary"]["val"]

            means_by_size = [np.mean(res_val[metric], axis=1) for metric in metrics]
            means_by_size = np.append(sizes, means_by_size, axis=0)

            param_str = np.reshape([str(params)] * nsizes, (1, -1))
            means_by_size = np.append(param_str, means_by_size, axis=0).T

            report_by_size[:, :, p] = means_by_size

        return report_by_size

    def get_best_params(self):
        """
        Determine the best parameters for each train size based on the optimization metric.

        :return: Indices of the best parameter set for each train size.
        """
        report_by_size = self.report_by_size
        metrics = list(self.results[0]["summary"]["val"].keys())

        metric_idx = metrics.index(self.opt_metric)
        report_opt_metric = report_by_size[:, metric_idx + 2, :]

        if self.maximize_opt_metric:
            best_param_inds = np.argmax(report_opt_metric, axis=1)
        else:
            best_param_inds = np.argmin(report_opt_metric, axis=1)

        return best_param_inds

    def get_best_params_strings(self):
        """
        Get strings of the best parameters for each train size.

        :return: List of parameter strings for each size.
        """
        return [str(self.results[p]["params"]) for p in self.best_param_inds]

    def get_report_best_params_for_size(self, size):
        """
        Get validation summary for the best parameters for a specific train size.

        :param size: Index of the train size.
        :return: DataFrame of the best parameter report for the size.
        """
        best_param_inds = self.best_param_inds
        report_by_size = self.report_by_size

        bp_index = best_param_inds[size]
        metrics = list(self.results[0]["summary"]["val"].keys())
        colnames = ["params", "size"] + metrics

        report = pd.DataFrame(
            report_by_size[size].T[bp_index].reshape(1, -1), columns=colnames
        )
        return report

    def plot_cv(self, foldsindices, results, summary, metrics, size):
        """
        Plot train and validation performance for each rotation.

        :param foldsindices: Indices of the train sets
        :param results: Cross-validation results
        :param summary: Summary statistics
        :param metrics: List of metrics to plot
        :param size: Train set size
        :return: Figure and axes handles
        """
        nmetrics = len(metrics)
        fig, axs = plt.subplots(nmetrics, 1, figsize=(12, 6))
        axs = np.array(axs).ravel()

        for metric, ax in zip(metrics, axs):
            res_train = np.mean(results["train"][metric], axis=1)
            res_val = np.mean(results["val"][metric], axis=1)

            ax.plot(foldsindices, res_train, label="train")
            ax.plot(foldsindices, res_val, label="val")

            ax.set(ylabel=metric)
        axs[-1].set(xlabel="Fold Index")
        axs[0].set(title=f"Performance for Train Set Size {size}")
        axs[0].legend(loc="upper right")

        return fig, axs

    def plot_param_train_val(self, metrics, paramidx=0, view_test=False):
        """
        Plot train, validation, and optionally test set performance for a parameter set.

        :param metrics: List of metrics to plot
        :param paramidx: Parameter set index
        :param view_test: Whether to include test performance
        :return: Figure and axes handles
        """
        sizes = self.trainsizes
        summary = self.results[paramidx]["summary"]

        nmetrics = len(metrics)
        fig, axs = plt.subplots(nmetrics, 1, figsize=(12, 6))
        axs = np.array(axs).ravel()

        for metric, ax in zip(metrics, axs):
            res_train = np.mean(summary["train"][metric], axis=1)
            res_val = np.mean(summary["val"][metric], axis=1)

            ax.plot(sizes, res_train, label="train")
            ax.plot(sizes, res_val, label="val")
            if view_test:
                res_test = np.mean(summary["test"][metric], axis=1)
                ax.plot(sizes, res_test, label="test")

            ax.set(ylabel=metric)
            ax.set_xticks(sizes)

        axs[-1].set(xlabel="Train Set Size (# of folds)")
        axs[0].set(title=f"Performance for Parameter Set {paramidx}")
        axs[0].legend(loc="upper right")

        return fig, axs

    def plot_allparams_val(self, metrics):
        """
        Plot validation performance for all parameter sets.

        :param metrics: List of metrics to plot
        :return: Figure and axes handles
        """
        sizes = self.trainsizes
        nmetrics = len(metrics)

        fig, axs = plt.subplots(nmetrics, 1, figsize=(10, 6))
        axs = np.array(axs).ravel()

        for metric, ax in zip(metrics, axs):
            for param_result in self.results:
                res_val = np.mean(param_result["summary"]["val"][metric], axis=1)
                ax.plot(sizes, res_val, label=str(param_result["params"]))

            ax.set(ylabel=metric)
            ax.set_xticks(sizes)

        axs[-1].set(xlabel="Train Set Size (# of folds)")
        axs[0].set(title="Validation Performance")
        axs[0].legend(loc="upper right", bbox_to_anchor=(1.05, 1))

        return fig, axs

    def plot_best_params_by_size(self):
        """
        Plot best parameter performance for each train size.

        :return: Figure and axes handles
        """
        sizes = np.array(self.trainsizes)
        unique_param_sets = np.unique(self.best_param_inds)

        fig, axs = plt.subplots(3, 1, figsize=(10, 8))
        axs = np.array(axs).ravel()

        for i, (set_name, ax) in enumerate(zip(["train", "val", "test"], axs)):
            for p in unique_param_sets:
                param_sizes = sizes[np.where(self.best_param_inds == p)]
                param_sizes.sort()
                param_summary = self.results[p]["summary"][set_name]
                metric_scores = np.mean(
                    param_summary[self.opt_metric][np.where(self.best_param_inds == p)],
                    axis=1,
                )
                ax.scatter(
                    param_sizes, metric_scores, s=120, label=str(self.paramsets[p])
                )

            ax.set_xticks(sizes)
            ax.set(ylabel=self.opt_metric, title=f"{set_name.capitalize()} Performance")

        axs[-1].set(xlabel="Train Set Size (# of folds)")
        axs[0].legend(loc="upper right", bbox_to_anchor=(1.05, 1))

        return fig, axs


# -------------------------------------------------
# Function Definition: plot_surface
# -------------------------------------------------


def plot_surface(
    xlist,
    ylist,
    Z_train,
    Z_val,
    xlabel,
    ylabel,
    zlabel,
    elev=30,
    angle=45,
    title_suffix="",
    xticks=None,
    yticks=None,
    xlog=False,
    ylog=False,
    figsize=(10, 5),
    tick_decimals=None,
):
    """
    Helper plotting function. x-axis is always alpha

    REQUIRES: from mpl_toolkits.mplot3d import Axes3D

    PARAMS:
        xlist: list of x values for the first axis
        ylist: list of y values for the second axis
        Z_train: matrix of performance results from the training set
        Z_val: matrix of performance results from the validation set
        xlabel: x-axis label
        ylabel: y-axis label
        zlabel: z-axis label
        elev: elevation of the 3D plot for the view
        angle: angle in degrees of the 3D plot for the view
        title_suffix: string to append to each subplot title
        xticks: specify x tick locations
        yticks: specify y tick locations
        xlog: Use log scale for x axis
        ylog: Use log scale for y axis
        figsize: Size of the figure
        tick_decimals: If specified, number of digits after the decimal point

    """
    # Initialize figure
    fig = plt.figure(figsize=figsize)

    # Create the X/Y coordinates for the grid
    X, Y = np.meshgrid(xlist, ylist)

    # Use log of X in plot ?
    if xlog:
        X = np.log10(X)
        xticks = np.log10(xticks)

    # Use log of Y in plot ?
    if ylog:
        Y = np.log10(Y)
        yticks = np.log10(yticks)

    # Plot Training and Validation performance
    for i, (Z, set_name) in enumerate(
        zip(
            (Z_train, Z_val),
            ("Training", "Validation"),
        )
    ):
        # Plot the surface
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        surf = ax.plot_surface(
            X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False
        )
        title = "%s Performance %s" % (set_name, title_suffix)
        ax.view_init(elev=elev, azim=angle)
        ax.set(title=title)
        ax.set(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)

        # Set tick marks
        if xticks is not None:
            tmp = xticks

            # Round the ticks if requested
            if tick_decimals is not None:
                tmp = np.round(xticks, decimals=tick_decimals)
            ax.set_xticks(tmp)

        # Round the ticks if requested
        if yticks is not None:
            tmp = yticks
            if tick_decimals is not None:
                tmp = np.round(yticks, decimals=tick_decimals)
            ax.set_yticks(tmp)

    return fig


# -------------------------------------------------
# Function Definition: plot_param_val_surface_RL
# -------------------------------------------------


def plot_param_val_surface_RL(
    crossval, metric, alphas, metric_scale=1, elev=30, angle=245
):
    """
    Plotting function for after cross_validation_gridsearch(),
    displaying the mean (summary) train and val set performances
    for each alpha, for all sizes, for RIDGE and LASSO only

    REQUIRES: from mpl_toolkits.mplot3d import Axes3D

    PARAMS:
        crossval: cross validation object
        metric: summary metric to plot. '_mean' or '_std' must be
                append to the end of the base metric name. These
                base metric names are the keys in the dict returned
                by eval_func
        alphas: list of alpha values
        matric_scale: Scale factor to be applied to the metric values for display
        elev: elevation of the 3D plot for the view
        angle: angle in degrees of the 3D plot for the view

    """
    sizes = crossval.trainsizes
    results = crossval.results
    best_param_inds = crossval.best_param_inds
    nalphas = len(alphas)

    nsizes = len(sizes)

    # Initialize the matrices for the surface
    Z_train = np.empty((nsizes, nalphas))
    Z_val = np.empty((nsizes, nalphas))

    # Obtain the mean performance for the surface
    for param_res in results:
        params = param_res["params"]
        summary = param_res["summary"]

        alpha_idx = alphas.index(params["alpha"])

        # Compute the mean for multiple outputs
        res_train = np.mean(summary["train"][metric], axis=1)
        Z_train[:, alpha_idx] = res_train

        # Compute the mean for multiple outputs
        res_val = np.mean(summary["val"][metric], axis=1)
        Z_val[:, alpha_idx] = res_val

    fig = plot_surface(
        alphas,
        sizes,
        Z_train * metric_scale,
        Z_val * metric_scale,
        "log alpha",
        "size (# of folds)",
        metric,
        elev,
        angle,
        xticks=alphas,
        yticks=sizes,
        xlog=True,
        tick_decimals=1,
    )

    return fig


# -------------------------------------------------
# Initialization and Setup
# -------------------------------------------------

trainsizes = [1, 2, 3, 4, 5, 7, 9]
opt_metric = "fvaf"
maximize_opt_metric = True
skip = 1
predictand = torque

# -------------------------------------------------
# Linear Regression Setup and Grid Search
# -------------------------------------------------

checkpoint_fname_lnr = "hw07_linear_checkpoint.pkl"
model_lnr = LinearRegression()

crossval_lnr = CrossValidationGridSearch(
    model=model_lnr,  # Linear regression model
    paramsets=[{}],  # No hyperparameters to tune for LinearRegression
    eval_func=predict_score_eval,  # Evaluation function
    opt_metric=opt_metric,  # Optimize FVAF
    maximize_opt_metric=maximize_opt_metric,  # Maximize FVAF
    trainsizes=trainsizes,  # Training sizes
    rotation_skip=skip,  # Rotation skip value
)

force = False
# force = True

if force and os.path.exists(checkpoint_fname_lnr):
    os.remove(checkpoint_fname_lnr)  # Delete checkpoint if force is True

# Perform grid search for Linear Regression
crossval_lnr.cross_validation_gridsearch(
    X=MI_clean,  # Feature data
    y=predictand,  # Target variable
    folds_idx=folds_idx,  # Fold indices for cross-validation
)

# -------------------------------------------------
# Ridge Regression Setup and Grid Search
# -------------------------------------------------

checkpoint_fname_ridge = "hw07_ridge_checkpoint.pkl"

# Parameters for Ridge Regression
alphas = np.logspace(0, 4, base=10, num=6, endpoint=True)  # Alpha values
max_iter = 10000  # Maximum iterations
tol = 0.001  # Tolerance

model_ridge = Ridge(alpha=alphas[0], max_iter=max_iter, tol=tol)

param_lists_ridge = {
    "alpha": alphas,
    "max_iter": [max_iter],
    "tol": [tol],
}

allparamsets_ridge = generate_paramsets(param_lists_ridge)

crossval_ridge = CrossValidationGridSearch(
    model=model_ridge,
    paramsets=allparamsets_ridge,
    eval_func=predict_score_eval,
    opt_metric=opt_metric,
    maximize_opt_metric=maximize_opt_metric,
    trainsizes=trainsizes,
    rotation_skip=skip,
)

force = False
# force = True

if force and os.path.exists(checkpoint_fname_ridge):
    os.remove(checkpoint_fname_ridge)

# Perform grid search for Ridge Regression
crossval_ridge.cross_validation_gridsearch(
    X=MI_clean,
    y=predictand,
    folds_idx=folds_idx,
)

# -------------------------------------------------
# Lasso Regression Setup and Grid Search
# -------------------------------------------------

checkpoint_fname_lasso = "hw07_lasso_checkpoint.pkl"

# Parameters for Lasso Regression
lasso_alphas = np.logspace(-6, -1, base=10, num=6, endpoint=True)
max_iter = 10000
tol = 0.001

model_lasso = Lasso(alpha=lasso_alphas[0], max_iter=max_iter, tol=tol)

param_lists_lasso = {
    "alpha": lasso_alphas,
    "max_iter": [max_iter],
    "tol": [tol],
}

allparamsets_lasso = generate_paramsets(param_lists_lasso)

crossval_lasso = CrossValidationGridSearch(
    model=model_lasso,
    paramsets=allparamsets_lasso,
    eval_func=predict_score_eval,
    opt_metric=opt_metric,
    maximize_opt_metric=maximize_opt_metric,
    trainsizes=trainsizes,
    rotation_skip=skip,
)

force = True

if force and os.path.exists(checkpoint_fname_lasso):
    os.remove(checkpoint_fname_lasso)

# Perform grid search for Lasso Regression
crossval_lasso.cross_validation_gridsearch(
    X=MI_clean,
    y=predictand,
    folds_idx=folds_idx,
)

# -------------------------------------------------
# Reporting and Visualization
# -------------------------------------------------

crossval_report_lnr = crossval_lnr.get_reports_all()
crossval_report_ridge = crossval_ridge.get_reports_all()
crossval_report_lasso = crossval_lasso.get_reports_all()

# Ridge Regression Surface Plot
elev = 30
angle = 300
plot_param_val_surface_RL(
    crossval_ridge,
    crossval_ridge.opt_metric,
    list(alphas),
    elev=elev,
    angle=angle,
)
plt.show()

# Lasso Regression Surface Plot
elev = 30
angle = 285
plot_param_val_surface_RL(
    crossval_lasso,
    crossval_lasso.opt_metric,
    list(lasso_alphas),
    elev=elev,
    angle=angle,
)
plt.show()

# Best Parameters by Size
crossval_lnr.plot_best_params_by_size()
plt.show()

crossval_ridge.plot_best_params_by_size()
plt.show()

crossval_lasso.plot_best_params_by_size()
plt.show()

# -------------------------------------------------
# Function Definition: extract_test_stats
# -------------------------------------------------


def extract_test_stats(cv_list, metric, size_idx):
    """
    :param cv_list: List of cross-validation grid search instances
    :param metric: Name of the metric that we are fetching from the report
    :param size_idx: Index of the training set size in the trainsizes variable.

    :return: List of test set performance structures.  One structure is
            returned for each element in the cv_list
    """

    out = []

    for cv in cv_list:
        # Fetch the results
        all_results = cv.results

        # Fetch best parameters
        best_params_idx = cv.best_param_inds[size_idx]

        # Test set performance for training size size_idx
        test_perf = all_results[best_params_idx]["results"][size_idx]["test"][metric]

        # Test set performance average across both shoulder and elbow
        test_perf_avg = np.mean(test_perf, axis=1)

        out.append(test_perf_avg)

    return out


# -------------------------------------------------
# Extract Test Statistics
# -------------------------------------------------

metric = "fvaf"
training_size_idx = 0

test_lnr, test_ridge, test_lasso = extract_test_stats(
    cv_list=[
        crossval_lnr,
        crossval_ridge,
        crossval_lasso,
    ],  # List of cross-validation objects
    metric=metric,  # Metric to extract (e.g., fvaf)
    size_idx=training_size_idx,  # Training size index
)

# Display the extracted test statistics
print(f"Linear Regression Test Stats (N=20): {test_lnr}")
print(f"Ridge Regression Test Stats (N=20): {test_ridge}")
print(f"Lasso Regression Test Stats (N=20): {test_lasso}")

# -------------------------------------------------
# Statistical Analysis: Linear vs. Lasso Regression
# -------------------------------------------------

from scipy.stats import ttest_rel

# Compute the pairwise differences
differences = test_lnr - test_lasso

# Perform the paired t-test
t_stat, p_value = ttest_rel(test_lnr, test_lasso)

# Compute the mean of the pairwise differences
mean_diff = np.mean(differences)

# Display the results
print("\n--- Linear vs. Lasso Regression ---")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4e}")
print(f"Mean of the Pairwise Differences: {mean_diff:.4f}")

# Check if we reject the null hypothesis at 95% confidence
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference.")
else:
    print("Fail to reject the null hypothesis: No significant difference.")

# -------------------------------------------------
# Statistical Analysis: Linear vs. Ridge Regression
# -------------------------------------------------

# Compute the pairwise differences
differences = test_lnr - test_ridge

# Perform the paired t-test
t_stat, p_value = ttest_rel(test_lnr, test_ridge)

# Compute the mean of the pairwise differences
mean_diff = np.mean(differences)

# Display the results
print("\n--- Linear vs. Ridge Regression ---")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4e}")
print(f"Mean of the Pairwise Differences: {mean_diff:.4f}")

# Check if we reject the null hypothesis at 95% confidence
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: There is a significant difference.")
else:
    print("Fail to reject the null hypothesis: No significant difference.")
