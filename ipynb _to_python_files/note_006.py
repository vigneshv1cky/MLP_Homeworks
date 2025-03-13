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
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["font.size"] = 12
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 14


fname = "/Users/vignesh/PycharmProjects/MLP_Homeworks/hw6/Dataset/bmi_dataset_local.pkl"

with open(fname, "rb") as f:
    bmi = pkl.load(f)

    MI_folds = bmi["MI"]
    theta_folds = bmi["theta"]
    dtheta_folds = bmi["dtheta"]
    ddtheta_folds = bmi["ddtheta"]
    torque_folds = bmi["torque"]
    time_folds = bmi["time"]


nfolds = len(MI_folds)
nfolds


folds_idx = [None] * nfolds

unified_idx = 0
for i, fold in enumerate(time_folds):
    folds_idx[i] = list(range(unified_idx, unified_idx + fold.shape[0]))
    unified_idx += fold.shape[0]


def concat_folds(folds):
    return np.concatenate(folds, axis=0)


MI = concat_folds(MI_folds)
theta = concat_folds(theta_folds)
dtheta = concat_folds(dtheta_folds)
ddtheta = concat_folds(ddtheta_folds)
torque = concat_folds(torque_folds)
time = concat_folds(time_folds)

print(MI.shape, theta.shape, dtheta.shape, ddtheta.shape, torque.shape, time.shape)

[folds_idx[i][0] for i in range(nfolds)]


def generate_paramsets(param_lists):
    keys, values = zip(*param_lists.items())
    combos = itertools.product(*values)
    combos_dicts = [dict(zip(keys, vals)) for vals in combos]
    return list(combos_dicts)


def predict_score_eval(model, X, y, convert_deg=False):
    preds = model.predict(X)

    mse = np.sum(np.square(y - preds), axis=0) / y.shape[0]
    var = np.var(y, axis=0)

    fvaf = 1 - mse / var

    rmse = np.sqrt(mse)

    if convert_deg:
        rmse = rmse * 180 / np.pi

    results = {
        "mse": np.reshape(mse, (1, -1)),
        "rmse": np.reshape(rmse, (1, -1)),
        "fvaf": np.reshape(fvaf, (1, -1)),
    }

    return results


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


param_lists = {
    "alpha": [10**i for i in range(7)],  # Factors of 10 from 1 to 1,000,000
    "max_iter": [10000],  # Fixed max_iter value
}


allparamsets = generate_paramsets(param_lists)
model = Pipeline(
    [
        ("scaler", StandardScaler()),  # Step 1: Standardize features
        ("regression", Ridge()),  # Step 2: Ridge Regression, named 'regression'
    ]
)


train_sizes = [1, 2, 3, 4, 5, 7, 10, 13, 16]
opt_metric = "rmse"

crossval = CrossValidationGridSearch(
    model=model,
    paramsets=allparamsets,  # All combinations of parameters (alpha, max_iter)
    eval_func=predict_score_eval,  # Evaluation function that returns rmse, mse, etc.
    opt_metric=opt_metric,  # Optimization metric: rmse
    maximize_opt_metric=False,  # We want to minimize RMSE, so set this to False
    trainsizes=train_sizes,  # Training set sizes
    rotation_skip=1,  # Use all folds for training/validation
    submodel_name="regression",  # Target the 'regression' step in the pipeline
)

fullcvfname = "hw6_crossval_ridge_checkpoint.pkl"

force = True
if force and os.path.exists(fullcvfname):
    # Delete the checkpoint file
    print(f"Deleting checkpoint file {fullcvfname} to force a fresh run.")
    os.remove(fullcvfname)

X = np.hstack([MI, theta, dtheta, ddtheta])  # Input features
y = torque  # Target output

crossval.cross_validation_gridsearch(
    X=X,  # The input data (features)
    y=y,  # The target data (labels)
    folds_idx=folds_idx,  # The fold indices for cross-validation
    checkpoint_fname=fullcvfname,  # Checkpoint file to store results
)


crossval_report = crossval.get_reports_all()
crossval_report.keys()
crossval.results[0]["summary"]["val"]
all_results = crossval.results
len(all_results)

best_param_inds = crossval_report["best_param_inds"]
best_param_inds

best_param_sets = crossval.get_best_params_strings()
print(best_param_sets)


metrics_to_plot = ["rmse_mean"]  # This specifies that we want to plot the RMSE mean
crossval.plot_allparams_val(metrics=metrics_to_plot)
plt.show()

crossval.plot_best_params_by_size()
plt.show()
