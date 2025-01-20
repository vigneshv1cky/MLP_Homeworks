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


"""
TODO: Complete K-Fold Cross-Validation implementation

"""


class KFoldHolisticCrossValidation:
    """
    Cross-validation class. This class will perform cross-validation across for a
    single hyper-parameter set.
    """

    def __init__(self, model, eval_func, rotation_skip=1):
        """
        :param model: The Scikit-Learn model to be trained
        :param eval_func': Python function that will be used to evaluate a model
                                parameters: (inputs, desired outputs, model predictions)
        :param rotation_skip: Number of CV rotations for every one rotation that is actually trained & evaluated.
                                Typical is 1 (train and evaluate all rotations), but when we are
                                debugging, it is helpful to perform a smaller number of train/evaluate
                                cycles.
        """
        self.model = model
        self.eval_func = eval_func
        self.rotation_skip = rotation_skip

    def perform_cross_validation(self, X, y, folds_idx, trainsize):
        """TODO

        Documentation is given above
        """

        nfolds = len(folds_idx)
        if trainsize > nfolds - 2:
            err_msg = (
                "ERROR: KFoldHolisticCrossValidation.perform_cross_validation() - "
            )
            err_msg += "trainsize (%d) cant be more than nfolds (%d) - 2" % (
                trainsize,
                nfolds,
            )
            raise ValueError(err_msg)

        results = {"train": None, "val": None, "test": None}
        summary = {"train": {}, "val": {}, "test": {}}

        model = self.model
        evaluate = self.eval_func

        for rotation in range(0, nfolds, self.rotation_skip):
            (Xtrain, ytrain, Xval, yval, Xtest, ytest) = self.get_data(
                X, y, folds_idx, nfolds, rotation, trainsize
            )

            print("Rotation:", rotation, "; train examples:", ytrain.shape)

            model.fit(Xtrain, ytrain)

            res_train = evaluate(model, Xtrain, ytrain)
            res_val = evaluate(model, Xval, yval)
            res_test = evaluate(model, Xtest, ytest)

            if results["train"] is None:
                results["train"] = res_train
                results["val"] = res_val
                results["test"] = res_test
            else:
                for metric in res_train.keys():
                    results["train"][metric] = np.append(
                        results["train"][metric], res_train[metric], axis=0
                    )
                    results["val"][metric] = np.append(
                        results["val"][metric], res_val[metric], axis=0
                    )
                    results["test"][metric] = np.append(
                        results["test"][metric], res_test[metric], axis=0
                    )

        for metric in results["train"].keys():
            for stat_set in ["train", "val", "test"]:
                summary[stat_set][metric + "_mean"] = np.mean(
                    results[stat_set][metric], axis=0
                ).reshape(1, -1)
                summary[stat_set][metric + "_std"] = np.std(
                    results[stat_set][metric], axis=0
                ).reshape(1, -1)

        return results, summary

    def get_data(self, X, y, folds_idx, nfolds, rotation, trainsize):
        trainfolds = (np.arange(trainsize) + rotation) % nfolds

        valfold = (nfolds - 2 + rotation) % nfolds

        testfold = (valfold + 1) % nfolds

        train_idx = []
        for i in trainfolds:
            train_idx += folds_idx[i]

        Xtrain = X[train_idx]
        ytrain = y[train_idx]

        val_idx = folds_idx[valfold]
        Xval = X[val_idx]
        yval = y[val_idx]

        test_idx = folds_idx[testfold]
        Xtest = X[test_idx]
        ytest = y[test_idx]

        return Xtrain, ytrain, Xval, yval, Xtest, ytest


class CrossValidationGridSearch:
    """
    This class is responsible for performing a grid trainsizes x paramsets CV experiments.
    For each grid point, N-fold crossvalidation is performed (with potential skips in the
    possible rotations).

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
        """TODO
        Class instance constructor

        :param model: Model to be trained
        :param paramsets: List of dicts.  Every dict contains a set of hyper-parameters for use in
                            one experiment
        :param eval_func: Python function that will be used to evaluate a model
                                parameters: (inputs, desired outputs, model predictions)
        :param opt_metric: Optimization metric to be used.  Must be included in the
                          list of metrics returned by eval_func
        :param maximize_opt_metric: True -> best model has high value for performance metric;
                                    False -> best model has low value
        :param trainsizes: A list of training set sizes (in terms of number of folds)
        :param rotation_skip: Number of CV rotations for every one rotation to train & evaluate.
                                Typical is 1 (train and evaluate all rotations), but when we are
                                debugging, it is helpful to perform a smaller number of train/evaluate
                                cycles
        :param submodel_name: If the model is a pipeline, then this is the name of the pipeline
                                component that will make use of the hyper-parameters
        """
        self.model = model
        self.paramsets = paramsets
        self.trainsizes = trainsizes
        self.eval_func = eval_func
        self.opt_metric = opt_metric + "_mean"
        self.maximize_opt_metric = maximize_opt_metric
        self.rotation_skip = rotation_skip
        self.submodel_name = submodel_name

        self.results = []
        self.results_partial = {}

        self.report_by_size = None

        self.best_param_inds = None
        self.id = random.randint(0, 10000001)

    def load_checkpoint(self, fname):
        """PROVIDED
        Load a checkpoint file into self.results

        :param fname: Full name of the file to load the checkpoint from.
        """
        if not os.path.exists(fname):
            raise ValueError("File %s does not exist" % fname)

        with open(fname, "rb") as f:
            self.results = pkl.load(f)
            self.results_partial = pkl.load(f)
            self.id = pkl.load(f)

    def dump_checkpoint(self, fname):
        """PROVIDED
        Write the current set of results to a checkpoint file

        :param fname: Full name of file to write checkpoint to
        """
        with open(fname, "wb") as f:
            pkl.dump(self.results, f)
            pkl.dump(self.results_partial, f)
            pkl.dump(self.id, f)

    def reset_results(self):
        """PROVIDED
        Reset the current set of results that are stored internally
        """
        self.results = []
        self.results_partial = {}

    def cross_validation_gridsearch(self, X, y, folds_idx, checkpoint_fname=None):
        """TODO

        Documentation given above.
        """

        cross_val = KFoldHolisticCrossValidation(
            self.model, self.eval_func, rotation_skip=self.rotation_skip
        )

        if checkpoint_fname is not None and os.path.exists(checkpoint_fname):
            self.load_checkpoint(checkpoint_fname)

        for params in self.paramsets:
            if params in [r["params"] for r in self.results]:
                print("already evaled:", params)
                continue

            print("evaling on:", params)

            param_results = []
            param_summary = None

            if self.submodel_name is not None:
                self.model[self.submodel_name].set_params(**params)
            else:
                self.model.set_params(**params)

            for size in self.trainsizes:
                if size in self.results_partial.keys():
                    print("%d cached" % (size))

                    result = (self.results_partial[size]["result"],)
                    summary = self.results_partial[size]["summary"]
                else:
                    print("Executing size %d" % (size))

                    result, summary = cross_val.perform_cross_validation(
                        X, y, folds_idx, size
                    )

                    self.results_partial[size] = {"result": result, "summary": summary}

                    if checkpoint_fname is not None:
                        self.dump_checkpoint(checkpoint_fname)

                param_results.append(result)

                if param_summary is None:
                    param_summary = summary
                else:
                    for metric in summary["train"].keys():
                        for stat_set in ["train", "val", "test"]:
                            param_summary[stat_set][metric] = np.append(
                                param_summary[stat_set][metric],
                                summary[stat_set][metric],
                                axis=0,
                            )

            self.results.append(
                {"params": params, "results": param_results, "summary": param_summary}
            )

            self.results_partial = {}

            if checkpoint_fname is not None:
                self.dump_checkpoint(checkpoint_fname)

    def get_reports_all(self):
        """PROVIDED
        Generate reports on the internally stored results

        :return: Dictionary containing two keys: 'report_by_size', 'best_param_inds'
        """
        self.report_by_size = self.get_reports()
        self.best_param_inds = self.get_best_params(
            self.opt_metric, self.maximize_opt_metric
        )
        print("CS:", self.id)

        return {
            "report_by_size": self.report_by_size,
            "best_param_inds": self.best_param_inds,
        }

    """ PROVIDED
    Functions to generate a report of the result of the cross-validation r5un
    """

    def get_reports(self):
        """PROVIDED
        Get the mean validation summary of all the parameters for each size
        for all metrics. This is used to determine the best parameter set
        for each size

        RETURNS: the report_by_size as a 3D s x r x p array. Where s is
                 the number of train sizes tried, r is the number of summary
                 metrics evaluated+2, and p is the number of parameter sets.
        """
        results = self.results
        sizes = np.reshape(self.trainsizes, (1, -1))

        nsizes = sizes.shape[1]
        nparams = len(results)

        metrics = list(results[0]["summary"]["val"].keys())
        colnames = ["params", "size"] + metrics
        report_by_size = np.empty((nsizes, len(colnames), nparams), dtype=object)

        for p, paramset_result in enumerate(results):
            params = paramset_result["params"]
            res_val = paramset_result["summary"]["val"]

            means_by_size = [np.mean(res_val[metric], axis=1) for metric in metrics]

            means_by_size = np.append(sizes, means_by_size, axis=0)

            param_strgs = np.reshape([str(params)] * nsizes, (1, -1))
            means_by_size = np.append(param_strgs, means_by_size, axis=0).T

            report_by_size[:, :, p] = means_by_size
        return report_by_size

    def get_best_params(self, opt_metric, maximize_opt_metric):
        """PROVIDED
        Determines the best parameter set for each train size,
        based on a specific metric.

        PARAMS:
            opt_metric: optimized metric. one of the metrics returned
                        from eval_func, with '_mean' appended for the
                        summary stat. This is the mean metric used to
                        determine the best parameter set for each
                        training set size

            maximize_opt_metric: True if the max of opt_metric should be
                                 used to determine the best parameters.
                                 False if the min should be used.

        RETURNS: list of best parameter set indicies for each training set size
        """
        results = self.results
        report_by_size = self.report_by_size

        metrics = list(results[0]["summary"]["val"].keys())

        best_param_inds = None
        metric_idx = metrics.index(opt_metric)

        report_opt_metric = report_by_size[:, metric_idx + 2, :]

        if maximize_opt_metric:
            best_param_inds = np.argmax(report_opt_metric, axis=1)
        else:
            best_param_inds = np.argmin(report_opt_metric, axis=1)

        return best_param_inds

    def get_best_params_strings(self):
        """PROVIDED
        Generates a list of strings of the best params for each size
        RETURNS: list of strings of the best params for each size
        """

        best_param_inds = self.best_param_inds
        results = self.results

        return [str(results[p]["params"]) for p in best_param_inds]

    def get_report_best_params_for_size(self, size):
        """PROVIDED
        Get the mean validation summary for the best parameter set
        for a specific size for all metrics.
        PARAMS:
            size: index of desired train set size for the best
                  paramset to come from. Size here is the index in
                  the trainsizes list, NOT the actual number of folds.

        RETURNS: the best parameter report for the size as an s x m
                 dataframe. Where each row is for a different size, and
                 each column is for a different summary metric.
        """
        best_param_inds = self.best_param_inds
        report_by_size = self.report_by_size

        bp_index = best_param_inds[size]
        size_len = len(size) if type(size) is list else 1

        metrics = list(self.results[0]["summary"]["val"].keys())
        colnames = ["params", "size"] + metrics
        report_best_params_for_size = pd.DataFrame(
            report_by_size[size_idx].T[bp_index].reshape(size_len, -1), columns=colnames
        )
        return report_best_params_for_size

    """ PROVIDED
    Plotting code to display the result of the grid search and cross-validation
    """

    def plot_cv(self, foldsindices, results, summary, metrics, size):
        """PROVIDED
        Plotting function for after perform_cross_validation(),
        displaying the train and val set performances for each rotation
        of the training set.

        PARAMS:
            foldsindices: indices of the train sets tried
            results: results from perform_cross_validation()
            summary: mean and standard deviations of the results
            metrics: list of result metrics to plot. Available metrics
                     are the keys in the dict returned by eval_func
            size: train set size

        RETURNS: the figure and axes handles
        """
        nmetrics = len(metrics)

        fig, axs = plt.subplots(nmetrics, 1, figsize=(12, 6))
        fig.subplots_adjust(hspace=0.4)

        axs = np.array(axs).ravel()

        for metric, ax in zip(metrics, axs):
            res_train = np.mean(results["train"][metric], axis=1)
            res_val = np.mean(results["val"][metric], axis=1)

            ax.plot(foldsindices, res_train, label="train")
            ax.plot(foldsindices, res_val, label="val")

            ax.set(ylabel=metric)
        axs[0].legend(loc="upper right")
        axs[0].set(xlabel="Fold Index")
        axs[0].set(title="Performance for Train Set Size " + str(size))
        return fig, axs

    def plot_param_train_val(self, metrics, paramidx=0, view_test=False):
        """PROVIDED
        Plotting function for after grid_cross_validation(),
        displaying the mean (summary) train and val set performances
        for each train set size.

        PARAMS:
            metrics: list of summary metrics to plot. '_mean' or '_std'
                     must be appended to the end of the base metric name.
                     These base metric names are the keys in the dict
                     returned by eval_func
            paramidx: parameter set index
            view_test: flag to view the test set results

        RETURNS: the figure and axes handles
        """
        sizes = self.trainsizes
        results = self.results

        summary = results[paramidx]["summary"]
        params = results[paramidx]["params"]

        nmetrics = len(metrics)

        fig, axs = plt.subplots(nmetrics, 1, figsize=(12, 6))

        axs = np.array(axs).ravel()

        for metric, ax in zip(metrics, axs):
            res_train = np.mean(summary["train"][metric], axis=1)
            res_val = np.mean(summary["val"][metric], axis=1)

            # Plot
            ax.plot(sizes, res_train, label="train")
            ax.plot(sizes, res_val, label="val")
            if view_test:
                res_test = np.mean(summary["test"][metric], axis=1)
                ax.plot(sizes, res_test, label="test")
            ax.set(ylabel=metric)
            ax.set_xticks(sizes)

        # Final labels
        axs[-1].set(xlabel="Train Set Size (# of folds)")
        axs[0].set(title=str(params))
        axs[0].legend(loc="upper right")
        return fig, axs

    def plot_allparams_val(self, metrics):
        """PROVIDED
        Plotting function for after grid_cross_validation(), displaying
        mean (summary) validation set performances for each train size
        for all parameter sets for the specified metrics.

        PARAMS:
            metrics: list of summary metrics to plot. '_mean' or '_std'
                     must be append to the end of the base metric name.
                     These base metric names are the keys in the dict
                     returned by eval_func

        RETURNS: the figure and axes handles
        """
        sizes = self.trainsizes
        results = self.results

        nmetrics = len(metrics)

        # Initialize figure plots
        fig, axs = plt.subplots(nmetrics, 1, figsize=(10, 6))
        # fig.subplots_adjust(hspace=.4)
        # When 1 metric is provided, allow the axs to be iterable
        axs = np.array(axs).ravel()

        # Construct each subplot: one for each metric
        for metric, ax in zip(metrics, axs):
            # Iterate over the hyper-parameter sets
            for p, param_results in enumerate(results):
                summary = param_results["summary"]
                params = param_results["params"]
                # Compute the mean for multiple outputs
                res_val = np.mean(summary["val"][metric], axis=1)
                ax.plot(sizes, res_val, label=str(params))

            # Labels for this metric
            ax.set(ylabel=metric)
            ax.set_xticks(sizes)

        # Final labels
        axs[-1].set(xlabel="Train Set Size (# of folds)")
        axs[0].set(title="Validation Performance")
        axs[0].legend(
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            ncol=1,
            borderaxespad=0.0,
            prop={"size": 8},
        )
        return fig, axs

    def plot_best_params_by_size(self):
        """PROVIDED
        Plotting function for after grid_cross_validation(), displaying
        mean (summary) train and validation set performances for the best
        parameter set for each train size for the optimized metric.

        RETURNS: the figure and axes handles
        """
        results = self.results
        metric = self.opt_metric
        best_param_inds = self.best_param_inds
        sizes = np.array(self.trainsizes)

        # Unique set of best params for the legend
        unique_param_sets = np.unique(best_param_inds)
        lgnd_params = [self.paramsets[p] for p in unique_param_sets]

        # Data set types to display
        set_names = ["train", "val", "test"]

        # Initialize figure
        fig, axs = plt.subplots(len(set_names), 1, figsize=(10, 8))

        # When more than one metric is provided, allow the axs to be iterable
        axs = np.array(axs).ravel()

        # Construct each subplot: iterate over data set types (train and val only)
        for i, (ax, set_name) in enumerate(zip(axs, set_names)):
            # Iterate over the unique set of hyperparameters
            for p in unique_param_sets:
                # Obtain indices of sizes this paramset was best for
                param_size_inds = np.where(best_param_inds == p)[0]
                param_sizes = sizes[param_size_inds]
                param_sizes.sort()
                # Compute the mean over multiple outputs for each size
                param_summary = results[p]["summary"][set_name]
                metric_scores = np.mean(
                    param_summary[metric][param_size_inds, :], axis=1
                )
                # Plot the param results for each size it was the best for
                ax.scatter(param_sizes, metric_scores, s=120, marker=(p + 2, 1))

            # Ticks for all data set sizes
            ax.set_xticks(sizes)

            set_name += " Set Performance"
            ax.set(ylabel=metric, title=set_name)

        # Final labels
        axs[-1].set(xlabel="Train Set Size (# of folds)")
        axs[0].legend(
            lgnd_params,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            ncol=1,
            borderaxespad=0.0,
            prop={"size": 8},
        )

        return fig, axs


""" TODO

"""


param_lists = {
    "alpha": [10**i for i in range(7)],  # Factors of 10 from 1 to 1,000,000
    "max_iter": [10000],  # Fixed max_iter value
}


allparamsets = generate_paramsets(param_lists)

allparamsets


""" TODO

"""
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


""" PROVIDED: EXECUTE CELL

Checkpoint file: for a given gridsearch, we checkpoint the individual training experiments to
a file.  This way, if the search is interrupted, we can restart the search where it left off.  
Also - we can add more hyper-parameter sets later and restart the search.

"""

fullcvfname = "hw6_crossval_ridge_checkpoint.pkl"


""" TODO
Execute the grid_cross_validation() procedure for all 
parameters and sizes

CLEAN
"""


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

""" PROVIDED: EXECUTE CELL
Obtain all the results for all parameters, for all sizes, for all
rotations. This is the results attribute of the crossval object 
"""
all_results = crossval.results
len(all_results)


""" PROVIDED: EXECUTE CELL
Obtain and display the indices of the best hyper-parameters for each training set size  
using the 'best_param_inds' item from the crossval_report dict
"""
best_param_inds = crossval_report["best_param_inds"]
best_param_inds


""" TODO
Display the list of the best parameter sets for each size. Use
get_best_params_strings()
"""

best_param_sets = crossval.get_best_params_strings()
print(best_param_sets)


""" TODO
Plot the validation rmse mean results for all parameters over all train 
sizes, for the specified metrics. 
"""
metrics_to_plot = ["rmse_mean"]  # This specifies that we want to plot the RMSE mean
crossval.plot_allparams_val(metrics=metrics_to_plot)
plt.show()


""" TODO
Plot the mean (summary) train and validation set performances for 
the best parameter set for each train size for the optimized
metrics. Use plot_best_params_by_size()
"""


crossval.plot_best_params_by_size()


plt.show()
