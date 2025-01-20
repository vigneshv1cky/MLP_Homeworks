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


def remove_comments_from_file(input_file, output_file):
    with open(input_file, "r") as f:
        code = f.read()


    )


    code_no_singleline = "\n".join(
        line.split("
    )


    with open(output_file, "w") as f:
        f.write(code_no_singleline)



input_file = "paste.txt"
output_file = "cleaned_paste.txt"


remove_comments_from_file("note_006.py", "note_006.py")


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


    def __init__(self, model, eval_func, rotation_skip=1):

        self.model = model
        self.eval_func = eval_func
        self.rotation_skip = rotation_skip

    def perform_cross_validation(self, X, y, folds_idx, trainsize):



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

        if not os.path.exists(fname):
            raise ValueError("File %s does not exist" % fname)

        with open(fname, "rb") as f:
            self.results = pkl.load(f)
            self.results_partial = pkl.load(f)
            self.id = pkl.load(f)

    def dump_checkpoint(self, fname):

        with open(fname, "wb") as f:
            pkl.dump(self.results, f)
            pkl.dump(self.results_partial, f)
            pkl.dump(self.id, f)

    def reset_results(self):

        self.results = []
        self.results_partial = {}

    def cross_validation_gridsearch(self, X, y, folds_idx, checkpoint_fname=None):



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

        self.report_by_size = self.get_reports()
        self.best_param_inds = self.get_best_params(
            self.opt_metric, self.maximize_opt_metric
        )
        print("CS:", self.id)

        return {
            "report_by_size": self.report_by_size,
            "best_param_inds": self.best_param_inds,
        }




    def get_reports(self):

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


        best_param_inds = self.best_param_inds
        results = self.results

        return [str(results[p]["params"]) for p in best_param_inds]

    def get_report_best_params_for_size(self, size):

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




    def plot_cv(self, foldsindices, results, summary, metrics, size):

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


            ax.plot(sizes, res_train, label="train")
            ax.plot(sizes, res_val, label="val")
            if view_test:
                res_test = np.mean(summary["test"][metric], axis=1)
                ax.plot(sizes, res_test, label="test")
            ax.set(ylabel=metric)
            ax.set_xticks(sizes)


        axs[-1].set(xlabel="Train Set Size (
        axs[0].set(title=str(params))
        axs[0].legend(loc="upper right")
        return fig, axs

    def plot_allparams_val(self, metrics):

        sizes = self.trainsizes
        results = self.results

        nmetrics = len(metrics)


        fig, axs = plt.subplots(nmetrics, 1, figsize=(10, 6))


        axs = np.array(axs).ravel()


        for metric, ax in zip(metrics, axs):

            for p, param_results in enumerate(results):
                summary = param_results["summary"]
                params = param_results["params"]

                res_val = np.mean(summary["val"][metric], axis=1)
                ax.plot(sizes, res_val, label=str(params))


            ax.set(ylabel=metric)
            ax.set_xticks(sizes)


        axs[-1].set(xlabel="Train Set Size (
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

        results = self.results
        metric = self.opt_metric
        best_param_inds = self.best_param_inds
        sizes = np.array(self.trainsizes)


        unique_param_sets = np.unique(best_param_inds)
        lgnd_params = [self.paramsets[p] for p in unique_param_sets]


        set_names = ["train", "val", "test"]


        fig, axs = plt.subplots(len(set_names), 1, figsize=(10, 8))


        axs = np.array(axs).ravel()


        for i, (ax, set_name) in enumerate(zip(axs, set_names)):

            for p in unique_param_sets:

                param_size_inds = np.where(best_param_inds == p)[0]
                param_sizes = sizes[param_size_inds]
                param_sizes.sort()

                param_summary = results[p]["summary"][set_name]
                metric_scores = np.mean(
                    param_summary[metric][param_size_inds, :], axis=1
                )

                ax.scatter(param_sizes, metric_scores, s=120, marker=(p + 2, 1))


            ax.set_xticks(sizes)

            set_name += " Set Performance"
            ax.set(ylabel=metric, title=set_name)


        axs[-1].set(xlabel="Train Set Size (
        axs[0].legend(
            lgnd_params,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            ncol=1,
            borderaxespad=0.0,
            prop={"size": 8},
        )

        return fig, axs






param_lists = {
    "alpha": [10**i for i in range(7)],
    "max_iter": [10000],
}


allparamsets = generate_paramsets(param_lists)

allparamsets



model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("regression", Ridge()),
    ]
)


train_sizes = [1, 2, 3, 4, 5, 7, 10, 13, 16]


opt_metric = "rmse"


crossval = CrossValidationGridSearch(
    model=model,
    paramsets=allparamsets,
    eval_func=predict_score_eval,
    opt_metric=opt_metric,
    maximize_opt_metric=False,
    trainsizes=train_sizes,
    rotation_skip=1,
    submodel_name="regression",
)




fullcvfname = "hw6_crossval_ridge_checkpoint.pkl"











force = True

if force and os.path.exists(fullcvfname):

    print(f"Deleting checkpoint file {fullcvfname} to force a fresh run.")
    os.remove(fullcvfname)

X = np.hstack([MI, theta, dtheta, ddtheta])
y = torque

crossval.cross_validation_gridsearch(
    X=X,
    y=y,
    folds_idx=folds_idx,
    checkpoint_fname=fullcvfname,
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



metrics_to_plot = ["rmse_mean"]
crossval.plot_allparams_val(metrics=metrics_to_plot)
plt.show()





crossval.plot_best_params_by_size()
plt.show()