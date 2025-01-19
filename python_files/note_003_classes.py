import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Setting default figure parameters
plt.rcParams.update(
    {
        "figure.figsize": (10, 5),
        "font.size": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
    }
)

# Load dataset
fname = "../hw3/dataset/baby_data_raw"
baby_data_raw = pd.read_csv(fname)


# Define helper functions and classes
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribs):
        self.attribs = attribs

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return X[self.attribs]


class InterpolationImputer(BaseEstimator, TransformerMixin):
    def __init__(self, method="quadratic"):
        self.method = method

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return X.interpolate(method=self.method).ffill().bfill()


class Filter(BaseEstimator, TransformerMixin):
    def __init__(self, attribs=None, kernel=[]):
        self.attribs = attribs
        self.kernelsize = len(kernel)
        if self.kernelsize % 2 == 0:
            raise ValueError("Kernel size must be odd.")
        self.weights = kernel

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        Xout = X.copy()
        if self.attribs is None:
            self.attribs = Xout.columns

        for attrib in self.attribs:
            vals = Xout[attrib].values
            pad_size = self.kernelsize // 2
            vals = np.concatenate(
                (np.full(pad_size, vals[0]), vals, np.full(pad_size, vals[-1]))
            )
            Xout[attrib] = np.convolve(vals, self.weights, mode="valid")

        return Xout


class DerivativeComputer(BaseEstimator, TransformerMixin):
    def __init__(self, attribs=None, prefix="d_", dt=1.0):
        self.attribs = attribs
        self.prefix = prefix
        self.dt = dt

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        Xout = X.copy()
        if self.attribs is None:
            self.attribs = Xout.columns

        for attrib in self.attribs:
            vals = Xout[attrib].values
            deriv = np.append((vals[1:] - vals[:-1]) / self.dt, 0)
            Xout[self.prefix + attrib] = deriv

        return Xout


def computeBoxWeights(length=3):
    return np.ones(length) / length


# Preprocessing Pipeline
fieldsKin = [col for col in baby_data_raw.columns if col.endswith(("_x", "_y", "_z"))]
dt = 0.02
box_filter_kernel = computeBoxWeights(9)

pipe_preprocessor = Pipeline(
    [
        ("imputer", InterpolationImputer()),
        ("smoother", Filter(attribs=fieldsKin, kernel=box_filter_kernel)),
        ("deriv", DerivativeComputer(attribs=fieldsKin, dt=dt)),
    ]
)

# Apply preprocessing pipeline
baby_data_clean = pipe_preprocessor.fit_transform(baby_data_raw)

# Feature Selection Pipelines
fieldsKinVel = fieldsKin + ["d_" + f for f in fieldsKin]
fieldsRobot = ["robot_vel_l", "robot_vel_r"]

pipe_kin_vel = Pipeline([("selector", DataFrameSelector(attribs=fieldsKinVel))])
pipe_time = Pipeline([("time_selector", DataFrameSelector(attribs=["time"]))])
pipe_robot_vel = Pipeline([("robot_selector", DataFrameSelector(attribs=fieldsRobot))])

# Extract features
inputs_pos_vel = pipe_kin_vel.fit_transform(baby_data_clean).values
time = pipe_time.fit_transform(baby_data_clean).values
robot_vel = pipe_robot_vel.fit_transform(baby_data_clean).values

# Define Labels
labels_linear = robot_vel[:, 0] < -0.0025
labels_rotational = robot_vel[:, 1] < -0.02


# Plotting Functions
def visualize_model_output_timeseries(
    y, preds, scores, threshold=0, offsets=(0, -2, -8)
):
    plt.figure()
    plt.plot(y, label="True Class", color="g", alpha=0.6)
    plt.plot(preds + offsets[1], label="Predicted Class (Offset)", color="b", alpha=0.6)
    plt.plot(scores + offsets[2], label="Scores (Offset)", color="r", alpha=0.6)
    plt.axhline(
        threshold + offsets[2],
        color="k",
        linestyle="--",
        label=f"Threshold = {threshold}",
    )
    plt.xlabel("Time (s)")
    plt.legend()
    plt.title("Model Output Time Series")
    plt.show()


# Classifier for Linear Velocity
clf_linear = SGDClassifier(loss="log_loss", random_state=42, max_iter=10000, tol=1e-3)
scores_linear = cross_val_predict(
    clf_linear, inputs_pos_vel, labels_linear, cv=30, method="decision_function"
)
preds_linear = cross_val_predict(
    clf_linear, inputs_pos_vel, labels_linear, cv=30, method="predict"
)
visualize_model_output_timeseries(labels_linear, preds_linear, scores_linear)

# Classifier for Rotational Velocity
clf_rotational = SGDClassifier(
    loss="log_loss", random_state=42, max_iter=10000, tol=1e-3
)
scores_rotational = cross_val_predict(
    clf_rotational, inputs_pos_vel, labels_rotational, cv=30, method="decision_function"
)
preds_rotational = cross_val_predict(
    clf_rotational, inputs_pos_vel, labels_rotational, cv=30, method="predict"
)
visualize_model_output_timeseries(
    labels_rotational, preds_rotational, scores_rotational
)
