import pandas as pd
import numpy as np
import os, re, fnmatch
import matplotlib.pyplot as plt
import matplotlib.patheffects as peffects

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, confusion_matrix, roc_curve, auc
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
import scipy.stats as stats

# Default figure parameters
plt.rcParams["figure.figsize"] = (8, 4)
plt.rcParams["font.size"] = 12
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16

""" TODO
Load data from subject k2 for week 5
Display info() for the data

These are data obtained from a baby on the SIPPC. 3D Position (i.e. kinematic)
data are collected at 50 Hz, for the x, y, and z positions in meters, for 
various joints such as the wrists, elbows, shoulders, etc.
"""

# Local file name
fname = "../dataset/baby_data_raw"

baby_data_raw = pd.read_csv(fname)
baby_data_raw.info()

""" PROVIDED
"""


## Support for identifying kinematic variable columns
def get_kinematic_properties(data):
    # Regular expression for finding kinematic fields
    regx = re.compile("_[xyz]$")

    # Find the list of kinematic fields
    fields = list(data)
    fieldsKin = [x for x in fields if regx.search(x)]
    return fieldsKin


def position_fields_to_velocity_fields(fields, prefix="d_"):
    """
    Given a list of position columns, produce a new list
    of columns that include both position and velocity
    """
    fields_new = [prefix + x for x in fields]
    return fields + fields_new


""" PROVIDED
Get the names of the sets of fields for the kinematic features and the 
velocities
"""
fieldsKin = get_kinematic_properties(baby_data_raw)
fieldsKinVel = position_fields_to_velocity_fields(fieldsKin)
print(fieldsKinVel)

"""
Fields that describe the linear and rotational velocities of the robot
"""
fieldsRobot = ["robot_vel_l", "robot_vel_r"]


# DataFrameSelector
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribs):
        self.attribs = attribs

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        """
        :param X: a DataFrame
        :return: a DataFrame that contains the selected attributes
        """
        return X[self.attribs]


# InterpolationImputer
class InterpolationImputer(BaseEstimator, TransformerMixin):
    def __init__(self, method="quadratic"):
        self.method = method

    def fit(self, x, y=None):
        return self

    def transform(self, X):  # TODO
        """
        :param X: is a DataFrame
        :return: DataFrame without NaNs
        """
        # Interpolate holes within the data
        Xout = X.interpolate(method=self.method)

        # TODO: Fill in the NaNs on the edges of the data
        Xout = Xout.ffill().bfill()

        # Return the imputed dataframe
        return Xout


# Filter


def computeBoxWeights(length=3):
    """
    PROVIDED

    Computes the kernel weights for a Box Filter

    :param length: the number of terms in the filter kernel
    :return: a vector of the specified length
    """

    return np.ones((length,)) / length


class Filter(BaseEstimator, TransformerMixin):
    def __init__(self, attribs=None, kernel=[]):
        # Attributes to filter
        self.attribs = attribs

        # Number of kernel elements
        self.kernelsize = kernel.shape[0]

        # Check that we have an odd kernel size
        if self.kernelsize % 2 == 0:
            raise Exception("Expecting an odd kernel size")

        # Compute the kernel element values
        self.weights = kernel

    def fit(self, x, y=None):
        return self

    def transform(self, X):  # TODO
        """
        :param X: is a DataFrame
        :return:: a DataFrame with the smoothed signals
        """
        w = self.weights
        # ks = self.kernelsize
        # Create a copy of the original DataFrame
        Xout = X.copy()

        # Select all attributes if unspecified
        if self.attribs is None:
            self.attribs = Xout.columns

        # Iterate over the attributes
        for attrib in self.attribs:
            # Extract the numpy vector
            vals = Xout[attrib].values
            # TODO: pad signal at both the front and end of the vector so that after
            #   convolution, the length is the same as the lenght of vals.  Use
            #   vals[0] and vals[-1] to pad the front and back, respectively.
            #   You may assume that the kernel size is always odd

            # Padding size on each side
            pad_size = self.kernelsize // 2

            # Compute the front and back padding vectors
            frontpad = np.full(pad_size, vals[0])
            backpad = np.full(pad_size, vals[-1])
            vals = np.concatenate((frontpad, vals, backpad))

            # TODO: apply filter
            # Implementation is the same as for the DerivativeComputer element, but
            #   more general.  You must iterate over the kernel elements.
            #   (NOTE: due to the wonky way indexing works in python, you will have
            #   specific code for one index & iterate over the remaining k-1 indices)

            # Filter window offset
            ofst = self.kernelsize - 1
            # Last term
            avg = np.zeros(len(vals) - self.kernelsize + 1)

            # Rest of the terms
            for i in range(len(avg)):
                avg[i] += sum(vals[i + j] * w[j] for j in range(self.kernelsize))

            # replace noisy signal with filtered signal
            Xout[attrib] = pd.Series(avg)

        return Xout


"""
PROVIDED
"""


class DerivativeComputer(BaseEstimator, TransformerMixin):
    def __init__(self, attribs=None, prefix="d_", dt=1.0):
        self.attribs = attribs
        self.prefix = prefix
        self.dt = dt

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        """
        :param X: a DataFrame
        :return: a DataFrame with additional features for the derivatives
        """
        Xout = X.copy()
        if self.attribs is None:
            self.attribs = Xout.columns

        # Iterate over all of the attributes that we need to compute velocity over
        for attrib in self.attribs:
            # Extract the numpy array of data
            vals = Xout[attrib].values
            # Compute the difference between neighboring timeseries elements
            diff = vals[1:] - vals[0:-1]
            # Take into account the amount of time between timeseries samples
            deriv = diff / self.dt
            # Add a zero to the end so the resulting velocity vector is the same
            #   length as the position vector
            deriv = np.append(deriv, 0)

            # Add a new derivative attribute to the DataFrame
            attrib_name = self.prefix + attrib
            Xout[attrib_name] = pd.Series(deriv)

        return Xout


""" TODO
Create four pipelines. 

The first pipeline is used for the raw dataframe:
1.  Impute values for the kinematic features using a quadratic imputer
2.  Smooth the kinematic features.  Use a Box Filter of length 9 
3.  Compute derivatives of all of the kinematic features.  dt is 0.02 seconds
The output is a cleaned data frame.

The cleaned data frame will be input to several additional pipelines:

The second pipeline extracts the kinematic and velocity (derivative)
features from the dataframe.

The third pipeline extracts the time stamp from the dataframe.

The fourth pipeline extracts the robot velocity from the dataframe (both the linear and rotational velocity).
"""
# Sampling rate: number of seconds between each time sample
dt = 0.02

# Define the box filter kernel for smoothing
box_filter_kernel = computeBoxWeights(length=9)

# Initial pre-processing
pipe_preprocessor = Pipeline(
    [
        (
            "imputer",
            InterpolationImputer(method="quadratic"),
        ),  # Step 1: Impute missing values
        (
            "smoother",
            Filter(attribs=fieldsKin, kernel=box_filter_kernel),
        ),  # Step 2: Smooth data
        (
            "deriv",
            DerivativeComputer(attribs=fieldsKin, dt=dt),
        ),  # Step 3: Compute derivatives
    ]
)

# Position, velocity selector
pipe_kin_vel = Pipeline(
    [
        (
            "selector",
            DataFrameSelector(attribs=fieldsKinVel),
        )  # Selecting both position and velocity fields
    ]
)

# Time selector (assuming 'time' is the column name for the timestamps)
pipe_time = Pipeline(
    [
        (
            "time_selector",
            DataFrameSelector(attribs=["time"]),
        )  # Isolate the time stamp column
    ]
)

# Robot velocity selector
pipe_robot_vel = Pipeline(
    [
        (
            "robot_selector",
            DataFrameSelector(attribs=fieldsRobot),
        )  # Extract robot velocity fields
    ]
)


""" TODO
Use the above pipelines to extract the data with kinematic and velocity 
features, the time, and the robot velocity.

See the lecture on classifers for examples.
"""

# Use the first pipeline to perform an initial cleaning of the data
baby_data_clean = pipe_preprocessor.fit_transform(baby_data_raw)

# Use the result from the first pipeline to extract the kinematic and velocity features
data_pos_vel = pipe_kin_vel.fit_transform(baby_data_clean)

# Use the result from the first pipeline to extract the time stamps
data_time = pipe_time.fit_transform(baby_data_clean)

# Use the result from the first pipeline to get the robot velocity
data_robot_vel = pipe_robot_vel.fit_transform(baby_data_clean)

# Transform the dataframes into numpy arrays
inputs_pos_vel = data_pos_vel.values
time = data_time.values
robot_vel = data_robot_vel.values

# Determine the number of samples
nsamples = inputs_pos_vel.shape[0]


""" TODO
Create a plot that contains both the linear velocity (robot_vel[:,0]) and
rotational velocity (robot_vel[:,1]).  The plot should contain appropriate 
labels

Note: units are m/s and rad/s, respectively
"""
plt.style.use("default")
plt.figure(figsize=(10, 5))  # Adjust the figure size as needed

# Plotting linear velocity
plt.plot(
    time[:, 0], robot_vel[:, 0], label="Linear Velocity (m/s)", color="b"
)  # Assuming time is in the correct format and column
plt.xlabel("Time (s)")
plt.ylabel("Linear Velocity (m/s)")
plt.title("Linear and Rotational Velocity of the Robot")
plt.legend(loc="upper left")

# Creating a second y-axis for rotational velocity
ax2 = plt.gca().twinx()
ax2.plot(time[:, 0], robot_vel[:, 1], label="Rotational Velocity (rad/s)", color="r")
ax2.set_ylabel("Rotational Velocity (rad/s)")
ax2.legend(loc="upper right")

plt.show()


""" PROVIDED
Create labels that correspond to "fast backward motion" and
"fast right rotational motion"

"""
# Fast backward motion
labels_linear = robot_vel[:, 0] < -0.0025

# Rightward turns
labels_rotational = (robot_vel[:, 1]) < -0.02

""" TODO
Augment the figure you created above to show the two newly-created
class labels.  Make sure that the resulting figure is easy to read
"""

plt.figure(figsize=(10, 5))  # Increase figure size for better readability

# Plotting linear velocity with the condition of fast backward motion highlighted
plt.plot(
    time[:, 0], robot_vel[:, 0], label="Linear Velocity (m/s)", color="b", alpha=0.3
)  # Light plot as background
plt.scatter(
    time[labels_linear, 0],
    robot_vel[labels_linear, 0],
    color="b",
    label="Fast Backward Motion (< -0.0025 m/s)",
    s=6,
)  # Highlight specific points

# Setting primary y-axis label
plt.xlabel("Time (s)")
plt.ylabel("Linear Velocity (m/s)")
plt.title("Linear and Rotational Velocity of the Robot with Motion Labels")
plt.legend(loc="upper left")

# Creating a second y-axis for rotational velocity
ax2 = plt.gca().twinx()
ax2.plot(
    time[:, 0],
    robot_vel[:, 1],
    label="Rotational Velocity (rad/s)",
    color="r",
    alpha=0.3,
)  # Light plot as background
ax2.scatter(
    time[labels_rotational, 0],
    robot_vel[labels_rotational, 1],
    color="r",
    label="Fast Right Rotational Motion (< -0.02 rad/s)",
    s=6,
)  # Highlight specific points
ax2.set_ylabel("Rotational Velocity (rad/s)")
ax2.legend(loc="upper right")

plt.show()


""" TODO
"""
# Input
X = inputs_pos_vel

# Desired output
y = labels_linear

# TODO: Create and fit the classifer
clf = SGDClassifier(loss="log_loss", random_state=1138, max_iter=10000, tol=1e-3)
clf.fit(X, y)

# TODO: extract the predictions and the decision function scores from the model for the
#  entire data set
preds = clf.predict(X)

scores = clf.decision_function(X)


""" PROVIDED
"""


# Generate a color map plot for a confusion matrix
def confusion_mtx_colormap(
    mtx, xnames, ynames, cbarlabel="", FIGWIDTH=5, FIGHEIGHT=5, FONTSIZE=14
):
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
    fig, ax = plt.subplots(figsize=(FIGWIDTH, FIGHEIGHT))
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
                "%s = %.3f" % (lbl[i, j], mtx[i, j]),
                ha="center",
                va="center",
                color="k",
            )
            # text.set_path_effects([peffects.withStroke(linewidth=2,
            # foreground='w')])

    return fig, ax


def display_confusion_matrix(y, preds, label_names):
    """
    Compute the confusion matrix using sklearn's confusion_matrix() function and
    generate a color map using the provided confusion_mtx_colormap() for the model
    built using the distance labels.

    :params y: Ground truth labels
    :params preds: Crisp predictions made by the model (i.e., after thresholding)
    :return: Number of positive and negative examples (ground truth)
    """
    dist_confusion_mtx = confusion_matrix(y, preds)
    confusion_mtx_colormap(
        dist_confusion_mtx, label_names, label_names, cbarlabel=""
    )  # TODO
    plt.show()
    nneg = dist_confusion_mtx[0].sum()
    npos = dist_confusion_mtx[1].sum()
    return npos, nneg


"""
TODO: Complete the visualization implementations
"""


def visualize_model_output_timeseries(
    y, preds, scores, threshold=0, offset_pred=-2, offset_scores=-8
):
    """
    Plot timeseries on a single axis:
    1. True class (y)
    2. Predicted class (preds)
    3. Prediction scores (scores)

    In addition, draw a horizontal line over the scores that shows the decision threshold
    (by default the decision threshold is zero)

    Don't forget to supply a meaningful legend and to label the horizontal axis
    """

    plt.figure(figsize=(10, 6))
    plt.clf()

    # Plot true class
    plt.plot(y, label="True Class", color="g", alpha=0.6)

    # Plot predicted class with offset
    plt.plot(
        preds + offset_pred, label="Predicted Class (Offset)", color="b", alpha=0.6
    )

    # Plot scores with offset
    plt.plot(scores + offset_scores, label="Scores (Offset)", color="r", alpha=0.6)

    # Plot threshold line
    plt.axhline(
        threshold + offset_scores,
        color="k",
        linestyle="--",
        label=f"Threshold = {threshold}",
    )

    plt.xlabel("Time (s)")
    plt.legend()
    plt.title("Model Output Time Series")
    plt.show()


"""
TODO

Compute the ROC Curve and generate the KS plot
"""


def ks_roc_plot(targets, scores, FIGWIDTH=16, FIGHEIGHT=4, FONTSIZE=16):
    """
    Generate a figure with two plots:
    1. Distributions of the TPR and FPR over a set of thresholds.  Include
    a vertical line that shows the threshold that maximizes the difference
    between TPR and FPR
    2. ROC Curve.  Show the point on the curve that corresponds to the same
    threshold

    PARAMS:
        targets: list of true target labels
        scores: list of predicted scores
    RETURNS:
        fpr: false positive rate
        tpr: true positive rate
        thresholds: thresholds used for the ROC curve
        auc: Area under the ROC Curve
        fig, axs: corresponding handles for the figure and axis
    """
    plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))
    plt.clf()  # Clear any previous figures

    # Compute the ROC curve
    fpr, tpr, thresholds = roc_curve(targets, scores)

    # Calculate the K-S statistic (maximum difference between TPR and FPR)
    diff = tpr - fpr
    auc_res = auc(fpr, tpr)
    elem_max = np.argmax(diff)
    thresh_max = thresholds[elem_max]
    print("K-S Distance:", diff[elem_max])

    # Generate figure with two axes
    fig, axs = plt.subplots(1, 2, figsize=(FIGWIDTH, FIGHEIGHT))

    # First subplot: Plot TPR, FPR, and their difference
    axs[0].plot(thresholds, tpr, label="TPR", color="g")
    axs[0].plot(thresholds, fpr, label="FPR", color="r")
    axs[0].plot(thresholds, diff, label="Difference", color="b")
    axs[0].axvline(thresh_max, color="k", linestyle="--", label="Best Threshold")
    axs[0].set_xlabel("Thresholds")
    axs[0].set_title("TPR, FPR, and Difference")
    axs[0].legend()

    # Second subplot: Plot ROC curve
    axs[1].plot(fpr, tpr, label="ROC curve", color="b")
    axs[1].scatter(
        fpr[elem_max],
        tpr[elem_max],
        color="r",
        label=f"Best Threshold ({thresh_max:.2f})",
    )
    axs[1].plot([0, 1], [0, 1], color="green", linestyle="--")
    axs[1].set_xlabel("False Positive Rate")
    axs[1].set_ylabel("True Positive Rate")
    axs[1].set_title("ROC Curve")
    auc_text = axs[1].text(
        0.6, 0.2, f"AUC = {auc_res:.4f}", color="k", fontsize=FONTSIZE
    )
    axs[1].legend()
    plt.show()
    print("AUC:", auc_res)

    return fpr, tpr, thresholds, thresh_max, auc_res, fig, axs


""" 
TODO

Plot histograms of the scores from the model.
1. Histogram of all scores
2. Overlapping histograms of the scores for the positive and negative examples

Make sure to include a horizontal line at the best threshold (K-S threshold).
"""


def plot_score_histograms(
    scores, y, best_thresh=None, nbins=41, FIGWIDTH=14, FIGHEIGHT=6
):
    """
    Generate two plots:
    1. Histogram of all scores
    2. Two histograms: one for positive examples and the other for negative examples

    :param scores: Model scores for all samples
    :param y: Ground truth labels for all samples
    """

    scores_pos = [s for (s, l) in zip(scores, y) if l]  # Positive scores
    scores_neg = [s for (s, l) in zip(scores, y) if not l]  # Negative scores

    plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))
    plt.clf()

    # First plot: Histogram of all scores
    plt.subplot(1, 2, 1)
    plt.hist(scores, bins=nbins, alpha=0.6, color="b", label="All scores")
    if best_thresh is not None:
        plt.axvline(
            best_thresh,
            color="r",
            linestyle="--",
            label=f"Best Threshold = {best_thresh:.2f}",
        )
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title("Histogram of All Scores")
    plt.legend()

    # Second plot: Histogram of positive and negative scores
    plt.subplot(1, 2, 2)
    plt.hist(scores_pos, bins=nbins, alpha=0.6, color="g", label="Positive Scores")
    plt.hist(scores_neg, bins=nbins, alpha=0.6, color="r", label="Negative Scores")
    if best_thresh is not None:
        plt.axvline(
            best_thresh,
            color="r",
            linestyle="--",
            label=f"Best Threshold = {best_thresh:.2f}",
        )
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title("Positive vs Negative Scores")
    plt.legend()
    plt.show()


# EXECUTE CELL: Visualize the predictions made by the model in timeseries form
visualize_model_output_timeseries(y, preds, scores)

# EXECUTE CELL
display_confusion_matrix(y, preds, ["not", "fast backward"])

# EXECUTE CELL: Generate the TPR/FPR and ROC plots
fpr, tpr, thresholds, best_thresh, auc_res, fig, axs = ks_roc_plot(y, scores)


# EXECUTE CELL: Plot score histograms
plot_score_histograms(scores, y, best_thresh)

""" TODO
"""
# Input
X = inputs_pos_vel

# Desired output
y = labels_rotational

# TODO: Create and fit the classifer
clf = SGDClassifier(loss="log_loss", random_state=1138, max_iter=10000, tol=1e-3)

# Fit the classifier
clf.fit(X, y)

# TODO: extract the predictions and the decision function scores from the model for the entire data set
preds = clf.predict(X)

scores = clf.decision_function(X)

# Output the predictions and scores
print("Predictions:", preds)
print("Scores:", scores)

# EXECUTE CELL: Visualize the predictions made by the model in timeseries form
visualize_model_output_timeseries(y, preds, scores)

# EXECUTE CELL
display_confusion_matrix(y, preds, ["not", "fast right"])

# EXECUTE CELL: Generate the TPR/FPR and ROC plots
fpr, tpr, thresholds, best_thresh, auc_res, fig, axs = ks_roc_plot(y, scores)

# EXECUTE CELL: Plot score histograms
plot_score_histograms(scores, y, best_thresh)

overlap = np.sum(labels_linear & labels_rotational)
total_time = len(labels_linear)
percentage_overlap = (overlap / total_time) * 100
percentage_overlap

nan_count = baby_data_clean.isnull().sum().sum()
if nan_count == 0:
    print("All NaNs were successfully eliminated in the preprocessing stage.")
else:
    print(f"There are still {nan_count} NaNs in the data.")


""" TODO
LINEAR VELOCITY

Create another SGDClassifier with the same parameters to predict the linear
velocity label as a function of the kinematic positions and velocities.

W will use cross_val_predict() to fit N models, and compute 
predictions for each sample and their corresponding scores. Use 30 cross 
validation splits (i.e. cv=30).

"""
# Model input
X = inputs_pos_vel
# Model output
y = labels_linear

# TODO: Create and fit the classifer
clf3 = SGDClassifier(loss="log_loss", random_state=1138, max_iter=10000, tol=1e-3)

# TODO: use cross_val_predict() to compute the scores by setting the 'method'
#       parameter equal to 'decision_function'. Please see the reference
#       links above
scores = cross_val_predict(clf3, X, y, cv=30, method="decision_function")

# TODO: use cross_val_predict() to compute the predicted labels by setting
#       the 'method' parameter equal to 'predict'. Please see the reference
#       links above
preds = cross_val_predict(clf3, X, y, cv=30, method="predict")


# EXECUTE CELL: Visualize the predictions made by the model in timeseries form
visualize_model_output_timeseries(y, preds, scores)

# EXECUTE CELL
display_confusion_matrix(y, preds, ["not", "fast backward"])

# EXECUTE CELL: Generate the TPR/FPR and ROC plots
fpr, tpr, thresholds, best_thresh, auc_res, fig, axs = ks_roc_plot(y, scores)
x = np.argmax(tpr - fpr)
print("Best:", x, best_thresh)

# Plot score histograms
plot_score_histograms(scores, y, best_thresh)


""" TODO
ROTATIONAL VELOCITY

Take the same cross-validation approach for the rotational velocity label

"""
# Model input
X = inputs_pos_vel
# Model output
y = labels_rotational

# TODO: Create and fit the classifer
clf4 = SGDClassifier(loss="log_loss", random_state=1138, max_iter=10000, tol=1e-3)
# clf.fit(X, y)

# TODO: use cross_val_predict() to compute the scores by setting the 'method'
#       parameter equal to 'decision_function'. Please see the reference
#       links above
scores = cross_val_predict(clf4, X, y, cv=30, method="decision_function")

# TODO: use cross_val_predict() to compute the predicted labels by setting
#       the 'method' parameter equal to 'predict'. Please see the reference
#       links above (unfortunately, we have to refit the models)
preds = cross_val_predict(clf4, X, y, cv=30, method="predict")

# EXECUTE CELL
display_confusion_matrix(y, preds, ["not", "fast right"])


# EXECUTE CELL: Generate the TPR/FPR and ROC plots
fpr, tpr, thresholds, best_thresh, auc_res, fig, axs = ks_roc_plot(y, scores)
x = np.argmax(tpr - fpr)
print("Best:", x, best_thresh)

# Plot score histograms
plot_score_histograms(scores, y, best_thresh)

preds_adjusted = (scores >= best_thresh).astype(int)  # Apply the best threshold
display_confusion_matrix(
    y, preds_adjusted, ["not", "fast right"]
)  # Display updated confusion matrix
