import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
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
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["lines.markersize"] = 2
plt.style.use("ggplot")

# Load data
fname = "../hw3/dataset/baby_data_raw"
baby_data_raw = pd.read_csv(fname)

# Visualize raw data
for col in baby_data_raw.columns:
    if col != "time":
        plt.figure(figsize=(20, 10))
        plt.plot(baby_data_raw["time"], baby_data_raw[col], label=col, marker="o")
        plt.title(f"Before Interpolation - {col}")
        plt.xlabel("Time")
        plt.ylabel("Values")
        plt.legend(loc="upper right", fontsize="small", ncol=2)
        plt.show()

# Preprocessing: Interpolation and Visualization
baby_data_raw.interpolate(method="quadratic", inplace=True)
baby_data_raw.fillna(method="bfill", inplace=True)
baby_data_raw.fillna(method="ffill", inplace=True)

for col in baby_data_raw.columns:
    if col != "time":
        plt.figure(figsize=(20, 10))
        plt.plot(baby_data_raw["time"], baby_data_raw[col], label=col, linestyle="-")
        plt.title(f"After Interpolation - {col}")
        plt.xlabel("Time")
        plt.ylabel(f"{col}")
        plt.legend(loc="upper right", fontsize="small", ncol=2)
        plt.show()

# Select kinematic fields and compute derivatives
fieldsKin = [col for col in baby_data_raw.columns if col.endswith(("_x", "_y", "_z"))]
for field in fieldsKin:
    baby_data_raw[f"d_{field}"] = baby_data_raw[field].diff().fillna(0) / 0.02

fieldsKinVel = fieldsKin + [f"d_{col}" for col in fieldsKin]
fieldsRobot = ["robot_vel_l", "robot_vel_r"]

# Extract relevant data
inputs_pos_vel = baby_data_raw[fieldsKinVel].values
time = baby_data_raw["time"].values
robot_vel = baby_data_raw[fieldsRobot].values

# Define labels
labels_linear = robot_vel[:, 0] < -0.0025
labels_rotational = robot_vel[:, 1] < -0.02

# Visualize Robot Velocities
plt.figure(figsize=(12, 6))
plt.plot(time, robot_vel[:, 0], label="Linear Velocity (m/s)")
plt.plot(time, robot_vel[:, 1], label="Rotational Velocity (rad/s)")
plt.title("Robot Velocities")
plt.xlabel("Time (s)")
plt.ylabel("Velocity")
plt.legend()
plt.show()

# Add labels to the velocity plot
plt.figure(figsize=(12, 6))
plt.plot(time, robot_vel[:, 0], label="Linear Velocity (m/s)", alpha=0.7)
plt.plot(time[labels_linear], robot_vel[labels_linear, 0], "bo", label="Fast Backward")
plt.plot(time, robot_vel[:, 1], label="Rotational Velocity (rad/s)", alpha=0.7)
plt.plot(
    time[labels_rotational],
    robot_vel[labels_rotational, 1],
    "ro",
    label="Fast Rightward",
)
plt.title("Robot Velocities with Labels")
plt.xlabel("Time (s)")
plt.ylabel("Velocity")
plt.legend()
plt.show()


# Helper Functions
def display_confusion_matrix(y, preds, label_names):
    conf_mtx = confusion_matrix(y, preds)
    plt.figure(figsize=(6, 6))
    plt.imshow(conf_mtx, cmap="summer", interpolation="nearest")
    # Add class labels
    plt.xticks([0, 1], label_names)
    plt.yticks([0, 1], label_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix with Annotations")
    # Add text annotations for TN, FP, FN, TP
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                f"{labels[i][j]}\n{conf_mtx[i, j]}",
                ha="center",
                va="center",
                color="black",
                fontsize=12,
            )
    # Add color bar
    plt.colorbar()
    plt.grid(False)
    plt.show()


def ks_roc_plot(targets, scores):
    fpr, tpr, thresholds = roc_curve(targets, scores)
    diff = tpr - fpr
    best_idx = np.argmax(diff)
    best_thresh = thresholds[best_idx]
    print(f"Best Threshold: {best_thresh}, KS Distance: {diff[best_idx]}")
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(thresholds, tpr, label="TPR")
    plt.plot(thresholds, fpr, label="FPR")
    plt.axvline(best_thresh, color="r", linestyle="--", label="Best Threshold")
    plt.legend()
    plt.title("TPR and FPR")
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], "k--")
    plt.scatter(fpr[best_idx], tpr[best_idx], c="red", label=f"Best Threshold")
    plt.legend()
    plt.title("ROC Curve")
    plt.show()
    return fpr, tpr, thresholds, best_thresh


# Linear Velocity Classifier
clf = SGDClassifier(loss="log_loss", random_state=1138, max_iter=10000, tol=1e-3)
clf.fit(inputs_pos_vel, labels_linear)
scores = clf.decision_function(inputs_pos_vel)
preds = clf.predict(inputs_pos_vel)
display_confusion_matrix(labels_linear, preds, ["Not", "Fast Backward"])
ks_roc_plot(labels_linear, scores)

# Rotational Velocity Classifier
clf_rot = SGDClassifier(loss="log_loss", random_state=1138, max_iter=10000, tol=1e-3)
clf_rot.fit(inputs_pos_vel, labels_rotational)
scores_rot = clf_rot.decision_function(inputs_pos_vel)
preds_rot = clf_rot.predict(inputs_pos_vel)
display_confusion_matrix(labels_rotational, preds_rot, ["Not", "Fast Right"])
ks_roc_plot(labels_rotational, scores_rot)

# Cross-validation for Linear Velocity
cv_scores = cross_val_predict(
    clf, inputs_pos_vel, labels_linear, cv=30, method="decision_function"
)
cv_preds = cross_val_predict(
    clf, inputs_pos_vel, labels_linear, cv=30, method="predict"
)
display_confusion_matrix(labels_linear, cv_preds, ["Not", "Fast Backward"])
ks_roc_plot(labels_linear, cv_scores)

# Cross-validation for Rotational Velocity
cv_scores_rot = cross_val_predict(
    clf_rot, inputs_pos_vel, labels_rotational, cv=30, method="decision_function"
)
cv_preds_rot = cross_val_predict(
    clf_rot, inputs_pos_vel, labels_rotational, cv=30, method="predict"
)
display_confusion_matrix(labels_rotational, cv_preds_rot, ["Not", "Fast Right"])
ks_roc_plot(labels_rotational, cv_scores_rot)


# Visualize histogram of scores
def plot_score_histograms(scores, labels, best_thresh=None):
    scores_pos = scores[labels]
    scores_neg = scores[~labels]

    plt.figure(figsize=(12, 6))
    plt.hist(scores, bins=40, alpha=0.6, label="All Scores")
    if best_thresh:
        plt.axvline(best_thresh, color="r", linestyle="--", label=f"Best Threshold")
    plt.xlabel("Scores")
    plt.ylabel("Frequency")
    plt.title("Histogram of All Scores")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.hist(scores_pos, bins=40, alpha=0.6, label="Positive Scores", color="g")
    plt.hist(scores_neg, bins=40, alpha=0.6, label="Negative Scores", color="r")
    if best_thresh:
        plt.axvline(best_thresh, color="r", linestyle="--", label=f"Best Threshold")
    plt.xlabel("Scores")
    plt.ylabel("Frequency")
    plt.title("Positive vs Negative Scores")
    plt.legend()
    plt.show()


plot_score_histograms(scores, labels_linear, best_thresh=None)
plot_score_histograms(scores_rot, labels_rotational, best_thresh=None)
