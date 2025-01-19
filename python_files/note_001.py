# Package imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import get_ipython
from sklearn.datasets import load_iris

# Figure params
FIGURESIZE = (8, 6)
FONTSIZE = 12
plt.rcParams["figure.figsize"] = FIGURESIZE
plt.rcParams["font.size"] = FONTSIZE
plt.rcParams["xtick.labelsize"] = FONTSIZE
plt.rcParams["ytick.labelsize"] = FONTSIZE
plt.style.use("ggplot")

iris_dataset = load_iris()
iris_dataset.keys()
print(iris_dataset["DESCR"])

# Storing the names of the features and target classes
feature_names = iris_dataset["feature_names"]
target_names = iris_dataset["target_names"]


print("Features: \n", feature_names)
print("Targets: \n", target_names)

# Creating variables for the feature and target data
X = iris_dataset["data"]
y = iris_dataset["target"]

# Printing the shapes of the y and X variables
print("Shape of y:", y.shape)
print("Shape of X:", X.shape)

# Storing the number of samples and the number of features
nsamples = X.shape[0]
nfeatures = X.shape[1]

# Printing the number of samples and number of features
print("Number of samples:", nsamples)
print("Number of features:", nfeatures)

predictors = [1, 3]
predictors

npredictors = len(predictors)

pred_names = [iris_dataset["feature_names"][i] for i in predictors]

plt.figure(figsize=(20, 4))
plt.subplots_adjust(wspace=0.3)
for i, fidx in enumerate(predictors, 1):
    plt.subplot(1, 3, i)
    plt.hist(X[:, fidx], bins=15, color="skyblue", edgecolor="black")
    plt.xlabel(iris_dataset["feature_names"][fidx])
    plt.ylabel("Count")
    plt.title(f"Histogram of {iris_dataset['feature_names'][fidx]}")


for fidx in predictors:
    plt.figure(figsize=(8, 4))
    plt.hist(X[:, fidx], bins=15, color="skyblue", edgecolor="black")
    plt.xlabel(iris_dataset["feature_names"][fidx])
    plt.ylabel("Count")
    plt.title(f"Histogram of {iris_dataset['feature_names'][fidx]}")
    plt.show()

# Create a bar plot for the counts for each target class
counts = np.bincount(y)
bin_centers = list(range(3))

plt.figure(figsize=(8, 6))
plt.bar(bin_centers, counts, color="skyblue", edgecolor="black")
plt.xlabel("Target Classes")
plt.ylabel("Count")
plt.title("Counts of Each Target Class")
plt.xticks(bin_centers, iris_dataset["target_names"])
plt.show()


Xpreds = X[:, predictors]

# boxplot for the selected features (sepal width and petal width)
plt.figure(figsize=(10, 6))
plt.boxplot(
    Xpreds,
    patch_artist=True,
    boxprops=dict(facecolor="skyblue", color="black"),
    medianprops=dict(color="red"),
)
plt.xticks([1, 2], pred_names)
plt.xlabel("Features")
plt.ylabel("Value Range")
plt.title("Boxplot of Selected Features")
plt.show()


# Create a version of X with some Nans (this is the original full feature set)
Xpreds2 = np.array(X)
Xpreds2[12, 0] = np.nan
Xpreds2[78, 1] = np.nan
Xpreds2[37, 0] = np.nan


# Computing the descriptive statistics for each feature in Xpreds
means = np.nanmean(Xpreds, axis=0)
medians = np.nanmedian(Xpreds, axis=0)
std_devs = np.nanstd(Xpreds, axis=0)
mins = np.nanmin(Xpreds, axis=0)
maxs = np.nanmax(Xpreds, axis=0)

# Printing the results
print("Means:", means)
print("Medians:", medians)
print("Standard Deviations:", std_devs)
print("Mins:", mins)
print("Maxs:", maxs)


"""
TODO

Using the scatter plot and hist functions, construct a grid of plots depicting the
correlation between all pairings of all of the features
and between all features and the target label.
The figure will contain R by R subplots, where R = npredictors + 1.
When i != j, subplot(i,j) is a scatter plot of the feature i versus feature j.
When i == j, plot the histogram of feature i instead of a scatter plot.
We are also interested in the correlation between each of the features 
and the target label, thus we will combine the selected feature matrix
and the target vector into one large matrix for convenience.

See subfigures: 
https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html
"""

# Use the full set of features
npredictors = X.shape[1]

Xycombo = np.append(y.reshape(-1, 1), X, axis=1)

Xycolnames = ["target"] + iris_dataset["feature_names"]

fig, axs = plt.subplots(npredictors + 1, npredictors + 1, figsize=(15, 15))
fig.subplots_adjust(wspace=0.35, hspace=0.35)
for f1 in range(npredictors + 1):
    for f2 in range(npredictors + 1):
        if f1 == f2:
            axs[f1, f2].hist(
                Xycombo[:, f1], bins=15, color="skyblue", edgecolor="black"
            )
        else:
            axs[f1, f2].scatter(
                Xycombo[:, f2],
                Xycombo[:, f1],
                alpha=0.5,
                color="skyblue",
                edgecolor="black",
            )

        if f1 == npredictors:
            axs[f1, f2].set_xlabel(Xycolnames[f2])
        if f2 == 0:
            axs[f1, f2].set_ylabel(Xycolnames[f1])

plt.show()


""" PROVIDED: EXECUTE CELL
Generate a figure that plots the correlation matrix
as a colormap.
PARAMS:
    corrs: matrix of correlations between the features
    varnames: list of the names of each of the features 
              (e.g. the column names)
"""


def correlationmap(corrs, varnames):
    nvars = corrs.shape[0]

    fig, ax = plt.subplots()
    im = ax.imshow(corrs, cmap="RdBu", vmin=-1, vmax=1)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Pearson Correlation", rotation=-90, va="bottom")

    ax.set_xticks(range(nvars))
    ax.set_yticks(range(nvars))
    ax.set_xticklabels(varnames)
    ax.set_yticklabels(varnames)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(nvars):
        for j in range(nvars):
            text = ax.text(
                j, i, "%.3f" % corrs[i, j], ha="center", va="center", color="k"
            )


correlation_matrix = np.corrcoef(Xycombo.T)

correlationmap(correlation_matrix, Xycolnames)
