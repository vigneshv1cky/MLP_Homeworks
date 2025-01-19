# Import required packages
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Default figure parameters
plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams["font.size"] = 12
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["figure.constrained_layout.use"] = True
plt.style.use("ggplot")


fname = "../hw2/saved_file.csv"

baby_data_raw = pd.read_csv(fname)
baby_data_raw.info()

baby_data_raw.describe()
baby_data_raw.columns.tolist()
baby_data_raw.isna().any()


""" 
Pipeline component object for computing the 
derivative for specified features
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

        for attrib in self.attribs:
            vals = Xout[attrib].values
            diff = vals[1:] - vals[0:-1]
            deriv = diff / self.dt
            deriv = np.append(deriv, 0)

            attrib_name = self.prefix + attrib
            Xout[attrib_name] = pd.Series(deriv)

        return Xout


""" 
Pipeline component object for selecting a subset of specified features
"""


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribs):
        self.attribs = attribs

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return X[self.attribs]


"""

Complete the Pipeline component object for interpolating and filling in 
gaps within the data. Whenever data are missing in between valid values, 
use interpolation to fill in the gaps. For example (for linear interpolation),
    1.2 NaN NaN 1.5 
becomes
    1.2 1.3 1.4 1.5 

Whenever data are missing on the edges of the data, fill in the gaps
with the first available valid value. For example,
    NaN NaN 2.3 3.6 3.2 NaN
becomes
    2.3 2.3 2.3 3.6 3.2 3.2
The transform() method you create must fill in the holes and the edge cases.

Hint: there are DataFrame methods that will help you implement these features
"""


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
        Xout = Xout = Xout.ffill().bfill()

        # Return the imputed dataframe
        return Xout


"""
Filtering

With our infant data, the sensors can give us somewhat noisy data, adding
an extra, high-frequency signal on top of our estimates of sensor positions.
One way to smooth out these high-frequency effects is to convolve our 
original signal with a smoothing kernel.  Here, we will use a Box Filter kernel.
This smoothing process reduces these noise effects, and generally improves 
subsequent analysis.  

With a Box Filter kernel, the value of output of the convolution at time t, 
x'[t], is an average of the x[]'s around t.  You will create a function
that returns this kernel.  

Then, you will finish the implementation of the general Filter class for 
smoothing a timeseries signal with a given kernel that can be of any odd length.
Here is the example formula for a filter of size k=7:
    x'[t] = ( w[0]*x[t-3] + w[1]*x[t-2] + w[2]*x[t-1] + w[3]*x[t]
           + w[4]*x[t+1] + w[5]*x[t+2] + w[6]*x[t+3])
                
This can be implemented similarly to how the derivative is computed, but will
require:
1. Padding both ends of x with k//2 copies of the adjacent
value.  This will mean that the length of the vector
after convolution is the same as the original length of x.
For example:
                1.3 2.1 4.4 4.1 3.2
would be padded as follows:

    1.3 1.3 1.3 1.3 2.1 4.4 4.1 3.2 3.2 3.2 3.2
    
Note that we are assuming that our kernels lengths are odd.

2. Iterating over the k filter elements, rather than iterating over the 
samples in x.  Remember that for loops with many iterations in python 
are very inefficient relative to the loops inside of the numpy methods
(so, like our derivative computer, we will rely on numpy to implement the
loops across the timeseries)

"""


def computeBoxWeights(length=3):
    # Equal weights for the box filter
    weights = np.ones(length) / length

    return weights


""" 
Complete the implementation of the general Filter class
"""


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


kernel_3 = computeBoxWeights(3)
kernel_5 = computeBoxWeights(5)
kernel_11 = computeBoxWeights(11)
kernel_17 = computeBoxWeights(17)

# Display the kernels
print("Kernel of length 3:", kernel_3)
print("Kernel of length 5:", kernel_5)
print("Kernel of length 11:", kernel_11)
print("Kernel of length 17:", kernel_17)


plt.figure(figsize=(6, 6))
plt.plot(kernel_3, label="Kernel Length 3", marker="o")
plt.plot(kernel_5, label="Kernel Length 5", marker="o")
plt.plot(kernel_11, label="Kernel Length 11", marker="o")
plt.plot(kernel_17, label="Kernel Length 17", marker="o")
plt.title("Box Filter Kernels")
plt.xlabel("Kernel Index")
plt.ylabel("Weight")
plt.legend()
plt.grid(True)
plt.show()

# Construct Pipelines

selected_names = ["left_wrist_z", "right_wrist_z"]
nselected = len(selected_names)

"""
Create a pipeline that:
1. Selects a subset of features specified above
2. Fills gaps within the data using cubic interpolation, and fills
   gaps at the edges of the data with the first or last valid value
3. Computes the derivatives of the selected features. The data are 
   sampled at 50 Hz, therefore, the elapsed time (dt) between 
   the samples is .02 seconds (dt=.02)
"""

from sklearn.pipeline import Pipeline

pipe1 = Pipeline(
    [
        ("selector", DataFrameSelector(selected_names)),
        ("interpolator", InterpolationImputer(method="linear")),
        ("derivative", DerivativeComputer(attribs=selected_names, dt=0.02)),
    ]
)

""" 
Create a pipeline that:
1. Selects a subset of features specified above
2. Fills gaps within the data using cubic interpolation, and fills
   gaps at the edges of the data with the first or last valid value
3. Smooths the data with a Box Filter. Use kernel size of 11
4. Compute the derivatives of the selected features. The data are 
   sampled at 50 Hz, therefore, the period or elapsed time (dt) between 
   the samples is .02 seconds (dt=.02)
"""
box_kernel = computeBoxWeights(11)

# Define the corrected second pipeline with the derivative computation step included
pipe2 = Pipeline(
    [
        ("selector", DataFrameSelector(selected_names)),  # Step 1: Select features
        (
            "interpolator",
            InterpolationImputer(method="cubic"),
        ),  # Step 2: Fill gaps using cubic interpolation
        (
            "smoother",
            Filter(attribs=selected_names, kernel=box_kernel),
        ),  # Step 3: Smooth data with a Box Filter of size 11
        (
            "derivative",
            DerivativeComputer(attribs=selected_names, dt=0.02),
        ),  # Step 4: Compute derivatives
    ]
)


"""

Use the appropriate pipeline elements to extract the selected 
raw kinematic data and time
"""
dfs = DataFrameSelector(selected_names)


baby_data0 = dfs.fit_transform(baby_data_raw)
timeselector = DataFrameSelector(["time"])
time_df = timeselector.fit_transform(baby_data_raw)

""" TODO

Fit both Pipelines to the data and transform the data
"""
baby_data1 = pipe1.fit_transform(baby_data_raw)
baby_data2 = pipe2.fit_transform(baby_data_raw)

baby_data1.head()

""" 
EXECUTE CELL

Display the summary statistics for the data
from both pipelines
"""
baby_data1.describe()


""" TODO
For each selected feature, construct plots comparing the raw data 
to the data from both pipelines. For each selected 
feature, create a figure displaying the raw data and the cleaned 
data in the same subplot. There should be three subplots per feature 
figure. Each subplot is in a separate row.
    subplot(1) will compare the original raw data to the pipeline1 
               pre-processed data.  Vertically offset the two curves
    subplot(2) will compare the original raw data to the pipeline2 
               pre-processed data.  Vertically offset the two curves
    subplot(3) will compare pipeline1 to pipeline2. DO NOT OFFSET
                THE TWO CURVES
    
    Visualize just seconds 138-142.

For all subplots, include axis labels, legends and titles.
"""

plt.style.use("ggplot")
import matplotlib.pyplot as plt

# Define the time window for visualization
time_window = (baby_data_raw["time"] >= 138) & (baby_data_raw["time"] <= 142)

# Select the data within the time window
time = baby_data_raw["time"][time_window]
Xsel_raw = baby_data_raw[selected_names][time_window]
Xsel_clean1 = baby_data1[selected_names][time_window]
Xsel_clean2 = baby_data2[selected_names][time_window]

# Define the x-axis limits
xlim = [time.min(), time.max()]

# Iterate over the features (left_wrist_z and right_wrist_z)
for f, fname in enumerate(selected_names):
    # Create a figure with 3 sub-panels
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    axs = axs.ravel()

    # Subplot 1: Raw data vs. Pipeline 1
    axs[0].plot(time, Xsel_raw[fname], label="Raw Data", color="blue")
    axs[0].plot(
        time, Xsel_clean1[fname] + 0.1, label="Pipeline 1", color="red"
    )  # Vertically offset by 0.1
    axs[0].set_xlim(xlim)
    axs[0].set_ylabel("Z-Coordinate")
    axs[0].set_title(f"{fname}: Raw Data vs. Pipeline 1")
    axs[0].legend()

    # Subplot 2: Raw data vs. Pipeline 2
    axs[1].plot(time, Xsel_raw[fname], label="Raw Data", color="blue")
    axs[1].plot(
        time, Xsel_clean2[fname] + 0.1, label="Pipeline 2", color="green"
    )  # Vertically offset by 0.1
    axs[1].set_xlim(xlim)
    axs[1].set_ylabel("Z-Coordinate")
    axs[1].set_title(f"{fname}: Raw Data vs. Pipeline 2")
    axs[1].legend()

    # Subplot 3: Pipeline 1 vs. Pipeline 2
    axs[2].plot(time, Xsel_clean1[fname], label="Pipeline 1", color="red")
    axs[2].plot(time, Xsel_clean2[fname], label="Pipeline 2", color="green")
    axs[2].set_xlim(xlim)
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Z-Coordinate")
    axs[2].set_title(f"{fname}: Pipeline 1 vs. Pipeline 2")
    axs[2].legend()

    plt.show()


# Time window for visualizing data
time_window = (baby_data_raw["time"] >= 138) & (baby_data_raw["time"] <= 142)

# Extract time and data for the selected window
time = baby_data_raw["time"][time_window]
selected_features = ["left_wrist_z", "right_wrist_z"]
derivatives = ["d_left_wrist_z", "d_right_wrist_z"]

# Prepare data from both pipelines for visualization
data1_features = baby_data1[selected_features][time_window]
data1_derivatives = baby_data1[derivatives][time_window]
data2_features = baby_data2[selected_features][time_window]
data2_derivatives = baby_data2[derivatives][time_window]

# Loop through each feature and its derivatives to create plots
for feature, deriv in zip(selected_features, derivatives):
    plt.figure(figsize=(10, 12))

    # Subplot 1: Feature and derivative from Pipeline 1
    plt.subplot(3, 1, 1)
    plt.plot(
        time, data1_features[feature], label=f"{feature} - Pipeline 1", color="blue"
    )
    plt.plot(
        time,
        data1_derivatives[deriv],
        label=f"Derivative of {feature} - Pipeline 1",
        color="red",
    )
    plt.title(f"Pipeline 1: {feature} and its Derivative")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.legend()

    # Subplot 2: Feature and derivative from Pipeline 2
    plt.subplot(3, 1, 2)
    plt.plot(
        time, data2_features[feature], label=f"{feature} - Pipeline 2", color="green"
    )
    plt.plot(
        time,
        data2_derivatives[deriv],
        label=f"Derivative of {feature} - Pipeline 2",
        color="purple",
    )
    plt.title(f"Pipeline 2: {feature} and its Derivative")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.legend()

    # Subplot 3: Comparison of derivatives from both pipelines
    plt.subplot(3, 1, 3)
    plt.plot(
        time, data1_derivatives[deriv], label=f"Derivative from Pipeline 1", color="red"
    )
    plt.plot(
        time,
        data2_derivatives[deriv],
        label=f"Derivative from Pipeline 2",
        color="purple",
    )
    plt.title(f"Comparison of Derivatives: Pipeline 1 vs Pipeline 2")
    plt.xlabel("Time (s)")
    plt.ylabel("Derivative Value")
    plt.legend()
    plt.show()
