from pickle import FALSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from scipy.stats import iqr

plt.style.use("ggplot")

# Step 1: Load Data and Prepare Numeric Data
Train = pd.read_csv(
    "/Users/vignesh/RStudio/IDA_Homework/IDA Assignemnt 6/2024-ise-dsa-5103-ida-hw-6/Train.csv/Train.csv",
    encoding="ISO-8859-1",
)
Train_Numeric = Train.select_dtypes(include=[np.number])


class SummaryStatistics:
    """
    Class to compute and manage summary statistics for a DataFrame.
    """

    @staticmethod
    def Q1(x):
        return np.percentile(x.dropna(), 25)

    @staticmethod
    def Q3(x):
        return np.percentile(x.dropna(), 75)

    @staticmethod
    def compute_summary(x):
        return [
            len(x),
            x.nunique(),
            x.isna().sum(),
            x.mean(skipna=True),
            x.min(skipna=True),
            SummaryStatistics.Q1(x),
            x.median(skipna=True),
            SummaryStatistics.Q3(x),
            x.max(skipna=True),
            x.std(skipna=True),
        ]

    def generate_summary(self, dataframe):
        summary = pd.DataFrame(
            {col: self.compute_summary(dataframe[col]) for col in dataframe.columns}
        ).T
        summary.columns = [
            "n",
            "unique",
            "missing",
            "mean",
            "min",
            "Q1",
            "median",
            "Q3",
            "max",
            "sd",
        ]
        summary["missing_pct"] = 100 * summary["missing"] / summary["n"]
        summary["unique_pct"] = 100 * summary["unique"] / summary["n"]
        summary.reset_index(inplace=True)
        summary.rename(columns={"index": "variable"}, inplace=True)
        return summary


class DataPreprocessor:
    """
    Class to handle preprocessing steps such as removing unwanted columns and imputing missing values.
    """

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def remove_columns(self, columns):
        self.dataframe.drop(columns=columns, errors="ignore", inplace=True)
        return self.dataframe

    def impute_missing_values(self, columns, n_neighbors=5):
        imputer = KNNImputer(n_neighbors=n_neighbors)
        self.dataframe[columns] = imputer.fit_transform(self.dataframe[columns])
        return self.dataframe


# Step 2: Compute summary statistics
summary_stats = SummaryStatistics()
numeric_summary = summary_stats.generate_summary(Train_Numeric)

# Step 3: Remove unwanted columns
preprocessor = DataPreprocessor(Train_Numeric)
Train_Numeric = preprocessor.remove_columns(
    ["adwordsClickInfo.page", "bounces", "newVisits", "sessionId"]
)

# Recompute summary after removing columns
numeric_summary = summary_stats.generate_summary(Train_Numeric)

# Step 4: Impute missing values
Train_Numeric = preprocessor.impute_missing_values(["pageviews"])

# Recompute summary after imputation
numeric_summary = summary_stats.generate_summary(Train_Numeric)

# Display final numeric summary
print(numeric_summary)


class DataVisualizer:
    """
    A class to visualize data using histograms and scatter plots.
    """

    def __init__(self, dataframe):
        """
        Initializes the DataVisualizer with the given DataFrame.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame containing the data to be visualized.
        """
        self.dataframe = dataframe

    def plot_histogram(self, column_data, column_name):
        """
        Plots a histogram for a single column.

        Parameters:
        - column_data (pd.Series): The column data to plot.
        - column_name (str): The name of the column.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(column_data.dropna(), bins=30)
        plt.title(f"Histogram of {column_name}")
        plt.xlabel(column_name)
        plt.ylabel("Frequency")
        plt.show()

    def plot_all_histograms(self):
        """
        Plots histograms for all numeric columns in the DataFrame.
        """
        for col in self.dataframe.columns:
            self.plot_histogram(self.dataframe[col], col)

    def plot_scatter(self, x_col, y_col, log_y=False, exclude_index=None, alpha=0.5):
        """
        Plots a scatter plot for the given columns.

        Parameters:
        - x_col (str): The column for the x-axis.
        - y_col (str): The column for the y-axis.
        - log_y (bool): Whether to apply a log transformation to the y-axis.
        - exclude_index (list): List of row indices to exclude from the plot.
        - alpha (float): Transparency level for the scatter plot.
        """
        data = self.dataframe
        if exclude_index is not None:
            data = data.drop(index=exclude_index)

        y_data = np.log1p(data[y_col]) if log_y else data[y_col]
        sns.scatterplot(x=x_col, y=y_data, data=data, alpha=alpha)
        log_label = " (Log Revenue)" if log_y else ""
        plt.title(f"Scatter Plot of {y_col} vs {x_col}{log_label}")
        plt.xlabel(x_col)
        plt.ylabel(f"{y_col}{log_label}")
        plt.show()

    def plot_log_revenue_histogram(self, column_name):
        """
        Plots a histogram for the log-transformed revenue column.

        Parameters:
        - column_name (str): The name of the revenue column.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(np.log1p(self.dataframe[column_name].dropna()), bins=30)
        plt.title("Histogram of Log Revenue")
        plt.xlabel("Log Revenue")
        plt.ylabel("Frequency")
        plt.show()

    def run_all_visualizations(self):
        """
        Runs all visualizations including histograms, scatter plots, and log-revenue analysis.
        """
        # Plot all histograms
        self.plot_all_histograms()

        # Scatter plots
        self.plot_scatter("visitStartTime", "revenue", alpha=0.5)
        self.plot_scatter("pageviews", "revenue", exclude_index=[39511], alpha=0.5)
        self.plot_scatter(
            "timeSinceLastVisit", "revenue", exclude_index=[39511], alpha=0.5
        )
        self.plot_scatter("visitNumber", "revenue", exclude_index=[39511], alpha=0.5)

        # Log-transformed scatter plots
        self.plot_scatter("visitStartTime", "revenue", log_y=True, alpha=0.5)
        self.plot_scatter(
            "pageviews", "revenue", log_y=True, exclude_index=[39511], alpha=0.5
        )
        self.plot_scatter(
            "timeSinceLastVisit",
            "revenue",
            log_y=True,
            exclude_index=[39511],
            alpha=0.5,
        )
        self.plot_scatter(
            "visitNumber", "revenue", log_y=True, exclude_index=[39511], alpha=0.5
        )

        # Log-revenue histogram
        self.plot_log_revenue_histogram("revenue")


# Example usage
visualizer = DataVisualizer(Train_Numeric)
visualizer.run_all_visualizations()
