# Essential Libraries
import numpy as np
import pandas as pd

# Preprocessing Libraries
from sklearn.preprocessing import PowerTransformer, StandardScaler, FunctionTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from scipy.stats import mode

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

# Specialized ML Libraries
from xgboost import XGBClassifier
from sklearn.utils import parallel_backend
from sklearn import metrics

plt.style.use("default")

# ====================================================
# Loading and Preparing the Train Dataset
# ====================================================

# Step 1: Load Data
file_path = "/Users/vignesh/RStudio/IDA_Homework/IDA_HW_7/2024-dsa-ise-ida-classification-hw-7/hm7-Train-2024.csv"
train = pd.read_csv(file_path)

# View data structure
train.info()

# Check for missing values
train.isnull().sum()

# Check for duplicate rows
duplicates = train[train.duplicated()]
print(f"Number of duplicate rows: {len(duplicates)}")

# ====================================================
# Converting Selected Variables to Object Type
# ====================================================


# Convert specified columns to object (string) type
cols_to_convert_to_string = [
    "admission_type",
    "discharge_disposition",
    "admission_source",
    "readmitted",
]
train[cols_to_convert_to_string] = (
    train[cols_to_convert_to_string].fillna("").astype(str)
)

train.info()

# ====================================================
# Selecting Numeric Data and Computing Summary Statistics
# ====================================================

# Select numeric data
train_numeric = train.select_dtypes(include=[np.number])


# Function to compute summary statistics
def numeric_summary(df):
    if len(df) == 0:
        raise ValueError("Input DataFrame is empty.")

    summary = pd.DataFrame()
    summary["missing"] = df.isna().sum()
    summary["missing_pct"] = (summary["missing"] / len(df)) * 100
    summary["unique"] = df.nunique()
    summary["unique_pct"] = (summary["unique"] / len(df)) * 100
    descriptive_stats = df.describe().T
    summary = pd.concat([summary, descriptive_stats], axis=1)
    return summary


numeric_summary_df = numeric_summary(train_numeric)
numeric_summary_df


# ====================================================
# Plotting Histograms of Numeric Variables
# ====================================================


# Plot Histograms
def plot_histogram(data, column_name):
    plt.figure(figsize=(10, 5))
    sns.histplot(data[column_name], kde=True, bins=30)
    plt.title(f"Histogram of {column_name}")
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.show()


for col in train_numeric.columns:
    plot_histogram(train_numeric, col)


# Plot Boxplots of all variable
def plot_multiple_boxplots(data):
    # Convert the DataFrame to long format if needed
    data_long = data.melt(var_name="Variable", value_name="Value")
    plt.figure(figsize=(10, 5))
    sns.boxplot(x="Variable", y="Value", data=data_long)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.title("Boxplots of Multiple Variables")
    plt.xlabel("Variable")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()


plot_multiple_boxplots(train_numeric)


# Plot Boxplots
def plot_individual_boxplots(data):
    for column in data.columns:
        plt.figure(figsize=(10, 5))
        sns.boxplot(y=data[column])
        plt.title(f"Boxplot of {column}")
        plt.ylabel(column)
        plt.show()


# Call the function
plot_individual_boxplots(train_numeric)


# Plot histograms and Boxplots usig subplots
def plot_combined_histogram_boxplot(data):
    for column in data.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Histogram
        sns.histplot(data[column], kde=True, bins=30, ax=axes[0])
        axes[0].set_title(f"Histogram of {column}")
        axes[0].set_xlabel(column)
        axes[0].set_ylabel("Frequency")

        # Boxplot
        sns.boxplot(y=data[column], ax=axes[1])
        axes[1].set_title(f"Boxplot of {column}")
        axes[1].set_ylabel(column)

        plt.tight_layout()
        plt.show()


# Call the function
plot_combined_histogram_boxplot(train_numeric)

# ====================================================
# Outlier Detection and Removal
# ====================================================


# Remove Outlier using IQR
def remove_outliers_iqr_columnwise(data, multiplier=1.5):
    filtered_data = data.copy()
    for column in data.select_dtypes(
        include=["number"]
    ).columns:  # Apply only to numeric columns
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (multiplier * IQR)
        upper_bound = Q3 + (multiplier * IQR)
        # Keep only values within bounds for this column
        filtered_data = filtered_data[
            (filtered_data[column] >= lower_bound)
            & (filtered_data[column] <= upper_bound)
        ]
    return filtered_data


def set_outliers_to_nan_iqr(data, multiplier=1.5):
    filtered_data = data.copy()
    for column in data.select_dtypes(
        include=["number"]
    ).columns:  # Apply only to numeric columns
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (multiplier * IQR)
        upper_bound = Q3 + (multiplier * IQR)
        # Set outliers to NaN
        filtered_data.loc[
            (data[column] < lower_bound) | (data[column] > upper_bound), column
        ] = np.nan
    return filtered_data


cleaned_data = set_outliers_to_nan_iqr(train_numeric)


def plot_combined_histogram_boxplot(data):
    for column in data.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Histogram
        sns.histplot(data[column], kde=True, bins=30, ax=axes[0])
        axes[0].set_title(f"Histogram of {column}")
        axes[0].set_xlabel(column)
        axes[0].set_ylabel("Frequency")

        # Boxplot
        sns.boxplot(y=data[column], ax=axes[1])
        axes[1].set_title(f"Boxplot of {column}")
        axes[1].set_ylabel(column)

        plt.tight_layout()
        plt.show()


plot_combined_histogram_boxplot(cleaned_data)

cleaned_data.columns
cleaned_data.drop(
    columns=[
        "patientID",
        "number_inpatient",
        "number_emergency",
        "number_outpatient",
    ],
    inplace=True,
)

cleaned_data.columns.to_list()

numeric_summary(cleaned_data)

plot_combined_histogram_boxplot(cleaned_data)

# ====================================================
# Imputing Missing Values using KNN Imputation
# ====================================================

imputer = KNNImputer(n_neighbors=5)
train_numeric_imputed = pd.DataFrame(
    imputer.fit_transform(cleaned_data), columns=cleaned_data.columns
)

# Recompute summary statistics after imputation
numeric_summary(train_numeric_imputed)


plot_combined_histogram_boxplot(train_numeric_imputed)

# ====================================================
# Handling Skewness Using Box-Cox Transformation
# ====================================================


def compute_skewness(df):
    return df.apply(lambda x: x.skew()).sort_values()


skewness_values = compute_skewness(train_numeric_imputed)
print("Skewness of variables:")
print(skewness_values)

# ====================================================
# Numeric Transformation(Normalaization) and Skewness Summary
# ====================================================

# Apply Box-Cox transformation (requires positive data, so shift by 1 if necessary)
power_transformer = PowerTransformer(method="yeo-johnson", standardize=False)
transformed_data = power_transformer.fit_transform(train_numeric_imputed)
train_numeric_transformed = pd.DataFrame(
    transformed_data, columns=train_numeric_imputed.columns
)

# Recompute skewness after transformation
transformed_skewness = compute_skewness(train_numeric_transformed)
print("Skewness after transformation:")
print(transformed_skewness)

plot_combined_histogram_boxplot(train_numeric_transformed)

# ====================================================
# Numeric Scaling
# ====================================================

scaler = StandardScaler()

# Fit on transformed training data and scale it
train_numeric_scaled = pd.DataFrame(
    scaler.fit_transform(train_numeric_transformed),
    columns=train_numeric_transformed.columns,
)

# ====================================================
# FACTORS
# ====================================================

# ====================================================
# Selecting Factor Data, Computing Summary Statistics
# ====================================================

# Extract Factor Variables
train_factor = train.select_dtypes(include=["object"]).copy()
train_factor.info()
train_factor.describe().T.drop(columns="unique")


# Function to compute summary statistics
def factor_summary_2(df):
    summary = pd.DataFrame()
    summary["missing"] = df.isnull().sum()
    summary["missing_pct"] = (summary["missing"] / len(df)) * 100
    summary["unique"] = df.nunique()
    summary["unique_pct"] = (summary["unique"] / len(df)) * 100
    descriptive_stats = df.describe().T.drop(columns="unique")
    summary = pd.concat([summary, descriptive_stats], axis=1)
    return summary


# Compute Summary for Factor Data
def factor_summary(df):
    """
    Computes a summary of factor data for each column in the DataFrame.
    """
    summary = []
    for col in df.columns:
        col_data = df[col]
        value_counts = col_data.value_counts()
        unique_count = len(value_counts)
        total_count = len(col_data)
        missing_count = col_data.isnull().sum()
        unique_count = col_data.nunique()

        # Calculate modes and their counts
        most_common = value_counts.idxmax() if not value_counts.empty else np.nan
        most_common_count = value_counts.max() if not value_counts.empty else np.nan
        second_most_common = value_counts.index[1] if unique_count > 1 else np.nan
        second_most_common_count = value_counts.iloc[1] if unique_count > 1 else np.nan
        least_common = value_counts.idxmin() if not value_counts.empty else np.nan
        least_common_count = value_counts.min() if not value_counts.empty else np.nan

        summary.append(
            {
                "variable": col,
                "n": len(col_data),
                "missing": col_data.isnull().sum(),
                "missing_percentage": (missing_count / total_count * 100)
                if total_count > 0
                else np.nan,
                "unique": col_data.nunique(),
                "unique_percentage": (unique_count / total_count * 100)
                if total_count > 0
                else np.nan,
                "most_common": most_common,
                "most_common_count": most_common_count,
                "2nd_most_common": second_most_common,
                "2nd_most_common_count": second_most_common_count,
                "least_common": least_common,
                "least_common_count": least_common_count,
            }
        )
    return pd.DataFrame(summary)


factor_summary(train_factor)


# ====================================================
# Plotting Factor Variables with Barplots
# ====================================================


def plot_barplot(data, column_name):
    plt.figure(figsize=(8, 6))
    sns.countplot(
        data=data, x=column_name, order=data[column_name].value_counts().index
    )
    plt.title(f"Barplot of {column_name}")
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


for col in train_factor.columns:
    plot_barplot(train_factor, col)


def plot_barplot_with_hue(data, column):
    plt.figure(figsize=(20, 5))
    sns.countplot(data=data, x=column, hue="readmitted")
    plt.title(f"Distribution of {column} by Readmission Status", fontsize=16)
    plt.xlabel(column, fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=12)
    plt.show()


# Iterate over all factor variables and plot them against readmitted
for col in train_factor.columns:
    plot_barplot_with_hue(train_factor, col)

train_factor

# ====================================================
# Imputing Missing Factor Data
# ====================================================

train_factor_no_target = train_factor.drop("readmitted", axis=1)

# Create and apply the imputer on the remaining columns
factor_imputer = SimpleImputer(strategy="most_frequent")
train_factor_imputed = pd.DataFrame(
    factor_imputer.fit_transform(train_factor_no_target),
    columns=train_factor_no_target.columns,
)

# Optionally, add back the target variable if needed
train_factor_imputed["readmitted"] = train_factor["readmitted"]

# Recompute Summary of Factor Data After Imputation
factor_summary_df_imputed = factor_summary(train_factor_imputed)
factor_summary_df_imputed

# ====================================================
# Feature Selection of Factor Variables
# ====================================================

# Select Specific Factor Features
train_factor_selected = train_factor_imputed[["medical_specialty", "readmitted"]]
train_factor_selected.info()
factor_summary(train_factor_selected)

# ====================================================
# Collapsing Factor Levels
# ====================================================


def lump_top_n(series, n):
    """
    Groups all but the top `n` most frequent categories in a Pandas Series into an 'Other' category.

    Parameters:
        series (pd.Series): The series containing categorical data.
        n (int): Number of top categories to retain.

    Returns:
        pd.Series: Modified series with the top `n` categories retained and others lumped into 'Other'.
    """
    value_counts = series.value_counts()
    top_n_categories = value_counts.nlargest(n).index
    return series.apply(lambda x: x if x in top_n_categories else "Other")


train_factor_collapsed = train_factor_selected.copy()
train_factor_collapsed["medical_specialty"] = lump_top_n(
    train_factor_collapsed["medical_specialty"], n=19
)
train_factor_collapsed["medical_specialty"].value_counts()

# ====================================================
# Plotting Proportions by `readmitted`
# ====================================================


train_factor_collapsed.groupby(["medical_specialty", "readmitted"]).size().unstack(
    fill_value=0
).apply(lambda x: x / x.sum(), axis=1)


def plot_proportion_by_factor(data, column_name, target_name):
    prop_df = (
        data.groupby([column_name, target_name])
        .size()
        .unstack(fill_value=0)
        .apply(lambda x: x / x.sum(), axis=1)
    )
    prop_df.plot(kind="barh", stacked=True, figsize=(10, 6))
    plt.title(f"Proportion of {target_name} by {column_name}")
    plt.xlabel(column_name)
    plt.ylabel("Proportion")
    plt.legend(title=target_name)
    plt.tight_layout()
    plt.show()


for col in ["medical_specialty"]:
    plot_proportion_by_factor(train_factor_collapsed, col, "readmitted")

# ====================================================
# Merging Numeric and Factor Data
# ====================================================

final_data = pd.concat([train_numeric_scaled, train_factor_collapsed], axis=1)

# Checking the Final Data
final_data.info()
final_data.isnull().sum()
final_data
final_data.select_dtypes(include=["object"]).columns

print(final_data["medical_specialty"].nunique())
print(final_data["medical_specialty"].value_counts())

final_data = pd.get_dummies(final_data, columns=["medical_specialty"], drop_first=True)

final_data["readmitted"].value_counts()
final_data["readmitted"].value_counts(normalize=True)
final_data["readmitted"] = final_data["readmitted"].astype(int)
final_data["readmitted"].unique()


"""
# ====================================================
# UnderSampling
# ====================================================

# Separate the two classes
class_0 = final_data[final_data["readmitted"] == 0]
class_1 = final_data[final_data["readmitted"] == 1]

# Randomly undersample the majority class
class_0_downsampled = class_0.sample(n=len(class_1), random_state=42)

# Combine the two classes
final_data_balanced = pd.concat([class_0_downsampled, class_1])

# Shuffle the dataset
final_data_balanced = final_data_balanced.sample(frac=1, random_state=42).reset_index(
    drop=True
)

# ====================================================
# OverSampling
# ====================================================

class_0 = final_data[final_data["readmitted"] == 0]
class_1 = final_data[final_data["readmitted"] == 1]

# Randomly oversample the minority class
class_1_upsampled = class_1.sample(n=len(class_0), replace=True, random_state=42)

# Combine the two classes
final_data_balanced = pd.concat([class_0, class_1_upsampled])

# Shuffle the dataset
final_data_balanced = final_data_balanced.sample(frac=1, random_state=42).reset_index(
    drop=True
)
"""


# ====================================================
# Synthetic Oversampling
# ====================================================

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Separate features (X) and target (y)
X = final_data.drop(columns=["readmitted"])
y = final_data["readmitted"]

X.info()

# Apply SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# Combine X and y into a new balanced dataset
final_data_balanced_smote = pd.concat(
    [
        pd.DataFrame(X_smote, columns=X.columns),
        pd.DataFrame(y_smote, columns=["readmitted"]),
    ],
    axis=1,
)

# Verify the balance
print(final_data_balanced_smote["readmitted"].value_counts(normalize=True))


Final_Data = final_data_balanced_smote.copy()
Final_Data.columns.to_list()


# ====================================================
# Pipelines
# ====================================================

"""
# Define numeric and categorical features
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Preprocessing for numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create the full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
"""

# ====================================================
# Modelling
# ====================================================

# Define target variable and features
X = Final_Data.drop(columns=["readmitted"])
y = Final_Data["readmitted"]

# ====================================================
# Logistic Regression
# ====================================================

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Generate predictions and predicted probabilities
y_pred = model.predict(X)
y_pred_proba = model.predict_proba(X)[:, 1]

# Print classification report and confusion matrix
print("Classification Report:")
print(classification_report(y, y_pred))

cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:")
print(cm)


# Visualizing confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix after SMOTE")
plt.show()

# Compute ROC curve and ROC AUC score
fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
roc_auc = roc_auc_score(y, y_pred_proba)

# Plot the ROC curve
plt.figure(figsize=(10, 5))
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")  # Diagonal line for random performance
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.show()

# Print variable importance (i.e. coefficients)
importance = pd.DataFrame(
    {"Feature": X.columns, "Coefficient": model.coef_[0]}
).sort_values(by="Coefficient", key=abs, ascending=False)

print("Variable Importance:")
print(importance)

# ====================================================
# Logistic regression Hyperparamter tuning
# ====================================================

from scipy.stats import loguniform

param_grid = {"C": loguniform(0.001, 100), "penalty": ["l1", "l2"]}

# Initialize the logistic regression model with a solver that supports both L1 and L2 penalties
lr = LogisticRegression(max_iter=1000, solver="liblinear")

# Set up the GridSearchCV to tune hyperparameters based on ROC AUC score using 5-fold CV
random_search = RandomizedSearchCV(lr, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
random_search.fit(X, y)

# Print the best hyperparameters and best ROC AUC score from cross-validation
print("Best Hyperparameters:")
print(random_search.best_params_)
print("Best ROC AUC (CV):", random_search.best_score_)

# Use the best estimator for further evaluation
best_model = random_search.best_estimator_

# Generate predictions and predicted probabilities using the best model
y_pred = best_model.predict(X)
y_pred_proba = best_model.predict_proba(X)[:, 1]

# Print classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y, y_pred))

cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:")
print(cm)


# Compute ROC curve and ROC AUC score
fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
roc_auc = roc_auc_score(y, y_pred_proba)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")  # Diagonal line for random performance
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.show()

# Print variable importance (i.e., the coefficients of the logistic regression model)
importance = pd.DataFrame(
    {"Feature": X.columns, "Coefficient": best_model.coef_[0]}
).sort_values(by="Coefficient", key=abs, ascending=False)

print("Variable Importance:")
print(importance)

###############################################
# Hyperparameter Tuning for -
# LogisticRegression,
# RandomForest,
# SVC,
# KNeighborsClassifier,
# AdaBoostClassifier
# GradientBoostingClassifier
###############################################


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import numpy as np


# Define hyperparameter distributions for different models
param_grids = {
    "LogisticRegression": {
        "C": np.logspace(-3, 3, 20),
        "penalty": ["elasticnet"],
        "l1_ratio": np.linspace(0, 1, 20),
        "solver": ["saga"],
    },
    "RandomForest": {
        "n_estimators": np.arange(50, 500, 50),
        "max_depth": [None, 10, 20, 30, 40, 50],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    },
    # "SVC": {
    #     "C": np.logspace(-3, 3, 20),
    #     "kernel": ["linear", "rbf", "poly", "sigmoid"],
    #     "gamma": ["scale", "auto"],
    #     "degree": np.arange(1, 5),
    # },
    "KNeighborsClassifier": {
        "n_neighbors": np.arange(3, 30, 2),
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": np.arange(20, 100, 10),
        "p": [1, 2],  # 1 for Manhattan distance, 2 for Euclidean distance
    },
    "AdaBoostClassifier": {
        "n_estimators": np.arange(50, 500, 50),
        "learning_rate": [0.001, 0.01, 0.1, 1],
    },
    "GradientBoostingClassifier": {
        "n_estimators": np.arange(50, 500, 50),
        "learning_rate": [0.001, 0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7, 9],
        "subsample": [0.8, 0.9, 1.0],
    },
}

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=100, solver="saga"),
    "RandomForest": RandomForestClassifier(),
    # "SVC": SVC(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "AdaBoostClassifier": AdaBoostClassifier(),
    "GradientBoostingClassifier": GradientBoostingClassifier(),
}

best_params = {}
best_logloss_scores = {}

# Initialize a dictionary to store the logloss scores for each model and fold
fold_logloss_scores = {model_name: [] for model_name in models}

for model_name, model in models.items():
    print(f"Running RandomizedSearchCV for {model_name}...")

    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_grids[model_name],
        n_iter=10,  # Number of random combinations to try
        cv=5,
        scoring="neg_log_loss",  # Using negative log loss
        n_jobs=-1,
        random_state=42,
        return_train_score=False,  # We don't need training scores
    )

    random_search.fit(X, y)

    # Convert negative log loss to positive log loss values for clarity
    best_params[model_name] = random_search.best_params_
    best_logloss_scores[model_name] = -random_search.best_score_

    # Convert fold scores to positive values
    fold_logloss_scores[model_name] = -np.array(
        random_search.cv_results_["mean_test_score"]
    )

    print(f"Best parameters for {model_name}: {random_search.best_params_}")
    print(f"Best Logloss Score for {model_name}: {-random_search.best_score_}")
    print("-" * 50)

###############################################
# Print final best parameters and Logloss scores for all models
###############################################

print("Best hyperparameters and Logloss scores for each model:")
for model_name, params in best_params.items():
    print(f"{model_name}: {params} with Logloss = {best_logloss_scores[model_name]}")

###############################################
# Plot the Logloss scores for each fold for each model
###############################################

fig, ax = plt.subplots(figsize=(15, 8))

for model_name, logloss_scores in fold_logloss_scores.items():
    ax.plot(range(1, len(logloss_scores) + 1), logloss_scores, label=model_name)

ax.set_title("Logloss Scores for Each Fold (RandomizedSearchCV)")
ax.set_xlabel("Fold Number")
ax.set_ylabel("Logloss Score")
ax.legend()
fig.tight_layout()

plt.show()

###############################################
# Plotting the Logloss Scores for Each Model
###############################################

# Data for plotting
model_names = list(best_logloss_scores.keys())
logloss_scores = list(best_logloss_scores.values())

plt.figure(figsize=(10, 6))

# Using a colormap to create a gradient of colors
colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

plt.barh(model_names, logloss_scores, color=colors)
plt.xlabel("Logloss Score")
plt.title("Logloss Scores for Different Models")
plt.show()
##############################################################################################

param_grid = {
    "n_estimators": np.arange(50, 500, 50),
    "max_depth": [None, 10, 20, 30, 40, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
}

# Define the RandomForest model
rf_model = RandomForestClassifier()

print("Running RandomizedSearchCV for RandomForestClassifier...")

random_search = RandomizedSearchCV(
    rf_model,
    param_distributions=param_grid,
    n_iter=10,  # Number of random combinations to try
    cv=5,
    scoring="neg_log_loss",  # Using negative log loss
    n_jobs=-1,
    random_state=42,
    return_train_score=False,
)

# Fit the RandomizedSearchCV
random_search.fit(X, y)

# Retrieve best parameters and logloss score (converted to positive values)
best_params_rf = random_search.best_params_
best_logloss_rf = -random_search.best_score_
fold_logloss_scores_rf = -np.array(random_search.cv_results_["mean_test_score"])

print(f"Best parameters for RandomForestClassifier: {best_params_rf}")
print(f"Best Logloss Score for RandomForestClassifier: {best_logloss_rf}")
print("-" * 50)

###############################################
# Plot the Logloss scores for each fold for RandomForest
###############################################

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    range(1, len(fold_logloss_scores_rf) + 1),
    fold_logloss_scores_rf,
    marker="o",
    linestyle="-",
)
ax.set_title("RandomForest Logloss Scores for Each Fold (RandomizedSearchCV)")
ax.set_xlabel("Fold Number")
ax.set_ylabel("Logloss Score")
plt.show()

###############################################
# Plotting the Best Logloss Score for RandomForest
###############################################

plt.figure(figsize=(6, 4))
plt.barh(["RandomForest"], [best_logloss_rf])
plt.xlabel("Logloss Score")
plt.title("RandomForest Best Logloss Score")
plt.show()

###############################################
#
###############################################

# Option 2: Instantiate a new RandomForestClassifier using the best parameters
final_rf_model = RandomForestClassifier(**best_params_rf)
final_rf_model.fit(X, y)


# ====================================================
# Loading and Preparing the TEST Dataset
# ====================================================

# Step 1: Load Data
file_path = "/Users/vignesh/RStudio/IDA_Homework/IDA_HW_7/2024-dsa-ise-ida-classification-hw-7/hm7-Test-2024.csv"
test = pd.read_csv(file_path)

# Convert specified columns to string (if they exist)
cols_to_convert = [
    "admission_type",
    "discharge_disposition",
    "admission_source",
    "readmitted",
]
for col in cols_to_convert:
    if col in test.columns:
        test[col] = test[col].fillna("").astype(str)

# ---------------------------
# Process Numeric Data
# ---------------------------
# Select numeric columns
test_numeric = test.select_dtypes(include=[np.number]).copy()

# Handle outliers using the same function as in training (set values outside IQR bounds to NaN)
test_numeric_cleaned = set_outliers_to_nan_iqr(test_numeric)

# Drop columns that were dropped during training
cols_to_drop = [
    "patientID",
    "number_inpatient",
    "number_emergency",
    "number_outpatient",
]
test_numeric_cleaned.drop(
    columns=[col for col in cols_to_drop if col in test_numeric_cleaned.columns],
    inplace=True,
)

# Use the already fitted KNNImputer from training to impute missing numeric values
test_numeric_imputed = pd.DataFrame(
    imputer.transform(test_numeric_cleaned), columns=test_numeric_cleaned.columns
)

# Apply the already fitted PowerTransformer for normalization
test_numeric_transformed = pd.DataFrame(
    power_transformer.transform(test_numeric_imputed),
    columns=test_numeric_imputed.columns,
)

# Scale the numeric data using the already fitted StandardScaler
test_numeric_scaled = pd.DataFrame(
    scaler.transform(test_numeric_transformed), columns=test_numeric_transformed.columns
)

# ---------------------------
# Process Factor (Categorical) Data
# ---------------------------
# Select categorical columns
test_factor = test.select_dtypes(include=["object"]).copy()

# Use the already fitted SimpleImputer for factor data
test_factor_imputed = pd.DataFrame(
    factor_imputer.transform(test_factor), columns=test_factor.columns
)

# Drop the target column if present, as we are predicting it
if "readmitted" in test_factor_imputed.columns:
    test_factor_imputed.drop(columns=["readmitted"], inplace=True)

# For consistency with training, focus on the 'medical_specialty' variable
if "medical_specialty" in test_factor_imputed.columns:
    test_factor_selected = test_factor_imputed[["medical_specialty"]].copy()
    # Use the same lumping function to collapse levels into top 19 categories
    test_factor_selected["medical_specialty"] = lump_top_n(
        test_factor_selected["medical_specialty"], n=19
    )

    # Convert categorical data to dummy variables matching the training encoding
    test_factor_dummies = pd.get_dummies(
        test_factor_selected["medical_specialty"],
        prefix="medical_specialty",
        drop_first=True,
    )
else:
    test_factor_dummies = pd.DataFrame()

# ---------------------------
# Merge Processed Numeric and Factor Data
# ---------------------------
# Combine numeric and factor features
test_final = pd.concat([test_numeric_scaled, test_factor_dummies], axis=1)

# Align test feature columns with those used during training.
# Assume X (from training) was defined as:
#   X = Final_Data.drop(columns=["readmitted"])
train_feature_cols = X.columns  # from training

# Add any missing columns (if a category did not appear in test) with 0s
for col in train_feature_cols:
    if col not in test_final.columns:
        test_final[col] = 0
# Ensure the test data has the same column order as training
test_final = test_final[train_feature_cols]

# ---------------------------
# Generate Predictions Using the Fitted Model
# ---------------------------
# Use the best tuned model (best_model) from training to predict on test data
y_test_pred = final_rf_model.predict(test_final)
y_test_pred_proba = final_rf_model.predict_proba(test_final)[:, 1]

# Output predictions
print("Test Predictions:")
print(y_test_pred)
print("\nTest Predicted Probabilities:")
print(y_test_pred_proba)
