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

plt.style.use("ggplot")

# #################################################
# Loading and Preparing the Train Dataset
# #################################################

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

# #################################################
# Converting Selected Variables to Object Type
# #################################################


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

# #################################################
# Selecting Numeric Data and Computing Summary Statistics
# #################################################

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


# #################################################
# Plotting Histograms of Numeric Variables
# #################################################


# Plot Histograms
def plot_histogram(data, column_name):
    plt.figure(figsize=(6, 6))
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
    plt.figure(figsize=(6, 6))
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
        plt.figure(figsize=(6, 6))
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

# #################################################
# Outlier Detection and Removal
# #################################################


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

numeric_summary(cleaned_data)

plot_combined_histogram_boxplot(cleaned_data)

# #################################################
# Imputing Missing Values using KNN Imputation
# #################################################

imputer = KNNImputer(n_neighbors=5)
train_numeric_imputed = pd.DataFrame(
    imputer.fit_transform(cleaned_data), columns=cleaned_data.columns
)

# Recompute summary statistics after imputation
numeric_summary(train_numeric_imputed)


plot_combined_histogram_boxplot(train_numeric_imputed)

# #################################################
# Handling Skewness Using Box-Cox Transformation
# #################################################


def compute_skewness(df):
    return df.apply(lambda x: x.skew()).sort_values()


skewness_values = compute_skewness(train_numeric_imputed)
print("Skewness of variables:")
print(skewness_values)

# #################################################
# Numeric Transformation(Normalaization) and Skewness Summary
# #################################################

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

# #################################################
# Numeric Scaling
# #################################################

scaler = StandardScaler()

# Fit on transformed training data and scale it
train_numeric_scaled = pd.DataFrame(
    scaler.fit_transform(train_numeric_transformed),
    columns=train_numeric_transformed.columns,
)

# #################################################
# FACTORS
# #################################################

# #################################################
# Selecting Factor Data, Computing Summary Statistics
# #################################################

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


# #################################################
# Plotting Factor Variables with Barplots
# #################################################


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

# #################################################
# Imputing Missing Factor Data
# #################################################

factor_imputer = SimpleImputer(strategy="most_frequent")
train_factor_imputed = pd.DataFrame(
    factor_imputer.fit_transform(train_factor), columns=train_factor.columns
)

# Recompute Summary of Factor Data After Imputation
factor_summary_df_imputed = factor_summary(train_factor_imputed)
factor_summary_df_imputed

# #################################################
# Feature Selection of Factor Variables
# #################################################

# Select Specific Factor Features
train_factor_selected = train_factor_imputed[["medical_specialty", "readmitted"]]
train_factor_selected.info()
factor_summary(train_factor_selected)

# #################################################
# Collapsing Factor Levels
# #################################################


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

# #################################################
# Plotting Proportions by `readmitted`
# #################################################


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

# #################################################
# Merging Numeric and Factor Data
# #################################################

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
# #################################################
# UnderSampling
# #################################################

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

# #################################################
# OverSampling
# #################################################

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


# #################################################
# Synthetic Oversampling
# #################################################

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


# #################################################
# Pipelines
# #################################################

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

# #################################################
# Modelling
# #################################################


# #################################################
# Load and Preprocess Data
# #################################################

# Define target variable and features

X = Final_Data.drop(columns=["readmitted"])
y = Final_Data["readmitted"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# Logistic Regression
logreg_model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)

from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
roc_auc_scores = cross_val_score(logreg_model, X, y, cv=cv, scoring="roc_auc")
roc_auc_scores.mean()

logreg_model.fit(X_train, y_train)
logreg_roc = roc_auc_score(y_test, logreg_model.predict_proba(X_test)[:, 1])

# Variable Importance for Logistic Regression
logreg_importance = np.abs(logreg_model.coef_[0])
logreg_importance_features = pd.DataFrame(
    {"Feature": X.columns, "Importance": logreg_importance}
).sort_values(by="Importance", ascending=False)


# Random Forest with Grid Search
rf_param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
}
rf_grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42), rf_param_grid, cv=5, scoring="roc_auc"
)
rf_grid_search.fit(X_train, y_train)
rf_model = rf_grid_search.best_estimator_
rf_roc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

# Variable Importance for Random Forest
rf_importance = pd.DataFrame(
    {"Feature": X.columns, "Importance": rf_model.feature_importances_}
).sort_values(by="Importance", ascending=False)


# #################################################


# #################################################


# #################################################

# Load data (assuming Final_Data is a pandas DataFrame)
# Replace this with actual data loading code
# Final_Data = pd.read_csv("your_data.csv")

# Preprocessing
target = "readmitted"
X = Final_Data.drop(columns=[target])
y = Final_Data[target].apply(lambda x: 1 if x == "Yes" else 0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression
logreg_model = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)
logreg_model.fit(X_train, y_train)
logreg_roc = roc_auc_score(y_test, logreg_model.predict_proba(X_test)[:, 1])

# Variable Importance for Logistic Regression
logreg_importance = np.abs(logreg_model.coef_[0])
logreg_importance_features = pd.DataFrame(
    {"Feature": X.columns, "Importance": logreg_importance}
).sort_values(by="Importance", ascending=False)

# Lasso Regression
lasso_model = LassoCV(cv=10, random_state=42).fit(X_train, y_train)
selected_features = np.where(lasso_model.coef_ != 0)[0]
lasso_importance = pd.DataFrame(
    {
        "Feature": X.columns[selected_features],
        "Importance": lasso_model.coef_[selected_features],
    }
).sort_values(by="Importance", ascending=False)

# MARS using py-earth
from pyearth import Earth

mars_model = Earth(max_degree=2).fit(X_train, y_train)
mars_roc = roc_auc_score(y_test, mars_model.predict(X_test))

# Variable Importance for MARS
mars_importance = pd.DataFrame(
    {"Feature": X.columns, "Importance": mars_model.feature_importances_}
).sort_values(by="Importance", ascending=False)

# Random Forest with Grid Search
rf_param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
}
rf_grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42), rf_param_grid, cv=5, scoring="roc_auc"
)
rf_grid_search.fit(X_train, y_train)
rf_model = rf_grid_search.best_estimator_
rf_roc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

# Variable Importance for Random Forest
rf_importance = pd.DataFrame(
    {"Feature": X.columns, "Importance": rf_model.feature_importances_}
).sort_values(by="Importance", ascending=False)

# XGBoost with Grid Search
xgb_param_grid = {
    "n_estimators": [100, 150, 200],
    "max_depth": [3, 6, 9],
    "learning_rate": [0.01, 0.1, 0.3],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}
xgb_grid_search = GridSearchCV(
    XGBClassifier(random_state=42), xgb_param_grid, cv=5, scoring="roc_auc"
)
xgb_grid_search.fit(X_train, y_train)
xgb_model = xgb_grid_search.best_estimator_
xgb_roc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])

# Variable Importance for XGBoost
xgb_importance = pd.DataFrame(
    {"Feature": X.columns, "Importance": xgb_model.feature_importances_}
).sort_values(by="Importance", ascending=False)

# Neural Network
nn_model = MLPClassifier(
    hidden_layer_sizes=(6, 7, 8), alpha=0.8, max_iter=1000, random_state=42
)
nn_model.fit(X_train, y_train)
nn_roc = roc_auc_score(y_test, nn_model.predict_proba(X_test)[:, 1])

# Model Performance Comparison
performance = {
    "Logistic Regression": logreg_roc,
    "Lasso": lasso_model.score(X_test, y_test),
    "MARS": mars_roc,
    "Random Forest": rf_roc,
    "XGBoost": xgb_roc,
    "Neural Network": nn_roc,
}

print("Model Performance (ROC AUC):")
for model, score in performance.items():
    print(f"{model}: {score:.4f}")


# Log Loss Calculation
def calculate_log_loss(model, X, y_true):
    y_pred = model.predict_proba(X)[:, 1]
    return log_loss(y_true, y_pred)


log_loss_results = {
    "Logistic Regression": calculate_log_loss(logreg_model, X_test, y_test),
    "Random Forest": calculate_log_loss(rf_model, X_test, y_test),
    "XGBoost": calculate_log_loss(xgb_model, X_test, y_test),
    "Neural Network": calculate_log_loss(nn_model, X_test, y_test),
}

print("\nLog Loss Results:")
for model, loss in log_loss_results.items():
    print(f"{model}: {loss:.4f}")

# Variable Importance Summary
print("\nVariable Importance:")
print("Logistic Regression:")
print(logreg_importance_features.head())

print("Lasso Regression:")
print(lasso_importance.head())

print("MARS:")
print(mars_importance.head())

print("Random Forest:")
print(rf_importance.head())

print("XGBoost:")
print(xgb_importance.head())
