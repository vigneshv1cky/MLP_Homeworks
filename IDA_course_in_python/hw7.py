import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from scipy.stats import mode

# #################################################
# Loading and Preparing the Train Dataset
# #################################################

# Step 1: Load Data
file_path = "/Users/vignesh/RStudio/IDA_Homework/IDA_HW_7/2024-dsa-ise-ida-classification-hw-7/hm7-Train-2024.csv"
train = pd.read_csv(file_path)

# View data structure
print(train.info())

# Check for missing values
print(train.isnull().sum())

# Check for duplicate rows
duplicates = train[train.duplicated()]
print(f"Number of duplicate rows: {len(duplicates)}")

# #################################################
# Converting Selected Variables to Object Type
# #################################################

# Convert specified columns to object (string) type
cols_to_convert = [
    "admission_type",
    "discharge_disposition",
    "admission_source",
    "readmitted",
    "patientID",
]
train[cols_to_convert] = train[cols_to_convert].astype(str)

print(train.info())

# #################################################
# Selecting Numeric Data and Computing Summary Statistics
# #################################################

# Select numeric data
train_numeric = train.select_dtypes(include=[np.number])


# Function to compute summary statistics
def numeric_summary(df):
    summary = df.describe().T
    summary["missing"] = df.isnull().sum()
    summary["missing_pct"] = (summary["missing"] / len(df)) * 100
    summary["unique"] = df.nunique()
    summary["unique_pct"] = (summary["unique"] / len(df)) * 100
    return summary


numeric_summary_df = numeric_summary(train_numeric)
print(numeric_summary_df)

# #################################################
# Imputing Missing Values using KNN Imputation
# #################################################

imputer = KNNImputer(n_neighbors=5)
train_numeric_imputed = pd.DataFrame(
    imputer.fit_transform(train_numeric), columns=train_numeric.columns
)

# Recompute summary statistics after imputation
numeric_summary_df = numeric_summary(train_numeric_imputed)
print(numeric_summary_df)

# #################################################
# Plotting Histograms of Numeric Variables
# #################################################


def plot_histogram(data, column_name):
    plt.figure(figsize=(8, 6))
    sns.histplot(data[column_name], kde=False, bins=30, color="blue")
    plt.title(f"Histogram of {column_name}")
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.show()


for col in train_numeric_imputed.columns:
    plot_histogram(train_numeric_imputed, col)

# #################################################
# Handling Skewness Using Box-Cox Transformation
# #################################################


# Compute skewness
def compute_skewness(df):
    return df.apply(lambda x: x.skew()).sort_values()


skewness_values = compute_skewness(train_numeric_imputed)
print("Skewness of variables:")
print(skewness_values)

# Apply Box-Cox transformation (requires positive data, so shift by 1 if necessary)
power_transformer = PowerTransformer(method="yeo-johnson", standardize=False)
transformed_data = power_transformer.fit_transform(train_numeric_imputed + 1)
train_numeric_transformed = pd.DataFrame(
    transformed_data, columns=train_numeric_imputed.columns
)

# Recompute skewness after transformation
transformed_skewness = compute_skewness(train_numeric_transformed)
print("Skewness after transformation:")
print(transformed_skewness)

# #################################################
# Updating the Original Dataset
# #################################################

# Replace numeric columns with transformed data
train.update(train_numeric_transformed)
print(train.head())


# #################################################
# FACTORS
# #################################################


# Function to compute mode for Factor data
def get_modes(series, type_=1):
    counts = series.value_counts()
    if type_ == 1:
        return counts.idxmax()  # Most common
    elif type_ == 2:
        if len(counts) < 2:
            return np.nan  # No 2nd mode if only one unique value
        return counts.index[1]  # 2nd most common
    elif type_ == -1:
        return counts.idxmin()  # Least common
    else:
        raise ValueError("Invalid type selected")


# Function to compute the count of mode for Factor data
def get_modes_count(series, type_=1):
    counts = series.value_counts()
    if type_ == 1:
        return counts.max()  # Most common frequency
    elif type_ == 2:
        if len(counts) < 2:
            return np.nan  # No 2nd mode if only one unique value
        return counts.iloc[1]  # 2nd most common frequency
    elif type_ == -1:
        return counts.min()  # Least common frequency
    else:
        raise ValueError("Invalid type selected")


# #################################################
# Selecting Factor Data, Computing Summary Statistics
# #################################################

# Extract Factor Variables
train_factor = train.select_dtypes(include=["object"]).copy()


# Compute Summary for Factor Data
def factor_summary(df):
    summary = []
    for col in df.columns:
        summary.append(
            {
                "variable": col,
                "n": len(df[col]),
                "unique": df[col].nunique(),
                "missing": df[col].isnull().sum(),
                "most_common": get_modes(df[col], type_=1),
                "most_common_count": get_modes_count(df[col], type_=1),
                "2nd_most_common": get_modes(df[col], type_=2),
                "2nd_most_common_count": get_modes_count(df[col], type_=2),
                "least_common": get_modes(df[col], type_=-1),
                "least_common_count": get_modes_count(df[col], type_=-1),
            }
        )
    return pd.DataFrame(summary)


factor_summary_df = factor_summary(train_factor)
print(factor_summary_df)

train_factor.describe()
train_factor.isna().sum()

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
print(factor_summary_df_imputed)

# #################################################
# Feature Selection of Factor Variables
# #################################################

# Select Specific Factor Features
train_factor_selected = train_factor_imputed[
    ["medical_specialty", "diagnosis", "readmitted"]
]
print(train_factor_selected.info())

# #################################################
# Collapsing Factor Levels
# #################################################


def collapse_levels(series, top_n):
    top_categories = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top_categories), other="Other")


train_factor_collapsed = train_factor_selected.copy()
train_factor_collapsed["medical_specialty"] = collapse_levels(
    train_factor_collapsed["medical_specialty"], top_n=19
)
train_factor_collapsed["diagnosis"] = collapse_levels(
    train_factor_collapsed["diagnosis"], top_n=19
)

print(train_factor_collapsed.head())

# #################################################
# Plotting Proportions by `readmitted`
# #################################################


def plot_proportion_by_factor(data, column_name, target_name):
    prop_df = (
        data.groupby([column_name, target_name])
        .size()
        .unstack(fill_value=0)
        .apply(lambda x: x / x.sum(), axis=1)
    )
    prop_df.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="viridis")
    plt.title(f"Proportion of {target_name} by {column_name}")
    plt.xlabel(column_name)
    plt.ylabel("Proportion")
    plt.legend(title=target_name)
    plt.tight_layout()
    plt.show()


for col in ["medical_specialty", "diagnosis"]:
    plot_proportion_by_factor(train_factor_collapsed, col, "readmitted")

# #################################################
# Merging Numeric and Factor Data
# #################################################

final_data = pd.concat([train_numeric, train_factor_collapsed], axis=1)
final_data.drop(columns=["patientID"], inplace=True)

# Checking the Final Data
print(final_data.info())
print(final_data.isnull().sum())

Final_Data = final_data.copy()

# #################################################
# Modelling
# #################################################


import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.utils import parallel_backend
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LassoCV
from sklearn import metrics

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
