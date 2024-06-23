# -*- coding: utf-8 -*-
"""
Creating a predictive classification model that effectively distinguishes between
customers who are likely to leave the bank and customers who stay.

The goal is to find the key performance indicators that push people towards leaving
or staying with the bank.
"""

# Import Libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif, SelectFpr
from ydata_profiling import ProfileReport
from imblearn.over_sampling import ADASYN
from defining_functions import (eda,
                                check_outliers,
                                build_multiple_classifiers,
                                data_preprocessing_pipeline)
import joblib
import warnings

# Filter warnings
warnings.filterwarnings("ignore")

# Get Dataset
data = pd.read_csv("Customer-Churn-Records.csv")
dataset = pd.read_csv("Customer-Churn-Records.csv")

# # Exploratory Data Analysis
# initial_eda = eda(dataset)
# # ---> Pandas Profiling
# profile = ProfileReport(dataset,
#                         dark_mode = True,
#                         title = 'Bank Customers Dataset - EDA')
# profile.to_widgets()
# profile.to_file("pandas_profiling_report/initial_EDA.html")

# Dropping irrelevant columns for analysis
dataset = data_preprocessing_pipeline(dataset,
                                    drop_columns = ["RowNumber", "CustomerId", "Surname"])

# Check for outliers
dataset = check_outliers(dataset,
                          drop_column = ["Geography", "Gender", "Card Type"],
                          univariate_method_for_columns = None,
                          multivariate_metric = "mahalanobis",
                          remove_multivariate_outliers = True,
                          show_distance = False,
                          sig_level = 0.05)

# Preprocessing data
# ---> Fixing Zero-Inflated Distribution for Balance Variable
# Create a binary indicator for zero vs. non-zero balance
dataset['Balance_Zero'] = (dataset['Balance'] == 0).astype(int)

# Log-transform the non-zero balances to handle skewness
dataset['Balance_Log'] = np.log1p(dataset['Balance'])

# Filling log-transformed balance for zero values with a specific constant (e.g., -1)
dataset.loc[data['Balance'] == 0, 'Balance_Log'] = -1

# Drop original 'Balance' column if necessary
dataset = dataset.drop(columns=['Balance'])

# Dropping the complain column, Dealing with skewed distributions (Age and CreditScore), and Data Transformation
dataset = data_preprocessing_pipeline(dataset,
                                     log_col = ["Age", "CreditScore"],
                                     dummy_col = ["Geography", "Gender"],
                                     drop_columns = "Complain",
                                     replace_val = {"SILVER": 0, "GOLD": 1, "PLATINUM": 2, "DIAMOND": 3})

# Creating new features
# ---> New Features From the Distribution of the Data
dataset["Age_Group"], bins1 = pd.cut(dataset["Age"], 10, retbins = True, precision = 0, labels = False)
dataset["CreditScore_Group"], bins3 = pd.cut(dataset["CreditScore"], 10, retbins = True, precision = 0, labels = False)
dataset["PointEarned_Group"], bins2 = pd.qcut(dataset["Point Earned"], 5, retbins = True, precision = 0, labels = False)

# Exploratory Data Analysis
final_eda = eda(dataset)

# Select dependent and independent variables
X = dataset.drop("Exited", axis=1)
y = dataset.Exited

# Split dataset into training and test data
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 0,
                                                    stratify = y
                                                    )

# Validation data
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full,
                                                  test_size = 1/3,
                                                  random_state = 0,
                                                  stratify = y_train_full)

# Value counts for the target
count_target_category_train = y_train.value_counts()
count_target_category_test = y_test.value_counts()
count_target_category_val = y_val.value_counts()

# Resampling minority class
# ---> Resampling the training data seperately
resampler_train = ADASYN(random_state = 0)
X_resampled, y_resampled = resampler_train.fit_resample(X_train, y_train)

# Value counts for the target
count_target_resampled_train = y_resampled.value_counts()

# Scaling the dataset
# Scale specific features
scaler = MinMaxScaler()
X_resampled[["CreditScore", "Age", "Point Earned", "EstimatedSalary"]] = scaler.fit_transform(X_resampled[["CreditScore", "Age", "Point Earned", "EstimatedSalary"]])
X_train[["CreditScore", "Age", "Point Earned", "EstimatedSalary"]] = scaler.transform(X_train[["CreditScore", "Age", "Point Earned", "EstimatedSalary"]])
X_test[["CreditScore", "Age", "Point Earned", "EstimatedSalary"]] = scaler.transform(X_test[["CreditScore", "Age", "Point Earned", "EstimatedSalary"]])
X_val[["CreditScore", "Age", "Point Earned", "EstimatedSalary"]] = scaler.transform(X_val[["CreditScore", "Age", "Point Earned", "EstimatedSalary"]])

# Feature selection
# ---> Statistical Approach
selector = SelectFpr(score_func = f_classif)
X_resampled = pd.DataFrame(selector.fit_transform(X_resampled, y_resampled),
                            columns=selector.get_feature_names_out())
X_train = pd.DataFrame(selector.transform(X_train),
                        columns=selector.get_feature_names_out())
X_test = pd.DataFrame(selector.transform(X_test),
                      columns=selector.get_feature_names_out())
X_val = pd.DataFrame(selector.transform(X_val),
                      columns=selector.get_feature_names_out())
# ---> Statistical score for features with associated p-value
feature_info = pd.DataFrame({"Features": selector.feature_names_in_,
                              "Scores": np.around(selector.scores_, 2),
                              "P-Value": np.around(selector.pvalues_, 2)})


# Save python objects for training, testing, and validation
joblib.dump(X_resampled, "python_objects/X_resampled")
joblib.dump(y_resampled, "python_objects/y_resampled")
joblib.dump(X_resampled, "python_objects/X_train")
joblib.dump(y_resampled, "python_objects/y_train")
joblib.dump(X_resampled, "python_objects/X_test")
joblib.dump(y_resampled, "python_objects/y_test")
joblib.dump(X_resampled, "python_objects/X_val")
joblib.dump(y_resampled, "python_objects/y_val")



# Model Building - Logistic Regression, Gaussian Naive Bayes, KNN, SVM
classifier1 = LogisticRegression(C=0.01, l1_ratio=0.1, penalty=None, solver='newton-cg')
classifier2 = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_leaf=10, min_samples_split=10)
classifier3 = SVC(kernel = 'rbf', gamma = "auto", probability = True, random_state = 0, C = 1.0)
classifier4 = XGBClassifier(random_state = 0, learning_rate = 0.1, max_depth = 5, min_child_weight = 5, n_estimators = 300, subsample = 0.9)
classifier5 = LGBMClassifier(max_depth=6, min_child_samples=30, n_estimators=200, subsample=0.6)

# Compiling classifiers
classifiers = [
                classifier1,
                classifier2,
                classifier3,
                classifier4,
                classifier5,
                ]

algorithm_metrics, cross_validation_metrics, model_info = build_multiple_classifiers(classifiers,
                                                                                      X_resampled,
                                                                                      y_resampled,
                                                                                      X_test,
                                                                                      y_test,
                                                                                      X_val,
                                                                                      y_val,
                                                                                      cross_validate_Xtrain = X_train,
                                                                                      cross_validate_ytrain = y_train,
                                                                                      repeatedstratifiedkfold = 10,
                                                                                      repeatedstratifiedresampler = resampler_train,
                                                                                      n_repeats_stratified = 1
                                                                                      )

average_fit_time = cross_validation_metrics["Fit time"].mean()
average_score_time = cross_validation_metrics["Score time"].mean()
avg_score_fit = cross_validation_metrics.groupby("Algorithm").mean()
std_score_fit = np.around(cross_validation_metrics.groupby("Algorithm").std(), 3)

# Save models information
joblib.dump(model_info, "python_objects/model_info")

# Selecting the best model
model = model_info["LGBMClassifier"]["Model"]

# Selecting the best features
get_best_features = pd.DataFrame({"Features": X_resampled.columns,
                                 "Scores": model.feature_importances_,
                                 "Columns": model.feature_name_})
