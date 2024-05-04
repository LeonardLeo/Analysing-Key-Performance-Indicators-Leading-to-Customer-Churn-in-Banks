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
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, 
                              VotingClassifier, 
                              StackingClassifier, 
                              BaggingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif, chi2, SelectFpr
from ydata_profiling import ProfileReport
from imblearn.over_sampling import SMOTE, ADASYN
from defining_functions import (eda,
                                check_outliers,
                                build_classifier_model,
                                build_multiple_classifiers,
                                save_model_from_cross_validation,
                                create_group_clustering,
                                agg_two_groups,
                                data_preprocessing_pipeline,
                                classifier)
from category_encoders import TargetEncoder
from scipy import stats
import joblib
import shap

# Get Dataset
data = pd.read_csv("Customer-Churn-Records.csv")
dataset = pd.read_csv("Customer-Churn-Records.csv")


# Exploratory Data Analysis
initial_eda = eda(dataset,
                  graphs=False,
                  hist_figsize=(30, 20))


# Dataset with zero account balance
dataset_zero_balance = dataset[dataset["Balance"] == 0].drop("Balance", axis = 1)
eda_zero_balance = eda(dataset_zero_balance,
                       graphs=False,
                       hue = "Exited",
                       hist_figsize=(30, 20))
# Dataset with non-zero account balance
dataset_nonzero_balance = dataset[dataset["Balance"] != 0]
eda_nonzero_balance = eda(dataset_nonzero_balance,
                          graphs=False,
                          hue = "Exited",
                          hist_figsize=(30, 20))

# Preprocessing data
clean_nonzero_data = data_preprocessing_pipeline(dataset_nonzero_balance, 
                                                 drop_columns = ["RowNumber", 
                                                                 "CustomerId", 
                                                                 "Surname"])
clean_zero_data = data_preprocessing_pipeline(dataset_zero_balance, 
                                              drop_columns = ["RowNumber", 
                                                              "CustomerId", 
                                                              "Surname"])
clean_dataset = data_preprocessing_pipeline(dataset, 
                                            drop_columns = ["RowNumber", 
                                                            "CustomerId", 
                                                            "Surname"])



# # Check for outliers
# dataset = check_outliers(dataset,
#                          univariate_method_for_columns=None,
#                          multivariate_metric="mahalanobis",
#                          remove_multivariate_outliers=True,
#                          show_distance = False)

# # Creating new features
# # ---> New Features From the Distribution of the Data
# dataset["Age_Group"], bins1 = pd.qcut(dataset["Age"], 10, retbins = True, precision = 0, labels = False)
# dataset["Groups_Point_Earned"], bins2 = pd.qcut(dataset["Point Earned"], 10, retbins = True, precision = 0, labels = False)
# dataset["Groups_CreditScore"], bins3 = pd.qcut(dataset["CreditScore"], 10, retbins = True, precision = 0, labels = False)
# dataset["Groups_Balance"], bins4 = pd.cut(dataset["Balance"], 10, retbins = True, precision = 0, labels = False)
# dataset["Groups_EstimatedSalary"], bins5 = pd.qcut(dataset["EstimatedSalary"], 20, retbins = True, precision = 0, labels = False)

# # ---> New Features From How They are Clustering
# dataset["Age_Gender"] = create_group_clustering(dataset, columns = ["Age", "Gender"], n_clusters = 4, linkage = "ward")
# dataset["German_ActiveMember"] = create_group_clustering(dataset, columns = ["Geography_Germany", "IsActiveMember"], n_clusters = 4, linkage = "ward")

# # ---> Using Aggregations
# get_average = agg_two_groups(dataset, columns = ["Age_Group", "EstimatedSalary"], groupby = "Age_Group", agg_fun = {"EstimatedSalary": "mean"})
# dataset["AverageSalary_AgeGroup"] = dataset["Age_Group"].replace(get_average)


# Select dependent and independent variables
X = clean_dataset.drop("Exited", axis=1)
y = clean_dataset.Exited

# Feature selection
# ---> Statistical Approach
selector = SelectFpr(score_func = chi2)
new_X = pd.DataFrame(selector.fit_transform(X, y), 
                      columns=selector.get_feature_names_out())
# ---> Statistical score for features with associated p-value
feature_info = pd.DataFrame({"Features": selector.feature_names_in_,
                              "Scores": selector.scores_,
                              "P-Value": selector.pvalues_})


# Split dataset into training and test data
X_train, X_test, y_train, y_test = train_test_split(new_X, y,
                                                    test_size=0.2,
                                                    random_state=0)

# Resampling minority class
# ---> Resampling the training data seperately
resampler_train = ADASYN(random_state = 0)
X_resampled, y_resampled = resampler_train.fit_resample(X_train, y_train)
# X_train, y_train = resampler_train.fit_resample(X_train, y_train)

# # ---> Resampling the test data seperately
# resampler_test = ADASYN(random_state = 0)
# X_test, y_test = resampler_test.fit_resample(X_test, y_test)

# Final EDA - Training data 
eda_X_train = eda(X_resampled, graphs = False)
# eda_X_train = eda(X_train, graphs = False)

# Scaling the dataset
scaler = MinMaxScaler()
X_resampled = scaler.fit_transform(X_resampled)
# X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Model Building - Linear Regression, Logistic Regression, Gaussian Naive Bayes, KNN, SVM
classifier1 = LogisticRegression(random_state=0)
classifier2 = GaussianNB()
classifier3 = KNeighborsClassifier(n_neighbors=5)
classifier4 = RandomForestClassifier(
    random_state=0, max_depth=5, criterion="log_loss")
classifier5 = DecisionTreeClassifier(
    random_state=0, criterion="log_loss", min_samples_leaf=0.01, max_depth=5)
classifier6 = SVC(kernel='rbf', gamma="auto")
classifier7 = XGBClassifier(random_state=0)
classifier8 = LGBMClassifier()
classifier9 = VotingClassifier(estimators=[("RFC1", classifier4),
                                            ("LGBM", classifier8),
                                            ("RFC2", RandomForestClassifier(random_state=0, max_depth=5, criterion="log_loss", n_estimators=500))],
                                voting="hard")
# classifier10 = VotingClassifier(estimators=[("RFC1", classifier4),
#                                             ("LGBM", classifier8),
#                                             ("RFC2", RandomForestClassifier(random_state=0, max_depth=5, criterion="log_loss", n_estimators=500))],
#                                 voting="soft")
classifier11 = StackingClassifier(estimators=[("RFC1", classifier4),
                                              ("LGBM", classifier8),
                                              ("RFC2", RandomForestClassifier(random_state=0, max_depth=5, criterion="log_loss", n_estimators=500))],
                                  final_estimator=classifier7)
classifier12 = BaggingClassifier(estimator = classifier9)


# # Compiling classifiers
# classifiers = [
#                 classifier1,
#                 classifier2,
#                 classifier3,
#                 classifier4,
#                 classifier5,
#                 # classifier6,
#                 classifier7,
#                 classifier8,
#                 # classifier9,
#                 # # classifier10,
#                 # classifier11,
#                 # classifier12
#                 ]


# # algorithm_metrics, cross_validation_metrics, model_info = build_multiple_classifiers(classifiers,
# #                                                                                       X_train,
# #                                                                                       y_train,
# #                                                                                       X_test,
# #                                                                                       y_test,
# #                                                                                       pos_label = 1)

# # average_fit_time = cross_validation_metrics["Fit time"].mean()
# # average_score_time = cross_validation_metrics["Score time"].mean()

# # joblib.dump(model_info, "model_info")



c = classifier(train_classifier = classifier8,
               retrain_classifier = classifier1,
                X_train = X_resampled,
                y_train = y_resampled,
                X_test = X_test,
                y_test = y_test,
                fp_and_fn = True, 
                probabilities = True,
                probability_threshold = 0.3)



# # from sklearn.model_selection import cross_val_score, cross_validate
# # from sklearn.metrics import (confusion_matrix,
# #                              classification_report,
# #                              accuracy_score,
# #                              precision_score,
# #                              recall_score,
# #                              f1_score)
# # model = model_info["XGBClassifier"]["Cross Validation"]["Validation Models"]["estimator"][3]
# # # Model Prediction
# # y_pred = model.predict(X_train) # Training Predictions: Check OverFitting
# # y_pred1 = model.predict(X_test) # Test Predictions: Check Model Predictive Capacity

# # # Model Evalustion and Validation
# # # Training Evaluation: Check OverFitting
# # training_analysis = confusion_matrix(y_train, y_pred)
# # training_class_report = classification_report(y_train, y_pred)
# # training_accuracy = accuracy_score(y_train, y_pred)
# # training_precision = precision_score(y_train, y_pred, average='weighted', pos_label = 1)
# # training_recall = recall_score(y_train, y_pred, average='weighted', pos_label = 1)
# # training_f1_score = f1_score(y_train, y_pred, average='weighted', pos_label = 1)

# # # Test Evaluations: Check Model Predictive Capacity
# # test_analysis = confusion_matrix(y_test, y_pred1)
# # test_class_report = classification_report(y_test, y_pred1)
# # test_accuracy = accuracy_score(y_test, y_pred1)
# # test_precision = precision_score(y_test, y_pred1, average='weighted', pos_label = 1)
# # test_recall = recall_score(y_test, y_pred1, average='weighted', pos_label = 1)
# # test_f1_score = f1_score(y_test, y_pred1, average='weighted', pos_label = 1)

# # # Validation of Predictions
# # cross_val = cross_val_score(model, X_train, y_train, cv = 10)
# # cross_validation = cross_validate(model,
# #                                   X_train,
# #                                   y_train,
# #                                   cv = 10,
# #                                   return_estimator = True,
# #                                   return_train_score = True)
# # score_mean = round((cross_val.mean() * 100), 2)
# # score_std_dev = round((cross_val.std() * 100), 2)


# # from sklearn.inspection import permutation_importance

# # # Assuming you already have trained your model 'model' and have validation data 'X_val', 'y_val'
# # # Compute feature importances
# # result = permutation_importance(model_info['LGBMClassifier']['Model'], X_test, y_test, n_repeats=10, random_state=42)

# # # Get feature importances
# # importances = result.importances_mean

# # # Get feature names
# # feature_names = X.columns

# # # Print feature importances
# # features_imp = {}
# # for i, importance in enumerate(importances):
# #     features_imp[f"{feature_names[i]}"]: f"{importance}"



# 1) What location does the customer come from
