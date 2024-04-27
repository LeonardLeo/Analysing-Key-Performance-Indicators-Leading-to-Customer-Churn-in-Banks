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
from imblearn.over_sampling import RandomOverSampler, SMOTE
from defining_functions import (eda,
                                check_outliers,
                                build_multiple_classifiers,
                                save_model_from_cross_validation,
                                create_group_clustering,
                                agg_two_groups)
from category_encoders import TargetEncoder
from scipy import stats
import joblib
import shap

# Get Dataset
data = pd.read_csv("Customer-Churn-Records.csv")
dataset = pd.read_csv("Customer-Churn-Records.csv")

# Dropping Irrelevant Columns
dataset = dataset.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# Exploratory Data Analysis
initial_eda = eda(dataset,
                  graphs=False,
                  hist_figsize=(30, 20))

# Some potential things to consider
"""
1) What is the impact of our analysis from a specific geographical location?
i.e Does having an imbalanced geographical class impact our analysis
2) Can we create new features that capture the relationship between customers 
and churn. 
3) Create data binning to help with skewed distributions.
4) When binning the Card Type, it's worth considering if the column is truly Nominal
or should be represented as Ordinal data type. This should be considered, as within 
the bank, there is some level of inherent hierarchy among the card type a user holds
and as such, our transformation of the card type column needs to reflect this.
"""

# EDA Findings
"""
After exploratory data analysis was conduted, we make the following findings:
    - Without needing to preprocess the data, we highlight 3 features which won't be relevant
      for our analysis and drop them. These features include - "RowNumber", "CustomerId", "Surname".
      * Row number here is just an identifier and will not play any role in contributing to the 
      understanding of why bank customers leave the bank.
      * CustomerID similar to the row number is an identifier and will not contribute to our models
      predictive capacity.
      * Surname refers to the lastname of our customers. This doesn't add value to our prediction
      as we cannot say in any literature that given a person's surname, we can tell if he or she
      will leave the bank. The idea of churn has nothing to do with the customers surname. Hence,
      we drop this column.
    - For our analysis on bank customer churn, our dataset makes use of 5457 males and 4543 females.
    - Our data has no duplicate values across each row
    - No missing values were found
    - The bank customers ages range from 18 to 92
    - The data collected spans 3 geographical locations - France, Germany, and Spain.
      With the majority of the customers in our analysis coming from France. France has
      5014 customers analysed, Germany has 2509 customers,while Spain has 2477 customers.
    - Customers within the bank have 4 different possible card type they can hold. These 
      include - Diamond cards, Gold cards, Silver cards, and Platinum cards.
    - When looking at the value counts for our customers age, it is seen that the larger
      majority of customers in our dataset are in their 30's.
    - Intrestingly, we see a 0.996 strong positive correlation between the complain column 
      and the exited column. This strong linear relationship between the complain feature and
      our target exited, suggests that we can use this singular complain column to predict
      customer churn in banks. The impact of this feature on our analysis needs to be explored
      further.
    - No strong linear correlation among other variables outside of the exited and complain 
      features.
    - The Age feature has a left-skewed distribution as seen in our histogram. While 
      creditscore is sligthly skewed to the right. We observe a uniform distribution among the
      following features - EstimatedSalary, IsActiveMember, and SatisfactionScore. All other 
      features have an unbalanced class distribution.
"""

# Data Quality Checks
"""
For our data quality checks, we consider the following checks:
1) Accuracy and Precision Check: 
	 e.g. currency exchange between £ and ¥: £1  ¥9.10 or £1  ¥9.056187
2) Correctness by Entry Check:
	 e.g. entered 9.056287 instead of 9.056187
3) Completeness Check:
	 e.g. no date of birth is given
4) Consistency (validity and integrity) Check:
	 e.g. book borrowing date: 02/02/2019, date of return: 03/01/2019
5) Redundancy (unnecessary redundancy) Check:
	 e.g. simply collecting together data on replication servers
6) Data Source Reliability Checks
"""


# Data Cleaning and Transformation
# Categorical to Numerical - ["Geography", "Gender", "Card Type"]
dataset = pd.get_dummies(dataset,
                         columns=["Geography"],
                         drop_first=True,
                         dtype=np.int64)

# Converting the gender column to numeric
gender = {"Female": 1, "Male": 0}
dataset = dataset.replace(gender)

# Converting the card type to numeric
card_hierarchy = {"SILVER": 0, "GOLD": 1, "PLATINUM": 2, "DIAMOND": 3}
dataset = dataset.replace(card_hierarchy)




"""
REMEMBEER YOU DID NORMAL CORRELATION BETWEEN VARIABLES. FOR BINARY TARGET, YOU NEED TO USE
POINT-BISERIAL CORRELATION
"""






# Drop the COMPLAIN COLUMN
"""
During our first iteration in our model building phase, the built model is able to predict accuratly
100% for test data across 5 different algorithms. While at first glance, it may seem
like the best way to go, however, when things are looking too good to be true, it is probably
too good to be true. This means when we have successfully created 
a model that predicts the likelihood a bank customer with am accuracy of 100%, the results should be
considered too good to be true and due to chance as predicting accurately a 100% accuracy means 
this model never fails. 

The possible reason for this behaviour in our model could be the relationship between the complain featrure
and the target exited. After removing outliers using the mahalanobis multivariate outlier technique, which 
was able to capture two groups of people between the complain feature and exited. These groups are:
    - Those who made complaints but didn't churn
    - Those who churned but didn't complain
Upon removing these rows highlighted by our mahalanobis algorithm, correlation between the complain
feature and exited is 1 indicating a perfect positive correlation between the variables. This relationship
therefore crates the illusion that complain and exited are the same. Theoritically, this is misleading while
trying to model the real-world as there exists no perfect correlation between these two variables. This proxy
variable complain is the feature our algorithm captures and uses as the most important feature to learn the
realationship between bank customers and why they churn.

The approach taken to mitigate this was to drop the complain column our dataset and build our predictive model 
around the other features.
"""
dataset = dataset.drop("Complain", axis=1)

# Check for outliers
dataset = check_outliers(dataset,
                         univariate_method_for_columns=None,
                         multivariate_metric="mahalanobis",
                         remove_multivariate_outliers=True,
                         show_distance = False)

# Creating new features
# ---> New Features From the Distribution of the Data
dataset["Age_Group"], bins1 = pd.qcut(dataset["Age"], 10, retbins = True, precision = 0, labels = False)
dataset["Groups_Point_Earned"], bins2 = pd.qcut(dataset["Point Earned"], 10, retbins = True, precision = 0, labels = False)
dataset["Groups_CreditScore"], bins3 = pd.qcut(dataset["CreditScore"], 10, retbins = True, precision = 0, labels = False)
dataset["Groups_Balance"], bins4 = pd.cut(dataset["Balance"], 10, retbins = True, precision = 0, labels = False)
dataset["Groups_EstimatedSalary"], bins5 = pd.qcut(dataset["EstimatedSalary"], 20, retbins = True, precision = 0, labels = False)

# ---> New Features From How They are Clustering
dataset["Age_Gender"] = create_group_clustering(dataset, columns = ["Age", "Gender"], n_clusters = 4, linkage = "ward")
dataset["German_ActiveMember"] = create_group_clustering(dataset, columns = ["Geography_Germany", "IsActiveMember"], n_clusters = 4, linkage = "ward")

# ---> Using Aggregations
get_average = agg_two_groups(dataset, columns = ["Age_Group", "EstimatedSalary"], groupby = "Age_Group", agg_fun = {"EstimatedSalary": "mean"})
dataset["AverageSalary_AgeGroup"] = dataset["Age_Group"].replace(get_average)


# Select dependent and independent variables
X = dataset.drop("Exited", axis=1)
y = dataset.Exited

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
resampler_train = SMOTE()
X_train, y_train = resampler_train.fit_resample(X_train, y_train)

# ---> Resampling the test data seperately
resampler_test = SMOTE()
X_test, y_test = resampler_test.fit_resample(X_test, y_test)

# Final EDA - Training data 
eda_X_train = eda(X_train, graphs = False)

# Scaling the dataset
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
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


# Compiling classifiers
classifiers = [
                classifier1,
                classifier2,
                classifier3,
                classifier4,
                classifier5,
                classifier6,
                classifier7,
                classifier8,
                classifier9,
                # classifier10,
                classifier11,
                classifier12
                ]


algorithm_metrics, cross_validation_metrics, model_info = build_multiple_classifiers(classifiers,
                                                                                      X_train,
                                                                                      y_train,
                                                                                      X_test,
                                                                                      y_test,
                                                                                      pos_label = 1)

average_fit_time = cross_validation_metrics["Fit time"].mean()
average_score_time = cross_validation_metrics["Score time"].mean()

joblib.dump(model_info, "model_info")




# from sklearn.model_selection import cross_val_score, cross_validate
# from sklearn.metrics import (confusion_matrix,
#                              classification_report,
#                              accuracy_score,
#                              precision_score,
#                              recall_score,
#                              f1_score)
# model = model_info["XGBClassifier"]["Cross Validation"]["Validation Models"]["estimator"][3]
# # Model Prediction
# y_pred = model.predict(X_train) # Training Predictions: Check OverFitting
# y_pred1 = model.predict(X_test) # Test Predictions: Check Model Predictive Capacity

# # Model Evalustion and Validation
# # Training Evaluation: Check OverFitting
# training_analysis = confusion_matrix(y_train, y_pred)
# training_class_report = classification_report(y_train, y_pred)
# training_accuracy = accuracy_score(y_train, y_pred)
# training_precision = precision_score(y_train, y_pred, average='weighted', pos_label = 1)
# training_recall = recall_score(y_train, y_pred, average='weighted', pos_label = 1)
# training_f1_score = f1_score(y_train, y_pred, average='weighted', pos_label = 1)

# # Test Evaluations: Check Model Predictive Capacity
# test_analysis = confusion_matrix(y_test, y_pred1)
# test_class_report = classification_report(y_test, y_pred1)
# test_accuracy = accuracy_score(y_test, y_pred1)
# test_precision = precision_score(y_test, y_pred1, average='weighted', pos_label = 1)
# test_recall = recall_score(y_test, y_pred1, average='weighted', pos_label = 1)
# test_f1_score = f1_score(y_test, y_pred1, average='weighted', pos_label = 1)

# # Validation of Predictions
# cross_val = cross_val_score(model, X_train, y_train, cv = 10)
# cross_validation = cross_validate(model,
#                                   X_train,
#                                   y_train,
#                                   cv = 10,
#                                   return_estimator = True,
#                                   return_train_score = True)
# score_mean = round((cross_val.mean() * 100), 2)
# score_std_dev = round((cross_val.std() * 100), 2)


# from sklearn.inspection import permutation_importance

# # Assuming you already have trained your model 'model' and have validation data 'X_val', 'y_val'
# # Compute feature importances
# result = permutation_importance(model_info['LGBMClassifier']['Model'], X_test, y_test, n_repeats=10, random_state=42)

# # Get feature importances
# importances = result.importances_mean

# # Get feature names
# feature_names = X.columns

# # Print feature importances
# features_imp = {}
# for i, importance in enumerate(importances):
#     features_imp[f"{feature_names[i]}"]: f"{importance}"
