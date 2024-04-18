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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from ydata_profiling import ProfileReport
from defining_functions import (eda, 
                                check_outliers,
                                build_multiple_classifiers,
                                save_model_from_cross_validation)
import joblib

# Get Dataset
dataset = pd.read_csv("Customer-Churn-Records.csv")

# Dropping Irrelevant Columns 
dataset = dataset.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)

# Exploratory Data Analysis
initial_eda = eda(dataset, graphs = False)

# Some potential things to consider
"""
1) What is the impact of our analysis from a specific geographical location?
i.e Does having an imbalanced geographical class impact our analysis
2) Can we create new features that capture the relationship between customers 
and churn. 
"""

# Data Cleaning and Transformation
# Categorical to Numerical - ["Geography", "Gender", "Card Type"]
dataset = pd.get_dummies(dataset, 
                         columns = ["Geography", "Card Type"], 
                         drop_first = True,
                         dtype = np.int64)

gender = {"Female": 1, "Male": -1}
dataset = dataset.replace(gender)

 

# Select dependent and independent variables
X = dataset.drop("Exited", axis = 1)
y = dataset.Exited

# Split dataset into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 0)

# Scaling the dataset
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Building - Linear Regression, Logistic Regression, Gaussian Naive Bayes, KNN, SVM
classifiers = [LogisticRegression(random_state = 0),
               GaussianNB(),
               KNeighborsClassifier(),
               RandomForestClassifier(random_state = 0),
               DecisionTreeClassifier(random_state = 0),
               SVC(kernel = 'rbf'),
               XGBClassifier(random_state = 0),
               LGBMClassifier()]

algorithm_metrics, cross_validation_metrics, model_info = build_multiple_classifiers(classifiers, 
                                                                                      X_train, 
                                                                                      y_train, 
                                                                                      X_test,
                                                                                      y_test)
       
average_fit_time = cross_validation_metrics["Fit time"].mean()     
average_score_time = cross_validation_metrics["Score time"].mean()  
              
joblib.dump(model_info, "model_info")



from sklearn.inspection import permutation_importance

# Assuming you already have trained your model 'model' and have validation data 'X_val', 'y_val'
# Compute feature importances
result = permutation_importance(model_info['LogisticRegression']['Model'], X_test, y_test, n_repeats=10, random_state=42)

# Get feature importances
importances = result.importances_mean

# Get feature names
feature_names = X.columns

# Print feature importances
features_imp = {}
for i, importance in enumerate(importances):
    features_imp[f"{feature_names[i]}"]: f"{importance}"
