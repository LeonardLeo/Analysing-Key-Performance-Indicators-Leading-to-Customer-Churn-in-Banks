# -*- coding: utf-8 -*-
"""
HyperParameter Tuning
"""

# Import Libraries
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Get Datasets
X_resampled = joblib.load("python_objects/X_resampled")
y_resampled = joblib.load("python_objects/y_resampled")
X_train = joblib.load("python_objects/X_train")
y_train = joblib.load("python_objects/y_train")
X_test = joblib.load("python_objects/X_test")
y_test = joblib.load("python_objects/y_test")
X_val = joblib.load("python_objects/X_val")
y_val = joblib.load("python_objects/y_val")



# Hyperparameter Tuning
# ---> LOGISTIC REGRESSION
# Define the parameter grid
param_grid_1 = {
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga', 'lbfgs', 'newton-cg'],
    'l1_ratio': [0.1, 0.5],
}

# Initialize StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)

# Initialize the classifier
classifier_1 = LogisticRegression()

# Initialize GridSearchCV
grid_search_1 = GridSearchCV(estimator = classifier_1,
                           param_grid = param_grid_1,
                           cv = stratified_kfold,
                           scoring = "recall",
                           n_jobs = -1,
                           verbose = 2)

# Fit the model
grid_search_1.fit(X_train, y_train)

# Best parameters and best score
best_params_1 = grid_search_1.best_params_
best_score_1 = grid_search_1.best_score_

# Evaluate on the test set
best_model_1 = grid_search_1.best_estimator_
test_score_1 = best_model_1.score(X_test, y_test)





# ---> SUPPORT VECTOR MACHINE CLASSIFIER
# Define the parameter grid
param_grid_2 = {
    'C': [0.01, 0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': [0.001, 0.01, 0.1, 1, 10],  # Only relevant for 'rbf', 'poly', 'sigmoid'
    'degree': [2, 3, 4],  # Only relevant for 'poly'
}

# Initialize StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)

# Initialize the classifier
classifier_2 = SVC()

# Initialize GridSearchCV
grid_search_2 = GridSearchCV(estimator = classifier_2,
                           param_grid = param_grid_2,
                           cv = stratified_kfold,
                           scoring = "recall",
                           n_jobs = -1,
                           verbose = 2)

# Fit the model
grid_search_2.fit(X_train, y_train)

# Best parameters and best score
best_params_2 = grid_search_2.best_params_
best_score_2 = grid_search_2.best_score_

# Evaluate on the test set
best_model_2 = grid_search_2.best_estimator_
test_score_2 = best_model_2.score(X_test, y_test)





# ---> DECISION TREE CLASSIFIER
# Define the parameter grid
param_grid_3 = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'max_features': [None, 'auto', 'sqrt', 'log2'],
    'class_weight': [None, 'balanced']
}

# Initialize StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)

# Initialize the classifier
classifier_3 = DecisionTreeClassifier()

# Initialize GridSearchCV
grid_search_3 = GridSearchCV(estimator = classifier_3,
                           param_grid = param_grid_3,
                           cv = stratified_kfold,
                           scoring = "recall",
                           n_jobs = -1,
                           verbose = 2)

# Fit the model
grid_search_3.fit(X_train, y_train)

# Best parameters and best score
best_params_3 = grid_search_3.best_params_
best_score_3 = grid_search_3.best_score_

# Evaluate on the test set
best_model_3 = grid_search_3.best_estimator_
test_score_3 = best_model_3.score(X_test, y_test)




# ---> EXTREME GRADIENT BOOSTING CLASSIFIER (XGBoost)
# Define the parameter grid
param_grid_4 = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 5, 10],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
}

# Initialize StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)

# Initialize the classifier
classifier_4 = XGBClassifier()

# Initialize GridSearchCV
grid_search_4 = GridSearchCV(estimator = classifier_4,
                           param_grid = param_grid_4,
                           cv = stratified_kfold,
                           scoring = "recall",
                           n_jobs = -1,
                           verbose = 2)

# Fit the model
grid_search_4.fit(X_train, y_train)

# Best parameters and best score
best_params_4 = grid_search_4.best_params_
best_score_4 = grid_search_4.best_score_

# Evaluate on the test set
best_model_4 = grid_search_4.best_estimator_
test_score_4 = best_model_4.score(X_test, y_test)




# ---> LIGHT GRADIENT BOOSTING MACHINE CLASSIFIER (LGBM)
# Define the parameter grid
param_grid_5 = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5, 6],
    'min_child_samples': [10, 20, 30, 40, 50],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
}

# Initialize StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)

# Initialize the classifier
classifier_5 = LGBMClassifier()

# Initialize GridSearchCV
grid_search_5 = GridSearchCV(estimator = classifier_5,
                           param_grid = param_grid_5,
                           cv = stratified_kfold,
                           scoring = "recall",
                           n_jobs = -1,
                           verbose = 2)

# Fit the model
grid_search_5.fit(X_train, y_train)

# Best parameters and best score
best_params_5 = grid_search_5.best_params_
best_score_5 = grid_search_5.best_score_

# Evaluate on the test set
best_model_5 = grid_search_5.best_estimator_
test_score_5 = best_model_5.score(X_test, y_test)














# RESULTS
# LOGISTICS REGRESSION
{'C': 0.01, 'l1_ratio': 0.1, 'penalty': None, 'solver': 'newton-cg'}
0.8168697836480027
LogisticRegression(C=0.01, l1_ratio=0.1, penalty=None, solver='newton-cg')
0.8361231417534823


# DECISION TREE CLASSIFIER
{'class_weight': None, 'criterion': 'entropy', 'max_depth': 10, 'max_features': None, 'min_samples_leaf': 10, 'min_samples_split': 10}
0.841036382789077
DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_leaf=10,
                       min_samples_split=10)
0.8687814585040384


# EXTREME GRADIENT BOOSTING CLASSIFIER
{'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 5, 'n_estimators': 300, 'subsample': 0.9}
0.8747381357534506
XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, device=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=0.1, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=5, max_leaves=None,
              min_child_weight=5, missing="nan", monotone_constraints=None,
              multi_strategy=None, n_estimators=300, n_jobs=None,
              num_parallel_tree=None, random_state=None)
0.953178040500995


# LIGHT GRADIENT BOOSTING CLASSIFIER
0.878920941036653
LGBMClassifier(max_depth=6, min_child_samples=30, n_estimators=200,
               subsample=0.6)
0.9594990050333606
