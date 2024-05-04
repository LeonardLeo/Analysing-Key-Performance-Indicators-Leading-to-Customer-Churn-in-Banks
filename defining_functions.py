# -*- coding: utf-8 -*-
"""
Functions utilised towards creating our bank churn classifier and saves the best model.
"""

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from scipy.stats import chi2
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (confusion_matrix, 
                             classification_report,
                             accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score)
from typing import Union
import joblib
import time

# Defining Functions
def eda(dataset: pd.DataFrame, 
        bin_size: int or list = None, 
        graphs: bool = False,
        hue: str = None,
        markers: list = None,
        only_graphs: bool = False,
        hist_figsize: tuple = (15, 10),
        corr_heatmap_figsize: tuple = (15, 10),
        pairplot_figsize: tuple = (15, 10)) -> dict:
    
    if only_graphs != True:
        data_unique = {}
        data_category_count = {}
        data_numeric_count = {}
        dataset.info()
        data_head = dataset.head()
        data_tail = dataset.tail()
        data_mode = dataset.mode().iloc[0]
        data_descriptive_stats = dataset.describe()
        data_more_descriptive_stats = dataset.describe(include = "all", 
                                                       datetime_is_numeric=True)
        data_correlation_matrix = dataset.corr(numeric_only = True)
        data_distinct_count = dataset.nunique()
        data_count_duplicates = dataset.duplicated().sum()
        data_duplicates = dataset[dataset.duplicated()]
        data_count_null = dataset.isnull().sum()
        # data_null = dataset[any(dataset.isna())]
        data_total_null = dataset.isnull().sum().sum()
        for each_column in dataset.columns: # Loop through each column and get the unique values
            data_unique[each_column] = dataset[each_column].unique()
        for each_column in dataset.select_dtypes(object).columns: 
            # Loop through the categorical columns and count how many values are in each category
            data_category_count[each_column] = dataset[each_column].value_counts()
        for each_column in dataset.select_dtypes(exclude = object).columns: 
            # Loop through the numeric columns and count how many values are in each category
            data_numeric_count[each_column] = dataset[each_column].value_counts()
        
    if graphs == True:
        # Visualising Histograms
        dataset.hist(figsize = hist_figsize, bins = bin_size)
        plt.show()
        
        if only_graphs != False:
            # Creating a heatmap for the correlation matrix
            plt.figure(figsize = corr_heatmap_figsize)
            sns.heatmap(data_correlation_matrix, annot = True, cmap = 'coolwarm')
            plt.show()
        
        # Creating the pairplot for the dataset
        plt.figure(figsize = pairplot_figsize)
        sns.pairplot(dataset, hue = hue, markers = markers) # Graph of correlation across each numerical feature
        plt.show()
    
    if only_graphs != True:
        result = {"data_head": data_head,
                  "data_tail": data_tail,
                  "data_mode": data_mode,
                  "data_descriptive_stats": data_descriptive_stats,
                  "data_more_descriptive_stats": data_more_descriptive_stats,
                  "data_correlation_matrix": data_correlation_matrix,
                  "data_distinct_count": data_distinct_count,
                  "data_count_duplicates": data_count_duplicates,
                  "data_count_null": data_count_null,
                  "data_total_null": data_total_null,
                  "data_unique": data_unique,
                  "data_duplicates": data_duplicates,
                  # "data_null": data_null,
                  "data_category_count": data_category_count,
                  "data_numeric_count": data_numeric_count,
                  }
        return result


def z_score(row, mean, std):
    if row == np.nan:
        return np.nan
    return (row - mean)/std


def values_z_score(row, mean, std):
    if row == np.nan:
        return np.nan
    return (row * std) + mean


def emperical_mean(matrix_x: np.array) -> np.array:
    """
    Creating the emperical mean using the matrix of features X.
    
    n = number of rows in matrix of features
    Jn = vector of ones of length n e.g. if n = 3; Jn = [1, 1, 1]
    X = matrix of features
    
    Formula = 1/n * matrix_multiplication(Jn, X)

    Parameters
    ----------
    matrix_x : np.array
        Matrix of features x. Typically the dataset you wish to find the emperical mean.

    Returns
    -------
    np.array
        Array of numbers representing the means.

    """
    n_rows = len(matrix_x)
    if isinstance(matrix_x, pd.DataFrame):
        matrix_x = matrix_x.values
    Jn = np.array([1] * n_rows)
    return (1/n_rows) * (np.transpose(matrix_x) @ np.transpose(Jn))  


def emperical_covariance(matrix_x: np.array) -> np.array:
    """
    Creating the emperical covariance matrix using the matrix of features X.
    
    n = number of rows in matrix of features
    J = matrix of ones of size (n x n) e.g. if n = 3; J = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    I = Identity matrix of size (n x n) e.g. if n = 3; I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    X = matrix of features
    
    Formula = matrix_multiplication( (1/n * transpose(X)), matrix_multiplication((I - (1/n * J)), X) )

    Parameters
    ----------
    matrix_x : np.array
        Matrix of features x. Typically the dataset you wish to find the emperical covariance.

    Returns
    -------
    np.array
        Array of numbers representing the covariance matirx.

    """
    n_rows = len(matrix_x)
    if isinstance(matrix_x, pd.DataFrame):
        matrix_x = matrix_x.values
    identity_n = np.identity(n = n_rows)
    J = np.ones(shape = (n_rows, n_rows))
    return ((1/n_rows) * np.transpose(matrix_x)) @ ((identity_n - ((1/n_rows) * J)) @ matrix_x)


def median_absolute_deviation(data: pd.Series or pd.DataFrame, col_name: str = None):
    def mad(data: pd.Series):
        median = data.median()
        new_data = []
        
        # Calculating the Median Absolute Deviation
        for each_value in data.values:
            if not np.isnan(each_value):
                x_minus_median = np.abs(each_value - median)
                new_data.append(x_minus_median)
            else:
                new_data.append(np.nan)
                
        median_abs_dev = np.nanmedian(np.array(new_data))
        
        store = []
        # Scaling the data by dividing each value by the MAD
        for value in data.values:
            if not np.isnan(value):
                store.append(value / median_abs_dev)
            else:
                store.append(np.nan)
                
        return pd.Series(store, name=col_name)
    
    # Dealing with a pandas dataframe
    if isinstance(data, pd.DataFrame):
        df = {}
        for each_col in data.columns:
            df[each_col] = mad(data[each_col])
        return pd.DataFrame(df)
    
    # Dealing with a pandas series
    elif isinstance(data, pd.Series):
        return mad(data)
    
                
def distance_from_mean(dataset,
                       drop_column: list = None,
                       metric: str = "euclidean",
                       squared_distance: bool = False,
                       remove_outliers_multivariate: bool = False,
                       scale_data: str = None,
                       alpha: float = 0.05,
                       critical_value_mahalanobis: float = "auto",
                       critical_value_euclidean: float = "auto",
                       critical_value_manhattan: float = "auto",
                       show_distance: bool = True):
    print("For best results, make sure to handle all missing values first.")
    
    # Store initial dataset - First Copy
    data = dataset
    
    if drop_column is not None:
        # Drop columns for analysis
        dataset = dataset.drop(drop_column, axis = 1)
    
    # Calculating the mean and covariance matrix
    mean = emperical_mean(dataset)
    cov_matrix = emperical_covariance(dataset)
    
    # Store values
    store_distances = []
    
    # Scaling the data
    if scale_data is not None:
        if scale_data.upper() in ["MEDIANABSOLUTEDEVIATION", "IQR", "Z-SCORE", "MINMAXSCALER"]:
            # Using IQR
            if scale_data.upper() == "IQR":
                scaler = RobustScaler()
                dataset = scaler.fit_transform(dataset)
            
            # Using Median Absolute Deviation
            elif scale_data.upper() == "MEDIANABSOLUTEDEVIATION":
                dataset = median_absolute_deviation(dataset)
                
            # Using Z-score
            elif scale_data.upper() == "Z-SCORE":
                scaler = StandardScaler()
                dataset = scaler.fit_transform(dataset)
                
            # Using Normalization
            elif scale_data.upper() == "MINMAXSCALER":
                scaler = MinMaxScaler()
                dataset = scaler.fit_transform(dataset)
        else:
            raise ValueError("Metric for scaling must be one of ['MeanAbsoluteDeviation', 'IQR', 'Z-Score', 'MinMaxScaler']")
    
    # Calculating the euclidean distance
    if metric.lower() == "euclidean": 
        # Convert dataset to array
        if not isinstance(dataset, np.ndarray):
            dataset = dataset.values
        
        # Calculate x - µ            
        for index, val in enumerate(dataset):
            x_minus_mean = val - mean
            if squared_distance == True:
                euclidean_dist = np.dot(x_minus_mean, x_minus_mean)
            else:
                euclidean_dist = np.sqrt(np.dot(x_minus_mean, x_minus_mean))
            euclidean_dist = np.around(euclidean_dist, 2)
            store_distances.append(euclidean_dist)
    
    # Calculating the manhattan distance
    if metric.lower() == "manhattan":                        
        # Convert dataset to array
        if not isinstance(dataset, np.ndarray):
            dataset = dataset.values
        
        # Calculate x - µ            
        for index, val in enumerate(dataset):
            x_minus_mean = val - mean
            if squared_distance == True:
                raise ValueError("Manhattan doesn't have a squared distance")
            else:
                manhattan_dist = np.sum(np.abs(x_minus_mean))
            manhattan_dist = np.around(manhattan_dist, 2)
            store_distances.append(manhattan_dist)
    
    # Calculating the mahalanobis distance
    elif metric.lower() == "mahalanobis":                        
        # Convert dataset to array
        if not isinstance(dataset, np.ndarray):
            dataset = dataset.values
        
        # Calculate x - µ
        for index, val in enumerate(dataset):
            x_minus_mean = val - mean
            precision_matrix = np.linalg.inv(cov_matrix)
            if squared_distance == True:
                mahalanobis_dist = np.dot(np.dot(x_minus_mean.T, precision_matrix), x_minus_mean)
            else:
                mahalanobis_dist = np.sqrt(np.dot(np.dot(x_minus_mean.T, precision_matrix), x_minus_mean))
            mahalanobis_dist = np.around(mahalanobis_dist, 2)
            store_distances.append(mahalanobis_dist)
     
    data[f"{metric.lower().capitalize()} Distance"] = store_distances
    # Setting crtical values and removing the outliers
    if (remove_outliers_multivariate == True) and (critical_value_mahalanobis == "auto"):
        print("\nThe remove_outliers_multivariate command works only for the mahalanobis distance as a measure for outlier detection. Use the critical_value_euclidean and critical_value_manhattan parameters for specifying tresholds for the Euclidean and Manhattan distances\n")
        number_of_dimensions = data.shape[1]
        # The probability of getting a Type 1 error. 
        # That is, how likely we are to call a point an outlier when it isn't. 
        # This is what we are testing against with our Chi-Square distibution.
        # ---> Anything below the significance level, we reject it as being a type 1 error
        significance_level = alpha 
        # Calculating ppf - Inverse of cumulative_distribution_function
        ppf = chi2.ppf(q = (1 - significance_level), df = number_of_dimensions)
        ppf = np.around(ppf, 2)
        print(f"Using the Chi Square distribution for setting critical values to detect outliers, any point greater than {ppf}, will be considered an outlier.\n\n")
        
        condition = data[data["Mahalanobis Distance"] <= ppf]
        data_count_outliers = data.shape[0] - condition.shape[0]
        data = data[(data["Mahalanobis Distance"] < ppf)]
        print(f"{data_count_outliers} outliers detected and removed.")
    
    elif (remove_outliers_multivariate == True) and (critical_value_mahalanobis != "auto"):
        critical_value_mahalanobis = float(critical_value_mahalanobis)
        print("\nThe remove_outliers_multivariate command works only for the mahalanobis distance as a measure for outlier detection. Use the critical_value_euclidean and critical_value_manhattan parameters for specifying tresholds for the Euclidean and Manhattan distances\n")
        condition = data[data["Mahalanobis Distance"] <= critical_value_mahalanobis]
        data_count_outliers = data.shape[0] - condition.shape[0]
        data = data[(data["Mahalanobis Distance"] < critical_value_mahalanobis)]
        print(f"{data_count_outliers} outliers detected and removed.")
        
    if critical_value_euclidean != "auto":
        critical_value_euclidean = float(critical_value_euclidean)
        condition = data[data["Euclidean Distance"] <= critical_value_euclidean]
        data_count_outliers = data.shape[0] - condition.shape[0]
        data = data[(data["Euclidean Distance"] < critical_value_euclidean)]
        print(f"{data_count_outliers} outliers detected and removed.")
        
    if critical_value_manhattan != "auto":
        critical_value_manhattan = float(critical_value_manhattan)
        condition = data[data["Manhattan Distance"] <= critical_value_manhattan]
        data_count_outliers = data.shape[0] - condition.shape[0]
        data = data[(data["Manhattan Distance"] < critical_value_manhattan)]
        print(f"{data_count_outliers} outliers detected and removed.")
    
    # Condition to determine whether distances are displayed
    if show_distance != True:
        data = data.drop(f"{metric.lower().capitalize()} Distance", axis = 1)
        
    return data     
        
    
def check_outliers(dataset: pd.DataFrame,
                   univariate_method_for_columns: dict = None,
                   multivariate_metric: str = None,
                   alpha_isolation_forest: int = "auto",
                   n_estimators: int = 100,
                   max_samples: int = "auto",
                   max_features: float = 1.0,
                   bootstrap: bool = False,
                   n_jobs: int = None,
                   random_state: int = None,
                   warm_start: bool = False,
                   drop_column: list = None,
                   squared_distance: bool = False,
                   scale_data: str = None,
                   remove_multivariate_outliers: bool = False,
                   sig_level: float = 0.05,
                   specify_mahalanobis_critical_value: float = "auto",
                   specify_euclidean_critical_value: float = "auto",
                   specify_manhattan_critical_value: float = "auto",
                   show_distance: bool = True) -> pd.DataFrame:
            
    # Filtering outliers based on specified method
    # ---> Univariate Outlier Detectors
    if (multivariate_metric is None) and (univariate_method_for_columns is not None):
        if drop_column is not None:
            # Drop columns for analysis
            dataset = dataset.drop(drop_column, axis = 1)
            
        for col, method in univariate_method_for_columns.items():
            # BoxPlot - InterQuartile Range
            if method.upper() == "IQR":
                # Finding Q1 and Q3
                Q1 = dataset[col].quantile(0.25)
                Q3 = dataset[col].quantile(0.75)
                
                # Finding the Inter-Quantile Range
                IQR = Q3 - Q1
                
                # Creating whiskers
                lower_whisker = Q1 - (1.5 * IQR)
                upper_whisker = Q3 + (1.5 * IQR)
                
                dataset[col] = dataset[col].where((dataset[col] >= lower_whisker) & (dataset[col] <= upper_whisker))
            
            # Z-Score
            elif method.upper() == "Z-SCORE":
                # Finding the Mean and Standard Deviation
                col_mean = dataset[col].mean()
                col_std = dataset[col].std()
                
                # Applying specified functions for z-score
                dataset[col] = dataset[col].apply(z_score, args = (col_mean, col_std))
                dataset[col] = dataset[col].where((dataset[col] >= -3) & (dataset[col] <= 3), np.nan)
                dataset[col] = dataset[col].apply(values_z_score, args = (col_mean, col_std))
        
        # Count outliers
        outliers_in_columns = dataset.isnull().sum()
        total_outliers = dataset.isnull().sum().sum()
        
        print(f"\n\nTotal Outliers Found:  \n               {total_outliers}\n\n")
        print("Outliers in columns: \n")
        print(outliers_in_columns)
        
        # Return dataset after turning all outliers to nan
        return dataset
    
    elif (multivariate_metric is not None) and (univariate_method_for_columns is None):
        # ---> Multivariate Outlier Detectors
        if multivariate_metric.upper() in ["EUCLIDEAN", "MAHALANOBIS", "MANHATTAN", "ISOLATIONFOREST"]:
            # Euclidean
            if multivariate_metric.upper() == "EUCLIDEAN":
                data = distance_from_mean(dataset = dataset, 
                                   drop_column = drop_column,
                                   squared_distance = squared_distance,
                                   remove_outliers_multivariate = remove_multivariate_outliers,
                                   scale_data = scale_data,
                                   alpha = sig_level,
                                   critical_value_mahalanobis = specify_mahalanobis_critical_value,
                                   critical_value_euclidean = specify_euclidean_critical_value,
                                   critical_value_manhattan = specify_manhattan_critical_value,
                                   show_distance = show_distance)
            # Manhattan
            if multivariate_metric.upper() == "MANHATTAN":
                data = distance_from_mean(dataset = dataset, 
                                   drop_column = drop_column,
                                   squared_distance = squared_distance,
                                   metric = multivariate_metric,
                                   remove_outliers_multivariate = remove_multivariate_outliers,
                                   scale_data = scale_data,
                                   alpha = sig_level,
                                   critical_value_mahalanobis = specify_mahalanobis_critical_value,
                                   critical_value_euclidean = specify_euclidean_critical_value,
                                   critical_value_manhattan = specify_manhattan_critical_value,
                                   show_distance = show_distance)
            # Mahalanobis
            elif multivariate_metric.upper() == "MAHALANOBIS":
                data = distance_from_mean(dataset = dataset, 
                                   drop_column = drop_column,
                                   squared_distance = squared_distance,
                                   metric = multivariate_metric,
                                   remove_outliers_multivariate = remove_multivariate_outliers,
                                   scale_data = scale_data,
                                   alpha = sig_level,
                                   critical_value_mahalanobis = specify_mahalanobis_critical_value,
                                   critical_value_euclidean = specify_euclidean_critical_value,
                                   critical_value_manhattan = specify_manhattan_critical_value,
                                   show_distance = show_distance)
            
            # Isolation Forest
            elif multivariate_metric.upper() == "ISOLATIONFOREST":
                if drop_column is not None:
                    dataset = dataset.drop(drop_column, axis = 1)
                    
                # Scaling the data
                if scale_data is not None:
                    if scale_data.upper() in ["MEDIANABSOLUTEDEVIATION", "IQR", "Z-SCORE", "MINMAXSCALER"]:
                        # Using IQR
                        if scale_data.upper() == "IQR":
                            scaler = RobustScaler()
                            dataset = scaler.fit_transform(dataset)
                        
                        # Using Median Absolute Deviation
                        elif scale_data.upper() == "MEDIANABSOLUTEDEVIATION":
                            dataset = median_absolute_deviation(dataset)
                            
                        # Using Z-score
                        elif scale_data.upper() == "Z-SCORE":
                            scaler = StandardScaler()
                            dataset = scaler.fit_transform(dataset)
                            
                        # Using Normalization
                        elif scale_data.upper() == "MINMAXSCALER":
                            scaler = MinMaxScaler()
                            dataset = scaler.fit_transform(dataset)
                    else:
                        raise print("Metric for scaling must be one of ['MeanAbsoluteDeviation', 'IQR', 'Z-Score', 'MinMaxScaler']")
                        
                detector = IsolationForest(contamination = alpha_isolation_forest,
                                           n_estimators = n_estimators,
                                           max_samples = max_samples,
                                           max_features = max_features,
                                           bootstrap = bootstrap,
                                           n_jobs = n_jobs,
                                           random_state = random_state,
                                           warm_start = warm_start)
                print(dataset.columns)
                predict_outliers = detector.fit_predict(dataset)
                
                # Count the number of outliers
                num_outliers = (predict_outliers == -1).sum()
                print("Number of outliers using Isolation Forest:", num_outliers)
                
                dataset["Isolation Forest Outliers"] = predict_outliers
                return dataset
            
        # Return data after locating outliers in data cloud
        return data
    
    else:
        raise ValueError("Check that the parameters specified work for either univariate outlier removal only or multivarate outlier removal only")
        

def cummulative_frequency(column: pd.Series, sort: bool = False, axis_sort: int = -1):
    count = 0
    cf = []
    if sort == True:
        column = pd.Series(np.sort(column, axis = axis_sort))
    for each_value in column.values:
        if each_value == np.nan:
            count += 0
            cf.append(count)
        else:
            count += each_value
            cf.append(count)
    return pd.Series(cf, name = "Cummulative_Frequency")


def build_classifier_model(classifier, 
                           X_train: pd.DataFrame, 
                           y_train: pd.DataFrame, 
                           X_test: pd.DataFrame, 
                           y_test: pd.DataFrame, 
                           cross_validate_Xtrain: pd.DataFrame = None,
                           cross_validate_ytrain: pd.DataFrame = None,
                           kfold: int = 10,
                           pos_label: int = 1,
                           scoring_cross_validate: str or callable or list or tuple or dict = None,
                           scoring_cross_val_score: str or callable = None,
                           n_jobs: int = 1,
                           return_estimator: bool = True,
                           return_train_score: bool = True,
                           fp_and_fn: bool = False,
                           sample_weights: dict = None, 
                           probabilities: bool = False,
                           probability_threshold: float = 0.3) -> dict:
    
    # Model Training
    if (sample_weights is not None):
        model = classifier.fit(X_train, y_train, sample_weight = [sample_weights[label] for label in y_train])
    else:
        model = classifier.fit(X_train, y_train)
        
    
    # Model Prediction
    y_pred = model.predict(X_train) # Training Predictions: Check OverFitting
    y_pred1 = model.predict(X_test) # Test Predictions: Check Model Predictive Capacity
    
    # Predicting Probablities
    if hasattr(classifier, 'predict_proba') == True:
        y_pred_proba = model.predict_proba(X_train)
        y_pred1_proba = model.predict_proba(X_test)
    
    # Model Evalustion and Validation 
    # Training Evaluation: Check OverFitting
    training_analysis = confusion_matrix(y_train, y_pred)
    training_class_report = classification_report(y_train, y_pred)
    training_accuracy = accuracy_score(y_train, y_pred)
    training_precision = precision_score(y_train, y_pred, average='weighted', pos_label = pos_label)
    training_recall = recall_score(y_train, y_pred, average='weighted', pos_label = pos_label)
    training_f1_score = f1_score(y_train, y_pred, average='weighted', pos_label = pos_label)
    
    # Test Evaluations: Check Model Predictive Capacity
    test_analysis = confusion_matrix(y_test, y_pred1)
    test_class_report = classification_report(y_test, y_pred1)
    test_accuracy = accuracy_score(y_test, y_pred1)
    test_precision = precision_score(y_test, y_pred1, average='weighted', pos_label = pos_label)
    test_recall = recall_score(y_test, y_pred1, average='weighted', pos_label = pos_label)
    test_f1_score = f1_score(y_test, y_pred1, average='weighted', pos_label = pos_label)
    
    # Comparing false positives and false negatives for training and test data
    if fp_and_fn == True:
        store_data = {}
        data_Xtrain = X_train
        data_Xtest = X_test
        data_ytrain = y_train
        data_ytest = y_test
        data_ypred = y_pred
        data_ypred1 = y_pred1
        
        # Converting all to numpy for faster computations
        if isinstance(data_Xtrain, pd.DataFrame):
            data_Xtrain = data_Xtrain.to_numpy()
        if isinstance(data_Xtest, pd.DataFrame):
            data_Xtest = data_Xtest.to_numpy()
        if isinstance(data_ytrain, pd.Series):
            data_ytrain = data_ytrain.values
        if isinstance(data_ytest, pd.Series):
            data_ytest = data_ytest.values
        if isinstance(data_ypred, pd.Series):
            data_ypred = data_ypred.values
        if isinstance(data_ypred1, pd.Series):
            data_ypred1 = data_ypred1.values
            
        check_class_count = len(np.unique(data_ytrain))
        
        # Adding the predicted values and actual values to data_Xtrain and data_Xtest
        if hasattr(classifier, 'predict_proba') == True:
            data_Xtrain = np.hstack((data_Xtrain, y_pred_proba))
            data_Xtest = np.hstack((data_Xtest, y_pred1_proba))
        data_Xtrain = np.hstack((data_Xtrain, data_ytrain.reshape(-1, 1), data_ypred.reshape(-1, 1)))
        data_Xtest = np.hstack((data_Xtest, data_ytest.reshape(-1, 1), data_ypred1.reshape(-1, 1)))
        
        # Check if it is a binanry classification problem
        if check_class_count == 2:
            # --- Training Data
            train_fp = data_Xtrain[(data_Xtrain[:, -2] == 0) & (data_Xtrain[:, -1] == 1)]
            train_fn = data_Xtrain[(data_Xtrain[:, -2] == 1) & (data_Xtrain[:, -1] == 0)]
            
            # --- Test Data
            test_fp = data_Xtest[(data_Xtest[:, -2] == 0) & (data_Xtest[:, -1] == 1)]
            test_fn = data_Xtest[(data_Xtest[:, -2] == 1) & (data_Xtest[:, -1] == 0)]
            
            # --- Combined Data
            new_data_train = np.vstack((train_fp, train_fn))
            new_data_test = np.vstack((test_fp, test_fn))
            
            # Store the data
            store_data["Training FP and FN"] = {"False Positives": train_fp, "False Negatives": train_fn}
            store_data["Test FP and FN"] = {"False Positives": test_fp, "False Negatives": test_fn}
            store_data["ReTraining Data"] = new_data_train
            store_data["ReTesting Data"] = new_data_test
            store_data["X_train"] = data_Xtrain
        
        # Check if it is a multi-classification problem
        elif check_class_count > 2:
            get_unique = np.unique(data_ytrain)
            for value in get_unique:
                for each_value in get_unique:
                    if value != each_value:
                        # --- Training Data
                        train_fp = data_Xtrain[(data_Xtrain[:, -2] == value) & (data_Xtrain[:, -1] == each_value)]
                        
                        # --- Test Data
                        test_fp = data_Xtest[(data_Xtest[:, -2] == value) & (data_Xtest[:, -1] == each_value)]
                        
                        # Store the data
                        store_data[f"Class {value} | Class {each_value}"] = {"Training False Predictions": train_fp, "Test False Predictions": test_fp}
          
            
    # Getting probablities below a certain threshold for training and test data
    if probabilities == True:
        store_data_proba = {}
        data_Xtrain_proba = X_train
        data_Xtest_proba = X_test
        data_ytrain_proba = y_train
        data_ytest_proba = y_test
        data_ypred_proba = y_pred
        data_ypred1_proba = y_pred1
        
        # Converting all to numpy for faster computations
        if isinstance(data_Xtrain_proba, pd.DataFrame):
            data_Xtrain_proba = data_Xtrain_proba.to_numpy()
        if isinstance(data_Xtest_proba, pd.DataFrame):
            data_Xtest_proba = data_Xtest_proba.to_numpy()
        if isinstance(data_ytrain_proba, pd.Series):
            data_ytrain_proba = data_ytrain_proba.values
        if isinstance(data_ytest_proba, pd.Series):
            data_ytest_proba = data_ytest_proba.values
        if isinstance(data_ypred_proba, pd.Series):
            data_ypred_proba = data_ypred_proba.values
        if isinstance(data_ypred1_proba, pd.Series):
            data_ypred1_proba = data_ypred1_proba.values
            
        check_class_count = len(np.unique(data_ytrain_proba))
        
        # Adding the predicted values and actual values to data_Xtrain and data_Xtest
        if hasattr(classifier, 'predict_proba') == True:
            data_Xtrain_proba = np.hstack((data_Xtrain_proba, y_pred_proba))
            data_Xtest_proba = np.hstack((data_Xtest_proba, y_pred1_proba))
        data_Xtrain_proba = np.hstack((data_Xtrain_proba, data_ytrain_proba.reshape(-1, 1), data_ypred_proba.reshape(-1, 1)))
        data_Xtest_proba = np.hstack((data_Xtest_proba, data_ytest_proba.reshape(-1, 1), data_ypred1_proba.reshape(-1, 1)))
        
        # Check if it is a binary classification problem
        if check_class_count == 2:
            # --- Training Data
            train_fp_proba = data_Xtrain_proba[(data_Xtrain_proba[:, -4] >= probability_threshold) & (data_Xtrain_proba[:, -4] <= (1 - probability_threshold))]
            
            # --- Test Data
            test_fp_proba = data_Xtest_proba[(data_Xtest_proba[:, -4] >= probability_threshold) & (data_Xtest_proba[:, -4] <= (1 - probability_threshold))]
            
            # Store the data
            store_data_proba["ReTraining Data"] = train_fp_proba
            store_data_proba["ReTesting Data"] = test_fp_proba
            store_data_proba["X_train"] = data_Xtrain_proba
        
        # Check if it is a multi-classification problem
        # ---> NEEDS TO BE TESTED
        elif check_class_count > 2:
            get_unique = np.unique(data_ytrain_proba)
            for value in get_unique:
                for each_value in get_unique:
                    if value != each_value:
                        # --- Training Data
                        train_fp_proba = data_Xtrain_proba[(data_Xtrain_proba[:, -(len(get_unique) + 2):-2] <= probability_threshold)]
                        
                        # --- Test Data
                        test_fp_proba = data_Xtest_proba[(data_Xtest_proba[:, -(len(get_unique) + 2):-2] <= probability_threshold)]
                        
                        # Store the data
                        store_data_proba[f"Class {value} | Class {each_value}"] = {"Training Data": train_fp_proba, "Test Data": test_fp_proba}
            
    # Validation of Predictions
    if (cross_validate_Xtrain is None) and (cross_validate_ytrain is None):
        cross_val = cross_val_score(model, 
                                    X_train, 
                                    y_train, 
                                    cv = kfold,
                                    scoring = scoring_cross_val_score,
                                    n_jobs = n_jobs)  
        cross_validation = cross_validate(model, 
                                          X_train, 
                                          y_train, 
                                          cv = kfold, 
                                          return_estimator = return_estimator,
                                          return_train_score = return_train_score, 
                                          scoring = scoring_cross_validate,
                                          n_jobs = n_jobs)
        score_mean = round((cross_val.mean() * 100), 2)
        score_std_dev = round((cross_val.std() * 100), 2)
        return {
            "Model": model,
            "Predictions": {"Actual Training Y": y_train, 
                            "Actual Test Y": y_test, 
                            "Predicted Training Y": y_pred, 
                            "Predicted Test Y": y_pred1,
                            "Probabilities Training Y": y_pred_proba if hasattr(classifier, "predict_proba") == True else None,
                            "Probablilities Test Y": y_pred1_proba if hasattr(classifier, "predict_proba") == True else None,
                            },
            "Training Evaluation": {
                "Confusion Matrix": training_analysis,
                "Classification Report": training_class_report,
                "Model Accuracy": training_accuracy,
                "Model Precision": training_precision,
                "Model Recall": training_recall,
                "Model F1 Score": training_f1_score,
                },
            "Test Evaluation": {
                "Confusion Matrix": test_analysis,
                "Classification Report": test_class_report,
                "Model Accuracy": test_accuracy,
                "Model Precision": test_precision,
                "Model Recall": test_recall,
                "Model F1 Score": test_f1_score,
                },
            "Cross Validation": {
                "Cross Val Scores": cross_val,
                "Cross Validation Mean": score_mean, 
                "Cross Validation Standard Deviation": score_std_dev,
                "Validation Models": cross_validation
                },
            "False Positives and False Negatives": store_data if fp_and_fn == True else None,
            "Probabilities": store_data_proba if probabilities == True else None
            }
    
    else: 
        # Here, what we hope to achieve is the built model in cases of creating synthetic data
        # to be tested on the specified x_train and y_train which is most likely the data before
        # we fixed class imbablance.
        cross_val = cross_val_score(model, 
                                    cross_validate_Xtrain, 
                                    cross_validate_ytrain, 
                                    cv = kfold,
                                    scoring = scoring_cross_val_score,
                                    n_jobs = n_jobs)  
        cross_validation = cross_validate(model, 
                                          cross_validate_Xtrain, 
                                          cross_validate_ytrain, 
                                          cv = kfold, 
                                          return_estimator = return_estimator,
                                          return_train_score = return_train_score,
                                          scoring = scoring_cross_validate,
                                          n_jobs = n_jobs)
        score_mean = round((cross_val.mean() * 100), 2)
        score_std_dev = round((cross_val.std() * 100), 2)
        return {
            "Model": model,
            "Predictions": {"Actual Training Y": y_train, 
                            "Actual Test Y": y_test, 
                            "Predicted Training Y": y_pred, 
                            "Predicted Test Y": y_pred1,
                            "Probabilities Training Y": y_pred_proba if hasattr(classifier, "predict_proba") == True else None,
                            "Probablilities Test Y": y_pred1_proba if hasattr(classifier, "predict_proba") == True else None,
                            },
            "Training Evaluation": {
                "Confusion Matrix": training_analysis,
                "Classification Report": training_class_report,
                "Model Accuracy": training_accuracy,
                "Model Precision": training_precision,
                "Model Recall": training_recall,
                "Model F1 Score": training_f1_score,
                },
            "Test Evaluation": {
                "Confusion Matrix": test_analysis,
                "Classification Report": test_class_report,
                "Model Accuracy": test_accuracy,
                "Model Precision": test_precision,
                "Model Recall": test_recall,
                "Model F1 Score": test_f1_score,
                },
            "Cross Validation": {
                "Cross Val Scores": cross_val,
                "Cross Validation Mean": score_mean, 
                "Cross Validation Standard Deviation": score_std_dev,
                "Validation Models": cross_validation
                },
            "False Positives and False Negatives": store_data if fp_and_fn == True else None,
            "Probabilities": store_data_proba if probabilities == True else None
            }

def build_multiple_classifiers(classifiers: Union[list or tuple], 
                               x_train: pd.DataFrame, 
                               y_train: pd.DataFrame, 
                               x_test: pd.DataFrame, 
                               y_test: pd.DataFrame, 
                               kfold: int = 10,
                               pos_label: int = 1,
                               scoring_cross_val_score: str or callable = None,
                               n_jobs: int = 1,
                               return_estimator: bool = True,
                               return_train_score: bool = True,
                               fp_and_fn: bool = False) -> tuple:
    
    multiple_classifier_models = {} # General store for all metrics from each algorithm
    store_algorithm_metrics = [] # Store all metrics gotten from the algorithm at each iteration in the loop below
    dataframe = pd.DataFrame(columns = ["Algorithm",
                                        "Fit time",
                                        "Score time",
                                        "Test score",
                                        "Train score"]) # Store cross validation metrics
    
    # Creating a dataframe for all classifiers
    # ---> Loop through each classifier ain classifiers and do the following
    for algorithms in classifiers:
        store_cross_val_models = {}
        
        # Call the function build_classifier_model to get classifier metrics
        print(f"Building classifier model and metrics for {algorithms.__class__.__name__} model.")
        multiple_classifier_models[f"{algorithms.__class__.__name__}"] = build_classifier_model(classifier = algorithms, 
                                                                                                x_train = x_train, 
                                                                                                y_train = y_train, 
                                                                                                x_test = x_test, 
                                                                                                y_test = y_test, 
                                                                                                kfold = kfold,
                                                                                                pos_label = pos_label,
                                                                                                scoring_cross_val_score = scoring_cross_val_score,
                                                                                                n_jobs = n_jobs,
                                                                                                return_estimator = return_estimator,
                                                                                                return_train_score = return_train_score,
                                                                                                fp_and_fn = fp_and_fn)
        # Collecting individual metric to build algorithm dataframe
        training_accuracy = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Model Accuracy"]
        training_precision = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Model Precision"]
        training_recall = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Model Recall"]
        training_f1_score = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Model F1 Score"]
        test_accuracy = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Model Accuracy"]
        test_precision = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Model Precision"]
        test_recall = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Model Recall"]
        test_f1_score = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Model F1 Score"]
        cross_val_mean = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Cross Validation"]["Cross Validation Mean"]
        cross_val_std = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Cross Validation"]["Cross Validation Standard Deviation"]
        
        # Collecting indiviual metric to build cross validation dataframe
        cross_val_fit_time = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Cross Validation"]["Validation Models"]["fit_time"]
        cross_val_score_time = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Cross Validation"]["Validation Models"]["score_time"]
        cross_val_test_score = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Cross Validation"]["Validation Models"]["test_score"]
        cross_val_train_score = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Cross Validation"]["Validation Models"]["train_score"]
        
        # Storing all individual algorithm metrics from each iteration 
        store_algorithm_metrics.append([algorithms.__class__.__name__,
                                        training_accuracy,
                                        training_precision,
                                        training_recall,
                                        training_f1_score,
                                        test_accuracy,
                                        test_precision,
                                        test_recall,
                                        test_f1_score,
                                        cross_val_mean,
                                        cross_val_std])
        # Storing all individual cross validation metrics from each iteration 
        store_cross_val_models["Algorithm"] = algorithms.__class__.__name__
        store_cross_val_models["Fit time"] = cross_val_fit_time
        store_cross_val_models["Score time"] = cross_val_score_time
        store_cross_val_models["Test score"] = cross_val_test_score
        store_cross_val_models["Train score"] = cross_val_train_score
        # Creating dataframe for cross validation metric
        data_frame = pd.DataFrame(store_cross_val_models)
        dataframe = pd.concat([dataframe, data_frame])
        print("Model building completed.\n")
        
    # Creating dataframe for algorithm metric  
    df = pd.DataFrame(store_algorithm_metrics, columns = ["Algorithm",
                                                          "Training Accuracy",
                                                          "Training Precision",
                                                          "Training Recall",
                                                          "Training F1-Score",
                                                          "Test Accuracy",
                                                          "Test Precision",
                                                          "Test Recall",
                                                          "Test F1-Score",
                                                          "CV Mean",
                                                          "CV Standard Deviation"])
    # Save datasets in folder for analysis
    save_dataframe(dataset = dataframe, name = "Cross_Validation_Evaluation")
    save_dataframe(dataset = df, name = "Algorithm_Evaluation")
    return (df, dataframe, multiple_classifier_models)


def save_model_from_cross_validation(models_info: dict, algorithm: str, index: None):
    model_to_save = models_info[algorithm]["Cross Validation"]["Validation Models"]["estimator"][index]
    
    # Using Joblib to save the model in our folder
    joblib.dump(model_to_save, f"models/{algorithm}_Model_{index}.pkl")
    print(f"\nThis model is gotten from cross validating with the {algorithm} algorithm at iteration {index + 1}.")
    return models_info[algorithm]["Cross Validation"]["Validation Models"]["estimator"][index]


def save_dataframe(dataset: pd.DataFrame, name: str):
    """
    Save the data to the generated_data folder.

    Parameters
    ----------
    dataset : pd.DataFrame
        Dataset containing the information we want to save. For this project,
        it could be a dataframe of algorithm metrics or cross validation metrics.
    name: str
        A string indicating the name of the dataset and how it should be saved.

    Returns
    -------
    None.

    """
    try:
        data_name = name 
        date = time.strftime("%Y-%m-%d")
        dataset.to_csv(f"generated_data/{data_name}_{date}.csv", index = False)
        print("\nSuccessfully saved file to the specified folder ---> generated_data folder.")
    except FileNotFoundError:
        print("\nFailed to save file to the specified folder ---> generated_data folder.")


def create_group_clustering(dataset: pd.DataFrame,
                            columns: list,
                            n_clusters: int,
                            linkage: str):
    
    clusterer = AgglomerativeClustering(linkage = linkage, n_clusters = n_clusters)
    return clusterer.fit(dataset[columns]).labels_


def agg_two_groups(df: pd.DataFrame,
                  columns: list,
                  groupby: str,
                  agg_fun: dict):
    
    dataset = df[columns].groupby(groupby).agg(agg_fun).reset_index()
    mapping = {key: value for key, value in zip(dataset[columns[0]], dataset[columns[1]])}
    return mapping

def data_preprocessing_pipeline(dataset: pd.DataFrame,
                                drop_columns: list = None,
                                log_col: list = None):
    
    # Data Cleaning and Transformation
    dataset = pd.get_dummies(dataset,
                             columns = ["Geography", "Gender"],
                             drop_first=True,
                             dtype=np.int64)
    
    # Converting the card type to numeric
    card_hierarchy = {"SILVER": 0, "GOLD": 1, "PLATINUM": 2, "DIAMOND": 3}
    dataset = dataset.replace(card_hierarchy)
    
    # Dropping the complain column
    dataset = dataset.drop("Complain", axis=1)
    
    # Creating logrithmic columns
    if log_col is not None:
        for each_col in log_col:
            dataset[each_col] = np.log10(dataset[each_col])
    
    # Dropping columns
    if drop_columns is not None:
        dataset = dataset.drop(drop_columns, axis = 1)
            
    return dataset


def classifier(train_classifier, 
               retrain_classifier,
                X_train: pd.DataFrame, 
                y_train: pd.DataFrame, 
                X_test: pd.DataFrame, 
                y_test: pd.DataFrame, 
                cross_validate_Xtrain: pd.DataFrame = None,
                cross_validate_ytrain: pd.DataFrame = None,
                kfold: int = 10,
                pos_label: int = 1,
                scoring_cross_validate: str or callable or list or tuple or dict = None,
                scoring_cross_val_score: str or callable = None,
                n_jobs: int = 1,
                return_estimator: bool = True,
                return_train_score: bool = True,
                fp_and_fn: bool = False,
                sample_weights: dict = None, 
                probabilities: bool = False,
                probability_threshold: float = 0.3,
                retrain_prob_thresh: float = 0.3) -> dict:
    
    y_pred = []
    train = build_classifier_model(classifier = train_classifier,
                                    X_train = X_train, 
                                    y_train = y_train, 
                                    X_test = X_test, 
                                    y_test = y_test, 
                                    cross_validate_Xtrain = cross_validate_Xtrain,
                                    cross_validate_ytrain = cross_validate_ytrain,
                                    kfold = kfold,
                                    pos_label = pos_label,
                                    scoring_cross_validate = scoring_cross_validate,
                                    scoring_cross_val_score = scoring_cross_val_score,
                                    n_jobs = n_jobs,
                                    return_estimator = return_estimator,
                                    return_train_score = return_train_score,
                                    fp_and_fn = fp_and_fn,
                                    sample_weights = sample_weights, 
                                    probabilities = probabilities,
                                    probability_threshold = probability_threshold)
    train_model = train["Model"]
    

    retrain = build_classifier_model(classifier = retrain_classifier,
                                    X_train = X_train, 
                                    y_train = y_train, 
                                    X_test = X_test, 
                                    y_test = y_test, 
                                    cross_validate_Xtrain = cross_validate_Xtrain,
                                    cross_validate_ytrain = cross_validate_ytrain,
                                    kfold = kfold,
                                    pos_label = pos_label,
                                    scoring_cross_validate = scoring_cross_validate,
                                    scoring_cross_val_score = scoring_cross_val_score,
                                    n_jobs = n_jobs,
                                    return_estimator = return_estimator,
                                    return_train_score = return_train_score,
                                    fp_and_fn = fp_and_fn,
                                    sample_weights = sample_weights, 
                                    probabilities = probabilities,
                                    probability_threshold = probability_threshold)
    retrain_model = retrain["Model"]
    
    # Store models predictions
    if hasattr(train_classifier, "predict_proba"):
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        for each_row in X_test:
            prediction = train_model.predict_proba([each_row])
            print(prediction)
            if (prediction[0][0] >= retrain_prob_thresh) & (prediction[0][0] <= (1 - retrain_prob_thresh)):
                y_pred.append(retrain_model.predict([each_row]))
            else:
                y_pred.append(train_model.predict([each_row]))
                
    y_pred = np.array(y_pred)
    
    # Model Evalustion and Validation 
    training_analysis = confusion_matrix(y_test, y_pred)
    training_class_report = classification_report(y_test, y_pred)
    training_accuracy = accuracy_score(y_test, y_pred)
    training_precision = precision_score(y_test, y_pred, average='weighted', pos_label = pos_label)
    training_recall = recall_score(y_test, y_pred, average='weighted', pos_label = pos_label)
    training_f1_score = f1_score(y_test, y_pred, average='weighted', pos_label = pos_label)
    
    # Output
    return {
        "Model": [train_model, retrain_model],
        f"{train_model.__class__.__name__} Results": train,
        f"{retrain_model.__class__.__name__} Results": retrain,        
        "Predictions": {"Actual Y": y_test, 
                        "Predicted Y": y_pred, 
                        },
        "Training Evaluation": {
            "Confusion Matrix": training_analysis,
            "Classification Report": training_class_report,
            "Model Accuracy": training_accuracy,
            "Model Precision": training_precision,
            "Model Recall": training_recall,
            "Model F1 Score": training_f1_score,
            }
        }
    
    
    
    
    
    
    
