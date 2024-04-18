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
        data_count_null = dataset.isnull().sum()
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
        sns.pairplot(dataset) # Graph of correlation across each numerical feature
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
                           x_train: pd.DataFrame, 
                           y_train: pd.DataFrame, 
                           x_test: pd.DataFrame, 
                           y_test: pd.DataFrame, 
                           kfold: int = 10) -> dict:
    # Model Training
    model = classifier.fit(x_train, y_train)
    
    # Model Prediction
    y_pred = model.predict(x_train) # Training Predictions: Check OverFitting
    y_pred1 = model.predict(x_test) # Test Predictions: Check Model Predictive Capacity
    
    # Model Evalustion and Validation 
    # Training Evaluation: Check OverFitting
    training_analysis = confusion_matrix(y_train, y_pred)
    training_class_report = classification_report(y_train, y_pred)
    training_accuracy = accuracy_score(y_train, y_pred)
    training_precision = precision_score(y_train, y_pred, average='weighted')
    training_recall = recall_score(y_train, y_pred, average='weighted')
    training_f1_score = f1_score(y_train, y_pred, average='weighted')
    
    # Test Evaluations: Check Model Predictive Capacity
    test_analysis = confusion_matrix(y_test, y_pred1)
    test_class_report = classification_report(y_test, y_pred1)
    test_accuracy = accuracy_score(y_test, y_pred1)
    test_precision = precision_score(y_test, y_pred1, average='weighted')
    test_recall = recall_score(y_test, y_pred1, average='weighted')
    test_f1_score = f1_score(y_test, y_pred1, average='weighted')
    
    # Validation of Predictions
    cross_val = cross_val_score(model, x_train, y_train, cv = kfold)  
    cross_validation = cross_validate(model, 
                                      x_train, 
                                      y_train, 
                                      cv = kfold, 
                                      return_estimator = True,
                                      return_train_score = True)
    score_mean = round((cross_val.mean() * 100), 2)
    score_std_dev = round((cross_val.std() * 100), 2)
    return {
        "Model": model,
        "Predictions": {"Actual Training Y": y_train, 
                        "Actual Test Y": y_test, 
                        "Predicted Training Y": y_pred, 
                        "Predicted Test Y": y_pred1},
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
            "Cross Validation Mean": score_mean, 
            "Cross Validation Standard Deviation": score_std_dev,
            "Validation Models": cross_validation
            }
        }


def build_multiple_classifiers(classifiers: Union[list or tuple], 
                               x_train: pd.DataFrame, 
                               y_train: pd.DataFrame, 
                               x_test: pd.DataFrame, 
                               y_test: pd.DataFrame, 
                               kfold: int = 10) -> tuple:
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
                                                                                                kfold = kfold)
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
        

    