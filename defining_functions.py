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
from sklearn.model_selection import (cross_val_score,
                                     cross_validate,
                                     RepeatedStratifiedKFold)
from sklearn.metrics import (confusion_matrix,
                             classification_report,
                             accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score)
from imblearn.base import BaseSampler
from typing import Union, Callable, List, Tuple, Dict, Optional
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


def cummulative_frequency(column: pd.Series,
                          sort: bool = False,
                          axis_sort: int = -1):
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
                            X_val: pd.DataFrame,
                            y_val: pd.DataFrame,
                            cross_validate_Xtrain: Optional[pd.DataFrame],
                            cross_validate_ytrain: Optional[pd.DataFrame],
                            repeatedstratifiedkfold: int = 10,
                            repeatedstratifiedresampler: Optional[BaseSampler] = None,
                            n_repeats_stratified: int = 1,
                            scoring_cross_validate: Union[str, Callable, List[str], Tuple[str], Dict[str, str]] = None,
                            scoring_cross_val_score: Union[str, Callable] = None,
                            n_jobs: int = -1,
                            return_estimator: bool = True,
                            return_train_score: bool = True,
                            sample_weights: Optional[Dict] = None,
                            cv_random_state: int = None) -> dict:

    # Model Training
    if (sample_weights is not None):
        model = classifier.fit(X_train, y_train, sample_weight = [sample_weights[label] for label in y_train])
    else:
        model = classifier.fit(X_train, y_train)


    # Model Prediction
    y_pred = model.predict(X_train) # Training Predictions: Check OverFitting
    y_pred1 = model.predict(X_test) # Test Predictions: Check Model Predictive Capacity
    y_pred2 = model.predict(X_val) # Validation Predictions: For retraining the model

    # Predicting Probablities
    if hasattr(classifier, 'predict_proba') == True:
        y_pred_proba = model.predict_proba(X_train)
        y_pred1_proba = model.predict_proba(X_test)
        y_pred2_proba = model.predict_proba(X_val)

    # Model Evalustion and Validation
    # Training Evaluation: Check OverFitting
    training_analysis = confusion_matrix(y_train, y_pred)
    training_class_report = classification_report(y_train, y_pred)
    training_accuracy = accuracy_score(y_train, y_pred)
    training_precision = precision_score(y_train, y_pred, average='weighted')
    training_recall = recall_score(y_train, y_pred, average='weighted')
    training_f1_score = f1_score(y_train, y_pred, average='weighted')
    training_tpr = training_analysis[1, 1] / np.sum(training_analysis[1, :])
    training_tnr = training_analysis[0, 0] / np.sum(training_analysis[0, :])
    training_precision_1 = training_analysis[1, 1] / np.sum(training_analysis[:, 1])
    training_precision_0 = training_analysis[0, 0] / np.sum(training_analysis[:, 0])

    # Test Evaluations: Check Model Predictive Capacity
    test_analysis = confusion_matrix(y_test, y_pred1)
    test_class_report = classification_report(y_test, y_pred1)
    test_accuracy = accuracy_score(y_test, y_pred1)
    test_precision = precision_score(y_test, y_pred1, average='weighted')
    test_recall = recall_score(y_test, y_pred1, average='weighted')
    test_f1_score = f1_score(y_test, y_pred1, average='weighted')
    test_tpr = test_analysis[1, 1] / np.sum(test_analysis[1, :])
    test_tnr = test_analysis[0, 0] / np.sum(test_analysis[0, :])
    test_precision_1 = test_analysis[1, 1] / np.sum(test_analysis[:, 1])
    test_precision_0 = test_analysis[0, 0] / np.sum(test_analysis[:, 0])

    # Validation Evaluations: Check Model Predictive Capacity
    val_analysis = confusion_matrix(y_val, y_pred2)
    val_class_report = classification_report(y_val, y_pred2)
    val_accuracy = accuracy_score(y_val, y_pred2)
    val_precision = precision_score(y_val, y_pred2, average='weighted')
    val_recall = recall_score(y_val, y_pred2, average='weighted')
    val_f1_score = f1_score(y_val, y_pred2, average='weighted')
    val_tpr = val_analysis[1, 1] / np.sum(val_analysis[1, :])
    val_tnr = val_analysis[0, 0] / np.sum(val_analysis[0, :])
    val_precision_1 = val_analysis[1, 1] / np.sum(val_analysis[:, 1])
    val_precision_0 = val_analysis[0, 0] / np.sum(val_analysis[:, 0])

    # Validation of Predictions
    kfold = RepeatedStratifiedKFold(n_splits = repeatedstratifiedkfold, random_state = cv_random_state, n_repeats = n_repeats_stratified)
    cross_val_analysis = {}
    count = 0

    # Convert pandas series to numpy array
    if isinstance(cross_validate_ytrain, pd.Series):
        cross_validate_ytrain = cross_validate_ytrain.reset_index(drop = True)
    if isinstance(cross_validate_Xtrain, pd.DataFrame):
        cross_validate_Xtrain = np.array(cross_validate_Xtrain)

    # Enumerate splits
    for train_ix, test_ix in kfold.split(cross_validate_Xtrain, cross_validate_ytrain):
        # get data
        train_X, test_X = cross_validate_Xtrain[train_ix], cross_validate_Xtrain[test_ix]
        train_y, test_y = cross_validate_ytrain[train_ix], cross_validate_ytrain[test_ix]

        # Resampling minority class
        # ---> Resampling the training data seperately
        try:
            resampler_train = repeatedstratifiedresampler
            X_resampled, y_resampled = resampler_train.fit_resample(train_X, train_y)
        except:
            X_resampled, y_resampled = train_X, train_y

        # Value counts for the target
        count_target_resampled_train = y_resampled.value_counts()

        # Fit the model
        val_model = classifier.fit(X_resampled, y_resampled)

        # Make predictions
        val_model_predictions = val_model.predict(X_test)
        model_predictions = val_model.predict(test_X)
        predictions = val_model.predict(X_val)


        # Test Evaluations: Check Model Predictive Capacity
        model_analysis = confusion_matrix(y_test, val_model_predictions)
        model_class_report = classification_report(y_test, val_model_predictions)
        model_accuracy = accuracy_score(y_test, val_model_predictions)
        model_precision = precision_score(y_test, val_model_predictions, average='weighted')
        model_recall = recall_score(y_test, val_model_predictions, average='weighted')
        model_f1_score = f1_score(y_test, val_model_predictions, average='weighted')
        model_tpr = model_analysis[1, 1] / np.sum(model_analysis[1, :])
        model_tnr = model_analysis[0, 0] / np.sum(model_analysis[0, :])
        model_precision_1 = model_analysis[1, 1] / np.sum(model_analysis[:, 1])
        model_precision_0 = model_analysis[0, 0] / np.sum(model_analysis[:, 0])

        # Test evaluation for each fold
        kfold_model_analysis = confusion_matrix(test_y, model_predictions)
        kfold_model_class_report = classification_report(test_y, model_predictions)

        # Validation data evaluation for each fold
        val_model_analysis = confusion_matrix(y_val, predictions)
        val_model_class_report = classification_report(y_val, predictions)

        cross_val_analysis[count] = {}
        cross_val_analysis[count]["model"] = val_model
        cross_val_analysis[count]["y_train"] = y_resampled
        cross_val_analysis[count]["X_train"] = X_resampled
        cross_val_analysis[count]["y_test"] = test_y
        cross_val_analysis[count]["y_pred"] = val_model_predictions
        cross_val_analysis[count]["count_target_categories"] = count_target_resampled_train
        cross_val_analysis[count]["confusion_matrix"] = model_analysis
        cross_val_analysis[count]["confusion_matrix_kfold"] = kfold_model_analysis
        cross_val_analysis[count]["confusion_matrix_val_data"] = val_model_analysis
        cross_val_analysis[count]["classification_report"] = model_class_report
        cross_val_analysis[count]["classification_report_kfold"] = kfold_model_class_report
        cross_val_analysis[count]["classification_report_val_data"] = val_model_class_report
        cross_val_analysis[count]["accuracy"] = model_accuracy
        cross_val_analysis[count]["precision"] = model_precision
        cross_val_analysis[count]["recall"] = model_recall
        cross_val_analysis[count]["f1_score"] = model_f1_score
        cross_val_analysis[count]["true_positive_rate"] = model_tpr
        cross_val_analysis[count]["true_negative_rate"] = model_tnr
        cross_val_analysis[count]["false_positive_rate"] = model_precision_1
        cross_val_analysis[count]["false_negative_rate"] = model_precision_0

        # Increment the value of count
        count += 1

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
                        "Probablilities Validation Y": y_pred2_proba if hasattr(classifier, "predict_proba") == True else None,
                        },
        "Training Evaluation": {
            "Confusion Matrix": training_analysis,
            "Classification Report": training_class_report,
            "Model Accuracy": training_accuracy,
            "Model Precision": training_precision,
            "Model Recall": training_recall,
            "Model F1 Score": training_f1_score,
            "Model True Positive Rate": training_tpr,
            "Model True Negative Rate": training_tnr,
            "Model Precision (C_1)": training_precision_1,
            "Model Precision (C_0)": training_precision_0,
            },
        "Test Evaluation": {
            "Confusion Matrix": test_analysis,
            "Classification Report": test_class_report,
            "Model Accuracy": test_accuracy,
            "Model Precision": test_precision,
            "Model Recall": test_recall,
            "Model F1 Score": test_f1_score,
            "Model True Positive Rate": test_tpr,
            "Model True Negative Rate": test_tnr,
            "Model Precision (C_1)": test_precision_1,
            "Model Precision (C_0)": test_precision_0,
            },
        "Validation Data Evaluation": {
            "Confusion Matrix": val_analysis,
            "Classification Report": val_class_report,
            "Model Accuracy": val_accuracy,
            "Model Precision": val_precision,
            "Model Recall": val_recall,
            "Model F1 Score": val_f1_score,
            "Model True Positive Rate": val_tpr,
            "Model True Negative Rate": val_tnr,
            "Model Precision (C_1)": val_precision_1,
            "Model Precision (C_0)": val_precision_0,
            },
        "Cross Validation": {
            "Cross Val Scores": cross_val,
            "Cross Validation Mean": score_mean,
            "Cross Validation Standard Deviation": score_std_dev,
            "Validation Models": cross_validation,
            "Validation Predictions": cross_val_analysis
            },
        }

def build_multiple_classifiers(classifiers: Union[List, Tuple],
                                X_train: pd.DataFrame,
                                y_train: pd.DataFrame,
                                X_test: pd.DataFrame,
                                y_test: pd.DataFrame,
                                X_val: pd.DataFrame,
                                y_val: pd.DataFrame,
                                cross_validate_Xtrain: Optional[pd.DataFrame] = None,
                                cross_validate_ytrain: Optional[pd.DataFrame] = None,
                                repeatedstratifiedkfold: int = 10,
                                repeatedstratifiedresampler: Optional[BaseSampler] = None,
                                n_repeats_stratified: int = 1,
                                scoring_cross_val_score: Union[str, Callable] = None,
                                n_jobs: int = -1,
                                return_estimator: bool = True,
                                return_train_score: bool = True,
                                sample_weights: Optional[Dict] = None,
                                figsize: tuple = (20, 10),
                                cv_random_state: int = None) -> Tuple:

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
                                                                                                X_train = X_train,
                                                                                                y_train = y_train,
                                                                                                X_test = X_test,
                                                                                                y_test = y_test,
                                                                                                X_val = X_val,
                                                                                                y_val = y_val,
                                                                                                repeatedstratifiedkfold = repeatedstratifiedkfold,
                                                                                                repeatedstratifiedresampler = repeatedstratifiedresampler,
                                                                                                n_repeats_stratified = n_repeats_stratified,
                                                                                                scoring_cross_val_score = scoring_cross_val_score,
                                                                                                n_jobs = n_jobs,
                                                                                                return_estimator = return_estimator,
                                                                                                return_train_score = return_train_score,
                                                                                                cross_validate_Xtrain = cross_validate_Xtrain,
                                                                                                cross_validate_ytrain = cross_validate_ytrain,
                                                                                                sample_weights = sample_weights,
                                                                                                cv_random_state = cv_random_state)
        # Collecting individual metric to build algorithm dataframe
        training_accuracy = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Model Accuracy"]
        training_precision = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Model Precision"]
        training_recall = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Model Recall"]
        training_f1_score = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Model F1 Score"]
        training_tpr = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Model True Positive Rate"]
        training_tnr = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Model True Negative Rate"]
        training_precision_1 = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Model Precision (C_1)"]
        training_precision_0 = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Training Evaluation"]["Model Precision (C_0)"]
        test_accuracy = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Model Accuracy"]
        test_precision = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Model Precision"]
        test_recall = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Model Recall"]
        test_f1_score = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Model F1 Score"]
        test_tpr = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Model True Positive Rate"]
        test_tnr = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Model True Negative Rate"]
        test_precision_1 = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Model Precision (C_1)"]
        test_precision_0 = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Test Evaluation"]["Model Precision (C_0)"]
        val_accuracy = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Validation Data Evaluation"]["Model Accuracy"]
        val_precision = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Validation Data Evaluation"]["Model Precision"]
        val_recall = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Validation Data Evaluation"]["Model Recall"]
        val_f1_score = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Validation Data Evaluation"]["Model F1 Score"]
        val_tpr = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Validation Data Evaluation"]["Model True Positive Rate"]
        val_tnr = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Validation Data Evaluation"]["Model True Negative Rate"]
        val_precision_1 = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Validation Data Evaluation"]["Model Precision (C_1)"]
        val_precision_0 = multiple_classifier_models[f"{algorithms.__class__.__name__}"]["Validation Data Evaluation"]["Model Precision (C_0)"]
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
                                        training_tpr,
                                        training_tnr,
                                        training_precision_1,
                                        training_precision_0,
                                        test_accuracy,
                                        test_precision,
                                        test_recall,
                                        test_f1_score,
                                        test_tpr,
                                        test_tnr,
                                        test_precision_1,
                                        test_precision_0,
                                        val_accuracy,
                                        val_precision,
                                        val_recall,
                                        val_f1_score,
                                        val_tpr,
                                        val_tnr,
                                        val_precision_1,
                                        val_precision_0,
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
                                                          "Training True Positive Rate",
                                                          "Training True Negative Rate",
                                                          "Training Precision (C_1)",
                                                          "Training Precision (C_0)",
                                                          "Test Accuracy",
                                                          "Test Precision",
                                                          "Test Recall",
                                                          "Test F1-Score",
                                                          "Test True Positive Rate",
                                                          "Test True Negative Rate",
                                                          "Test Precision (C_1)",
                                                          "Test Precision (C_0)",
                                                          "Validation Data Accuracy",
                                                          "Validation Data Precision",
                                                          "Validation Data Recall",
                                                          "Validation Data F1-Score",
                                                          "Validation Data True Positive Rate",
                                                          "Validation Data True Negative Rate",
                                                          "Validation Data Precision (C_1)",
                                                          "Validation Data Precision (C_0)",
                                                          "CV Mean",
                                                          "CV Standard Deviation"])
    for each_col in df.columns:
        if each_col != "Algorithm":
            df = df.sort_values(each_col, ascending = False)
            cm = plt.cm.Purples_r(np.linspace(0.2, 1, df.shape[1]))
            plt.figure(figsize = figsize)
            bar_chart = plt.bar(x = df.Algorithm, height = each_col, color = cm, data = df)
            plt.bar_label(container = bar_chart, labels = round(df[each_col], 2))
            plt.title(f"Analyzing Results of {each_col} for each Algorithm")
            plt.xlabel("Algorithms")
            plt.ylabel(f"{each_col}")
            plt.tight_layout()
            plt.show()

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


def data_preprocessing_pipeline(dataset: pd.DataFrame,
                                drop_columns: list = None,
                                log_col: list = None,
                                dummy_col: list = None,
                                replace_val: dict = None):
    # Data Cleaning and Transformation
    if dummy_col is not None:
        dataset = pd.get_dummies(dataset,
                                 columns = dummy_col,
                                 drop_first=True,
                                 dtype=np.int64)

    # Converting the card type to numeric
    if replace_val is not None:
        dataset = dataset.replace(replace_val)

    # Creating logrithmic columns
    if log_col is not None:
        for each_col in log_col:
            dataset[f"Log_{each_col}"] = np.log1p(dataset[each_col])

    # Dropping columns
    if drop_columns is not None:
        dataset = dataset.drop(drop_columns, axis = 1)

    return dataset
