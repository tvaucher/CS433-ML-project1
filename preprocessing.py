"""Module containing functions for preprocessing the data"""

import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def z_normalize_data(x):
    """
    Helper function which performs Z-normalization on the columns of the data matrix given as an argument
    Assumes all the column features follow a Gaussian distribution

    :param x: data matrix, numpy ndarray with shape with shape (N, D),
              where N is the number of samples and D is the number of features

    :returns: numpy ndarray matrix with shape (N, D) with Z-normalized columns
    """

    mean_x, std_x = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
    x_norm = (x - mean_x) / std_x
    return x_norm


def min_max_normalize_data(x, new_min=0, new_max=1):
    """
    Helper function which performs min-max-normalization on the columns of the data matrix given as an argument.

    :param x: data matrix, numpy ndarray with shape with shape (N, D),
              where N is the number of samples and D is the number of features
    :param new_min: lower boundary of the new interval, float, optional, the default value is 0
    :param new_max: upper boundary of the new interval, float, optional, the default value is 1

    :returns: numpy ndarray matrix with shape (N, D) with min-max-normalized columns
    """

    # Calculate minimum and maximum value for each column/feature
    min_x, max_x = np.nanmin(x, axis=0), np.nanmax(x, axis=0)

    # Apply a transformation, such that the values in every column with be in the interval [new_min, new_max]
    x_norm = new_min + ((new_max - new_min) * (x - min_x) / (max_x - min_x))

    return x_norm


def split_data_by_categorical_column(y, x, ids, column_index):
    """
    Helper function to split the data into multiple smaller datasets based on the value of one categorical column

    :param y: vector of target classes, numpy array with dimensions (N, 1)
    :param x: data matrix, numpy ndarray with shape with shape (N, D),
              where N is the number of samples and D is the number of features
    :param ids: vector of ids for the samples, numpy array with dimensions (N, 1)
    :param column_index: index of the categorical column, integer value from 0 to D - 1

    :returns: (y_splits, x_splits, ids_splits), tuple of lists of reduced sets of target classes, data and ids,
                                                one set of classes, data and ids for each category
    """

    categories = np.sort(np.unique(x[:, column_index]))
    category_indices = [np.where(x[:, column_index] == category)[0] for category in categories]
    x_splits = [np.delete(x[indices, :], column_index, axis=1) for indices in category_indices]
    y_splits = [y[indices] for indices in category_indices]
    ids_splits = [ids[indices] for indices in category_indices]
    return y_splits, x_splits, ids_splits


def remove_na_columns(x):
    """
    Helper function that removes columns in the data which contain only NaN values

    :param x: data matrix, numpy ndarray with shape with shape (N, D),
              where N is the number of samples and D is the number of features

    :returns: cleaned data matrix
    """

    na_mask = np.isnan(x)
    zero_mask = x == 0
    na_columns = np.all(na_mask | zero_mask, axis=0)
    return x[:, ~na_columns]


def remove_low_variance_features(x, percentile):
    """
    Helper function that removes features in the data which have the lowest variance (below a percentile value)

    :param x: data matrix, numpy ndarray with shape with shape (N, D),
              where N is the number of samples and D is the number of features
    :param percentile: variance percentile, float value between 0 and 1

    :returns: cleaned data matrix
    """
    variances = np.nanvar(x, axis=0)
    variance_percentile = np.percentile(variances, percentile * 100)
    low_variance_mask = variances <= variance_percentile
    return x[:, ~low_variance_mask]


def remove_correlated_features(x, min_abs_correlation):
    """
    Helper function that removes linearly correlated features in the data using Pearson's correlation coefficients

    :param x: data matrix, numpy ndarray with shape with shape (N, D),
              where N is the number of samples and D is the number of features
    :param min_abs_correlation: threshold for the correlation coefficient in absolute value, float value between 0 and 1

    :returns: cleaned data matrix
    """

    variances = np.nanvar(x, axis=0)
    correlation_coefficients = np.ma.corrcoef(np.ma.masked_invalid(x), rowvar=False)
    rows, cols = np.where(abs(correlation_coefficients) > min_abs_correlation)
    columns_to_remove = []
    for i, j in zip(rows, cols):
        if i >= j:
            continue
        if variances[i] < variances[j] and i not in columns_to_remove:
            columns_to_remove.append(i)
        elif variances[j] < variances[i] and j not in columns_to_remove:
            columns_to_remove.append(j)
    return np.delete(x, columns_to_remove, axis=1)


def augment_features_polynomial_basis(x, degree=2):
    """
    Helper function that augments the data with polynomial degrees of features up to a maximum degree
    A column of ones is also added for the constant term

    :param x: data matrix, numpy ndarray with shape with shape (N, D),
              where N is the number of samples and D is the number of features
    :param degree: maximum polynomial degree, integer (minimum 1), optional, the default value is 2

    :returns: Phi matrix of augmented polynomial basis features, numpy ndarray with shape (N, 1 + D * degree)
    """

    n = x.shape[0]
    powers = [x ** deg for deg in range(1, degree + 1)]
    phi = np.concatenate((np.ones((n, 1)), *powers), axis=1)
    return phi


def preprocessing_pipeline(data, *, nan_value=-999., low_var_threshold=0.1, corr_threshold=0.85, degree=3):
    """
    Function that performs the whole preprocessing pipeline

    :param data: data matrix, numpy ndarray with shape with shape (N, D),
                 where N is the number of samples and D is the number of features
    :param nan_value: integer of float value in the data that is to be regarded as numpy.NaN
    :param low_var_threshold: variance percentile, float value between 0 and 1
    :param corr_threshold: threshold for the correlation coefficient in absolute value, float value between 0 and 1
    :param degree: maximum polynomial degree, integer (minimum 1)

    :returns: preprocessed data matrix
    """
    data = data.copy()
    data[data == nan_value] = np.nan
    data = remove_na_columns(data)
    data = remove_low_variance_features(data, low_var_threshold)
    data = remove_correlated_features(data, corr_threshold)
    data = z_normalize_data(data)
    data[np.isnan(data)] = 0.
    data = augment_features_polynomial_basis(data, degree)
    return data
