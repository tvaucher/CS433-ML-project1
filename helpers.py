# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
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


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


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


def split_data_by_categorical_column(x, column_index):
    """
    Helper function to split the data into multiple smaller datasets based on the value of one categorical column

    :param x: data matrix, numpy ndarray with shape with shape (N, D),
              where N is the number of samples and D is the number of features
    :param column_index: index of the categorical column, integer value from 0 to D - 1

    :returns: list of reduced datasets, one dataset for each category
    """

    categories = np.sort(np.unique(x[:, column_index]))
    category_indices = [np.where(x[:, column_index] == category)[0] for category in categories]
    splits = [np.delete(x[indices, :], column_index, axis=1) for indices in category_indices]
    return splits


def augment_features_polynomial_basis(x, degree=2):
    """
    Helper function that augments the data with polynomial degrees of features up to a maximum degree
    A column of ones is also added for the constant term

    :param x: data matrix, numpy ndarray with shape with shape (N, D),
              where N is the number of samples and D is the number of features
    :param degree: maximum polynomial degree, integer (minimum 2), optional, the default value is 2

    :returns: Phi matrix of augmented polynomial basis features, numpy ndarray with shape (N, 1 + D * degree)
    """

    n = x.shape[0]
    phi = np.hstack((np.ones((n, 1), dtype=float), x))
    for deg in range(2, degree + 1):
        phi = np.hstack((phi, x ** deg))
    return phi


def compute_loss(y, x, w, mae=False):
    """
    Helper function that calculates MSE or MAE loss

    :param y: vector of target values, numpy array with dimensions (N, 1)
    :param x: data matrix, numpy ndarray with shape with shape (N, D),
              where N is the number of samples and D is the number of features
    :param w: vector of weights, numpy array with dimensions (D, 1)
    :param mae: whether to use MAE loss instead of MSE, boolean, optional, the default value is False

    :returns: loss value, float
    """

    n = x.shape[0]
    e = y - np.matmul(x, w)
    if mae:
        loss = np.sum(abs(e)) / n
    else:
        loss = np.sum(e ** 2) / (2 * n)
    return loss


def compute_gradient_mse(y, x, w):
    """
    Helper function that computes the gradient of the MSE loss function

    :param y: vector of target values, numpy array with dimensions (N, 1)
    :param x: data matrix, numpy ndarray with shape with shape (N, D),
              where N is the number of samples and D is the number of features
    :param w: vector of weights, numpy array with dimensions (D, 1)

    :returns: vector of gradients of the weights, numpy array with dimensions (D, 1)
    """

    n = x.shape[0]
    e = y - np.matmul(x, w)
    grd = (-1 / n) * np.matmul(np.transpose(x), e)
    return grd


def compute_subgradient_mae(y, x, w):
    """
    Helper function that computes a subgradient of the MAE loss function
    The subgradient value 0 is returned for the non-differentiable point at 0

    :param y: vector of target values, numpy array with dimensions (N, 1)
    :param x: data matrix, numpy ndarray with shape with shape (N, D),
              where N is the number of samples and D is the number of features
    :param w: vector of weights, numpy array with dimensions (D, 1)

    :returns: vector of gradients of the weights, numpy array with dimensions (D, 1)
    """

    n = x.shape[0]
    e = y - np.matmul(x, w)
    grd = (-1 / n) * np.matmul(np.transpose(x), np.sign(e))
    return grd


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def sigmoid(x):
    """
    Helper function that calculates the sigmoid function for a single input or an array of inputs

    :param x: input data, single float or one-dimensional numpy array

    :returns: sigmoid value(s), single float or one-dimensional numpy array
    """

    return 1 / (1 + np.exp(-x))


def map_target_classes_to_boolean(y):
    """
    Helper function that transforms the target classes {-1, 1} into the standard boolean {0, 1}

    :param y: vector of target classes, numpy array with dimensions (N, 1)

    :returns: vector of boolean classes, numpy array with dimensions (N, 1)
    """

    return (y == 1).astype(int)


def compute_loss_nll(y, x, w, lambda_=0):
    """
    Helper function that calculates the negative log likelihood loss for logistic regression
    Can also optionally include regularization

    :param y: vector of target classes, numpy array with dimensions (N, 1)
    :param x: data matrix, numpy ndarray with shape with shape (N, D),
              where N is the number of samples and D is the number of features
    :param w: vector of weights, numpy array with dimensions (D, 1)
    :param lambda_: regularization coefficient, positive float value, optional, the default value is 0

    :returns: negative log likelihood loss value, float
    """

    sgm = sigmoid(np.matmul(x, w))
    log_pos, log_neg = np.log(sgm), np.log(1 - sgm)
    loss = - np.matmul(np.transpose(y), log_pos) - np.matmul(np.transpose(1 - y), log_neg)
    loss += lambda_ * np.matmul(np.transpose(w), w)
    return loss


def compute_gradient_nll(y, x, w, lambda_=0):
    """
    Helper function that calculates the gradient of the negative log likelihood loss function for logistic regression
    Can also optionally include regularization

    :param y: vector of target classes, numpy array with dimensions (N, 1)
    :param x: data matrix, numpy ndarray with shape with shape (N, D),
              where N is the number of samples and D is the number of features
    :param w: vector of weights, numpy array with dimensions (D, 1)
    :param lambda_: regularization coefficient, positive float value, optional, the default value is 0

    :returns: vector of gradients of the weights, numpy array with dimensions (D, 1)
    """

    e = y - sigmoid(np.matmul(x, w))
    grd = - np.matmul(np.transpose(x), e)
    grd += 2 * lambda_ * w
    return grd


def compute_hessian_nll(y, x, w, lambda_=0):
    """
    Helper function that calculates the Hessian matrix of the negative log likelihood loss function for logistic regression
    Can also optionally include regularization

    :param y: vector of target classes, numpy array with dimensions (N, 1)
    :param x: data matrix, numpy ndarray with shape with shape (N, D),
              where N is the number of samples and D is the number of features
    :param w: vector of weights, numpy array with dimensions (D, 1)
    :param lambda_: regularization coefficient, positive float value, optional, the default value is 0

    :returns: Hessian matrix, numpy ndarray with dimensions (D, D)
    """

    sgm = sigmoid(x.dot(w)).reshape(-1)
    s = np.diag(sgm * (1 - sgm) + 2 * lambda_)
    return np.matmul(np.matmul(np.transpose(x), s), x)


def train_test_split_data(y, x, ratio, seed):
    """
    Helper function that splits data samples randomly into train and test sets by a given ratio

    :param y: vector of target values, numpy array with dimensions (N, 1)
    :param x: data matrix, numpy ndarray with shape with shape (N, D),
              where N is the number of samples and D is the number of features
    :param ratio: fraction of data samples that will be selected for trainings, float value between 0 and 1
    :param seed: seeding value for the random generator, integer

    :returns: tuple (x_train, y_train, x_test, y_test),
              containts samples and targets in train set, followed by samples and targets in test set
    """

    # Set the seed
    np.random.seed(seed)

    # Use the ratio to find the number of samples in the train set as an integer by rounding if necessary
    n = x.shape[0]
    num_train_samples = int(ratio * n)

    # Shuffle and split the sample indices into train and test parts
    indices = np.random.permutation(n)
    train_indices = indices[:num_train_samples]
    test_indices = indices[num_train_samples:]

    # Generate the output sets
    x_train, y_train = x[train_indices], y[train_indices]
    x_test, y_test = x[test_indices], y[test_indices]
    return x_train, y_train, x_test, y_test


def k_fold_cross_split_data(y, x, k, seed):
    """
    Helper function that splits data samples randomly into k folds, to be used afterwards for cross-validation

    :param y: vector of target values, numpy array with dimensions (N, 1)
    :param x: data matrix, numpy ndarray with shape with shape (N, D),
              where N is the number of samples and D is the number of features
    :param k: number of folds, positive integer
    :param seed: seeding value for the random generator, integer

    :returns: splits, list of tuples of the form (x_train, y_train, x_test, y_test),
              where each tuple contains samples and targets in train set, followed by samples and targets in test set
    """

    # Set the seed
    np.random.seed(seed)

    # Calculate the number of samples in each fold as an integer by rounding if necessary
    n = x.shape[0]
    fold_size = int(n / k)

    # Shuffle and split the sample indices into k folds
    indices = np.random.permutation(n)
    k_indices = [indices[i * fold_size: (i + 1) * fold_size] for i in range(k)]

    # Generate the k different train-test splits of the data using the indices
    # in each of the k folds as test indices and the rest as train indices
    splits = []
    for i, k_index in enumerate(k_indices):
        y_test, x_test = y[k_index], x[k_index]
        k_indices_without_test = k_indices[:i] + k_indices[i + 1:]
        train_indices = [index for k_index in k_indices_without_test for index in k_index]
        y_train, x_train = y[train_indices], x[train_indices]
        splits.append((x_train, y_train, x_test, y_test))
    return splits
