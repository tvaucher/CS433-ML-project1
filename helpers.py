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

    mean_x, std_x = np.mean(x, axis=0), np.std(x, axis=0)
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
    min_x, max_x = np.min(x, axis=0), np.max(x, axis=0)

    # Generate row vectors with the new minimum and maximum values
    new_min_x = new_min * np.ones(min_x.shape, dtype="float")
    new_max_x = new_max * np.ones(max_x.shape, dtype="float")

    # Apply a transformation, such that the values in every column with be in the interval [new_min, new_max]
    x_norm = new_min_x + ((new_max_x - new_min_x) * (x - min_x) / (max_x - min_x))

    return x_norm


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

    :param y: vector of target values, numpy array with dimensions (D, 1)
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
