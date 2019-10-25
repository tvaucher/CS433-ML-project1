# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred

def binarize(y):
    y[y <= 0.5] = 0
    y[y > 0.5] = 1

def predict_log_labels(weights, data):
    pred = sigmoid(data @ weights)
    binarize(pred)
    return pred

def compute_accuracy(predict, targets):
    return np.mean(predict == targets)

def map_target_classes_to_boolean(y):
    """
    Helper function that transforms the target classes {-1, 1} into the standard boolean {0, 1}

    :param y: vector of target classes, numpy array with dimensions (N, 1)

    :returns: vector of boolean classes, numpy array with dimensions (N, 1)
    """

    return (y == 1).astype(int)


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

    return 1.0 / (1 + np.exp(-x))


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
    def safe_log(x, MIN=1e-9):
        """
        Return the stable floating log (in case where x was very small)
        """
        return np.log(np.maximum(x, MIN))
    
    predict = sigmoid(x @ w)
    log_pos, log_neg = safe_log(predict), safe_log(1 - predict)
    loss = -(y.T @ log_pos + (1 - y).T @ log_neg)
    loss += lambda_ * w.dot(w).squeeze()
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
    predict = sigmoid(x @ w)
    grd = x.T @ (predict - y)
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

def compute_loss_hinge(y, x, w, lambda_=0):
    return np.maximum(1 - y * (x @ w), 0).sum() + lambda_ / 2 * w.dot(w)
    
def compute_gradient_hinge(y, x, w, lambda_=0):
    mask = (y * (x @ w)) < 1
    grad = np.zeros_like(w)
    grad -= (y[:, None] * x)[mask, :].sum(axis=0)
    grad += lambda_ * w
    return grad