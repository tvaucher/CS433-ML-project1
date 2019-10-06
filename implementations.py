# Module containing all implementations of ML techniques required for the project

import numpy as np
from helpers import compute_gradient_mse, compute_subgradient_mae, compute_loss, batch_iter

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    ...

def least_squares(y, tx):
    ...
def least_squares_GD(y, x, initial_w, max_iters, gamma, mae=False):
    """
    Implementation of the Gradient Descent optimization algorithm for linear regression
    Can be run with both MSE and MAE loss

    :param x: data matrix, numpy ndarray with shape with shape (N, D),
              where N is the number of samples and D is the number of features
    :param y: vector of target values, numpy array with dimensions (N, 1)
    :param initial_w: vector of initial weights, numpy array with dimensions (D, 1)
    :param max_iters: how many iterations to run the algorithm, integer
    :param gamma: learning rate, positive float value
    :param mae: whether to use MAE loss, boolean, optional, the default value is False

    :returns: (final weights, final loss value), tuple
    """

    # Set the initial values for the weights
    w = initial_w

    for n_iter in range(max_iters):

        # Compute the total gradient (or subgradient if MAE loss is used)
        grd = compute_subgradient_mae(y, x, w) if mae else compute_gradient_mse(y, x, w)

        # Update the weights using the gradient and learning rate
        w = w - gamma * grd

    # Compute the final loss value
    loss = compute_loss(y, x, w, mae)

    return w, loss

def ridge_regression(y, tx, lambda_):
    ...

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    ...

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    ...