# Module containing all implementations of ML techniques required for the project

import numpy as np
from helpers import compute_gradient_mse, compute_subgradient_mae, compute_loss, batch_iter


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


def least_squares_SGD(y, x, initial_w, max_iters, gamma, mae=False):
    """
    Implementation of the Stochastic Gradient Descent optimization algorithm for linear regression
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

    # Use the helper function batch_iter from Exercise 2,
    # to get a random sample from the data in the form (y_n, x_n) for each iteration
    batch_iterator = batch_iter(y, x, batch_size=1, num_batches=max_iters)

    for y_n, x_n in batch_iterator:
        # Compute the gradient for only one sample (or subgradient if MAE loss is used)
        grd = compute_subgradient_mae(y_n, x_n, w) if mae else compute_gradient_mse(y_n, x_n, w)

        # Update the weights using the gradient and learning rate
        w = w - gamma * grd

    # Compute the final loss value
    loss = compute_loss(y, x, w, mae)

    return w, loss


def least_squares(y, x):
    """
    Calculate the least squares solution explicitly using the normal equations

    :param x: data matrix, numpy ndarray with shape with shape (N, D),
              where N is the number of samples and D is the number of features
    :param y: vector of target values, numpy array with dimensions (N, 1)

    :raises AssertionError: if the Gram matrix has no inverse

    :returns: (weights, loss value), tuple
    """

    # Compute the Gram matrix
    x_t = np.transpose(x)
    gram_matrix = np.matmul(x_t, x)

    try:

        # A matrix has no inverse if the determinant is 0
        assert np.linalg.det(gram_matrix) != 0

        # Find the inverse of the Gram matrix
        gram_matrix_inv = np.linalg.inv(gram_matrix)

        # Use the normal equations to get the weights
        w = np.matmul(gram_matrix_inv, np.matmul(x_t, y))

        # Compute the loss
        loss = compute_loss(y, x, w)

        return w, loss

    except AssertionError:

        print("The Gram matrix is singular and as such the solution cannot be found!")
        return None


def ridge_regression(y, x, lambda_):
    """
    Calculate the least squares solution with L2 regularization explicitly using the normal equations

    :param x: data matrix, numpy ndarray with shape with shape (N, D),
              where N is the number of samples and D is the number of features
    :param y: vector of target values, numpy array with dimensions (N, 1)
    :param lambda_: regularization coefficient, positive float value

    :returns: (weights, loss value), tuple
    """

    # Compute the Gram matrix
    x_t = np.transpose(x)
    gram_matrix = np.matmul(x_t, x)

    # Update the Gram matrix using lambda_
    d = gram_matrix.shape[0]
    ridge_matrix = gram_matrix + lambda_ * np.identity(d)

    # Calculate the inverse of the updated Gram matrix
    ridge_matrix_inv = np.linalg.inv(ridge_matrix)

    # Calculate the weights using the normal equations
    w = np.matmul(ridge_matrix_inv, np.matmul(x_t, y))

    # Compute the loss
    loss = compute_loss(y, x, w)

    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    ...


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    ...
