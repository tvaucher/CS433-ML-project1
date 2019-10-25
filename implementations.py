"""Module containing all implementations of ML techniques required for the project"""

import numpy as np

from helpers import (batch_iter, compute_gradient_hinge,
                     compute_gradient_mse, compute_gradient_nll,
                     compute_hessian_nll, compute_loss, compute_loss_hinge,
                     compute_loss_nll, compute_subgradient_mae,
                     map_target_classes_to_boolean)


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

    :raises LinAlgError: if the Gram matrix has no inverse

    :returns: (weights, loss value), tuple
    """

    # Compute the Gram matrix
    gram = x.T @ x

    # Use the normal equations to find the best weights
    w = np.linalg.solve(gram, x.T @ y)

    # Compute the loss
    loss = compute_loss(y, x, w)

    return w, loss


def ridge_regression(y, x, lambda_):
    """
    Calculate the least squares solution with L2 regularization explicitly using the normal equations

    :param x: data matrix, numpy ndarray with shape with shape (N, D),
              where N is the number of samples and D is the number of features
    :param y: vector of target values, numpy array with dimensions (N, 1)
    :param lambda_: regularization coefficient, positive float value

    :returns: (weights, loss value), tuple
    """

    # Compute the Gram matrix and update it with the regularization term
    gram = x.T @ x
    gram += 2 * x.shape[0] * lambda_ * np.identity(gram.shape[0])

    # Use the normal equations to find the best weights
    w = np.linalg.solve(gram, x.T @ y)

    # Compute the loss
    loss = compute_loss(y, x, w)

    return w, loss


def logistic_regression(y, x, initial_w, max_iters, gamma):
    """
    Implementation of the Newton optimization algorithm for logistic regression

    :param x: data matrix, numpy ndarray with shape with shape (N, D),
              where N is the number of samples and D is the number of features
    :param y: vector of target values, numpy array with dimensions (N, 1)
    :param initial_w: vector of initial weights, numpy array with dimensions (D, 1)
    :param max_iters: how many iterations to run the algorithm, integer
    :param gamma: learning rate, positive float value

    :returns: (final weights, final loss value), tuple
    """

    # Map the {-1, 1} classes to {0, 1}
    y = map_target_classes_to_boolean(y)

    # Set the initial values for the weights
    w = initial_w

    for n_iter in range(max_iters):
        # Compute the gradient and Hessian of the loss function
        grd = compute_gradient_nll(y, x, w)
        # hess = compute_hessian_nll(y, x, w)

        # Update the weights using the gradient, Hessian and learning rate
        w = w - gamma * grad  # np.matmul(np.linalg.inv(hess), grd)

    # Compute the final loss value
    loss = compute_loss_nll(y, x, w)

    return w, loss


def reg_logistic_regression(y, x, lambda_, initial_w, max_iters, gamma):
    """
    Implementation of the Newton optimization algorithm for logistic regression with L2 regularization

    :param x: data matrix, numpy ndarray with shape with shape (N, D),
              where N is the number of samples and D is the number of features
    :param y: vector of target values, numpy array with dimensions (N, 1)
    :param lambda_: regularization coefficient, positive float value
    :param initial_w: vector of initial weights, numpy array with dimensions (D, 1)
    :param max_iters: how many iterations to run the algorithm, integer
    :param gamma: learning rate, positive float value

    :returns: (final weights, final loss value), tuple
    """
    # Map the {-1, 1} classes to {0, 1}
    y = map_target_classes_to_boolean(y)

    # Set the initial values for the weights
    w = initial_w

    for n_iter in range(max_iters):
        # Compute the gradient and Hessian of the loss function
        grd = compute_gradient_nll(y, x, w, lambda_)
        # hess = compute_hessian_nll(y, x, w, lambda_)

        # Update the weights using the gradient, Hessian and learning rate
        w -= gamma * grd

    # Compute the final loss value
    loss = compute_loss_nll(y, x, w, lambda_)

    return w, loss


def svm(y, x, lambda_, initial_w, max_iters, gamma):
    """
    Implementation of the linear SVM classification algorithm with L2 regularization
    The SVM is simulated through optimization of the Hinge loss function with gradient descent

    :param x: data matrix, numpy ndarray with shape with shape (N, D),
              where N is the number of samples and D is the number of features
    :param y: vector of target values, numpy array with length N
    :param lambda_: regularization coefficient, positive float value
    :param initial_w: vector of initial weights, numpy array with length D
    :param max_iters: how many iterations to run the algorithm, integer
    :param gamma: learning rate, positive float value

    :returns: (final weights, final loss value), tuple
    """

    # Set the initial values for the weights
    w = initial_w

    # Compute the initial loss value
    prev_loss = compute_loss_hinge(y, x, w, lambda_)

    for n_iter in range(max_iters):
        # Compute the gradient and Hessian of the loss function
        grd = compute_gradient_hinge(y, x, w, lambda_)

        # Update the weights using the gradient, Hessian and learning rate
        w -= gamma * grd

        # Compute the current loss and test convergence
        loss = compute_loss_hinge(y, x, w, lambda_)
        if abs(loss - prev_loss) < 1e-5:
            break
        prev_loss = loss

    # Compute the final loss value
    loss = compute_loss_hinge(y, x, w, lambda_)

    return w, loss
