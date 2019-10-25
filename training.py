import numpy as np
import matplotlib.pyplot as plt

from preprocessing import load_csv_data, preprocessing_pipeline, split_data_by_categorical_column

from cross_validation import k_fold_cross_split_data

from helpers import compute_loss, compute_loss_nll, compute_loss_hinge

from implementations import reg_logistic_regression, svm, ridge_regression


def find_best_hyperparameters(error_values, lambdas, degrees):
    """ A function which finds the hyperparameters that give the lowest error

    Args:
        error_values (np.array) Cross-validation error values of all different
        combinations of hyperparameters.
        lambdas (np.array): Array of different regularization coefficients to try.
        degrees (np.array): Array of different degrees coefficients to try for
        feature expansion.

    Returns:
        (float), (int): Lambda and degree combination that result in lowest error
    """
    best_lmbda, best_deg = np.unravel_index(np.argmin(error_values), error_values.shape)
    # Extract the lambda and degree resulting in the lowest error.
    degree_best = degrees[best_deg]
    lambda_best = lambdas[best_lmbda]
    return lambda_best, degree_best


PRI_JET_NUM_INDEX = 22
SEED = 2019
POSSIBLE_LAMBDA_VALUES = np.logspace(-5, -2, 5)
POSSIBLE_DEGREES = np.arange(1, 8)
ALGORITHMS = ("RIDGE", "LOGISTIC", "SVM")

if __name__ == "__main__":

    # Load the train dataset using the provided helper function load_csv_data
    train_classes, train_data, train_ids = load_csv_data("data/train.csv")

    # Split the data into 4 subsets based on the value of the PRI_JET_NUM categorical feature
    # In the dataset, whether some of the other features are defined (are not NaN) depends on the value of this feature
    train_classes_jet_num_splits, train_data_jet_num_splits, train_ids_jet_num_splits = \
        split_data_by_categorical_column(train_classes,
                                         train_data,
                                         train_ids,
                                         PRI_JET_NUM_INDEX)

    # Initialize a grid for cross-validation of ridge regression,
    # to record mean and std of train and validation error for each parameter combination
    grid_shape = (4, len(POSSIBLE_DEGREES), len(POSSIBLE_LAMBDA_VALUES), 2)
    train_loss_matrix = np.zeros(grid_shape)
    validation_loss_matrix = np.zeros(grid_shape)

    # Perform cross-validation on each of the 4 subsets,
    # to find the optimal values for the lambda and polynomial degree parameters
    # The train data is first preprocessed then split into 10 cross-folds
    for jet_num, (train_classes_split, train_data_split) in \
            enumerate(zip(train_classes_jet_num_splits, train_data_jet_num_splits)):
        for i, deg in enumerate(POSSIBLE_DEGREES):
            train_data_split = preprocessing(train_data_split, degree=deg)
            train_set_folds = k_fold_cross_split_data(train_classes_split, train_data_split, 10, SEED)
            for j, lambda_ in enumerate(POSSIBLE_LAMBDA_VALUES):
                print("PRI_JET_NUM: {0}; Degree: {1}; Lambda: {2:.5f}".format(jet_num, deg, lambda_))
                folds_train_losses = []
                folds_validation_losses = []
                for x_train, y_train, x_test, y_test in train_set_folds:
                    w, train_loss = ridge_regression(y_train, x_train, lambda_)
                    folds_train_losses.append(train_loss)
                    validation_loss = compute_loss(y_test, x_test, w)
                    folds_validation_losses.append(validation_loss)
                train_loss_matrix[jet_num, i, j] = \
                    (np.mean(folds_train_losses), np.std(folds_train_losses))
                validation_loss_matrix[jet_num, i, j] = \
                    (np.mean(folds_validation_losses), np.std(folds_validation_losses))

    # Find the best parameter combination for ridge regression for each of the 4 subsets
    # The best combination is the one with minimum mean validation loss
    ridge_best_params = \
        [find_best_hyperparameters(validation_loss_matrix[jet_num, :, :, 0], POSSIBLE_LAMBDA_VALUES, POSSIBLE_DEGREES)
         for jet_num in range(4)]
    np.save("ridge_best_params", ridge_best_params)

    # Train the weights for the 4 subsets using the best parameters, end result is 4 different models
    ridge_models = []
    for (lambda_, deg), train_classes_split, train_data_split in zip(ridge_best_params, train_classes_jet_num_splits,
                                                                     train_data_jet_num_splits):
        train_data_split = preprocessing(train_data_split, degree=deg)
        initial_w = np.zeros((train_data_split.shape[1],))
        w, loss = ridge_regression(train_classes_split, train_data_split, lambda_)

        print(
            f'Loss: {loss:.3f}; Accuracy : {compute_accuracy(predict_labels(w, train_data_split), train_classes_split)}')
        # w, loss = reg_logistic_regression(train_classes_split, data_split, 1e-3, initial_w, 1001, 1e-5)
        # w, loss = svm(train_classes_split, data_split, 1e-3, initial_w, 1001, 1e-5)#5e-8)
        ridge_models.append((w, loss))

    # Save the trained models for ridge regression in a local binary file
    np.save("ridge_models", ridge_models)

    print(np.load("ridge_models.npy"))
