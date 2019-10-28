""" Script where the training set is used to find the best hyperparameters using cross-validation """

import numpy as np

from preprocessing import load_csv_data, preprocessing_pipeline, split_data_by_categorical_column

from cross_validation import k_fold_indices, k_fold_cross_split_data

from helpers import predict_labels, compute_accuracy

from implementations import reg_logistic_regression, svm, ridge_regression


def find_best_hyperparameters(error_values, lambdas, degrees):
    """ A function which finds the hyperparameters that give the highest accuracy

    Args:
        error_values (np.array) Cross-validation error values of all different
        combinations of hyperparameters.
        lambdas (np.array): Array of different regularization coefficients to try.
        degrees (np.array): Array of different degrees coefficients to try for
        feature expansion.

    Returns:
        (float), (int): Lambda and degree combination that result in lowest error
    """
    best_deg, best_lmbda = np.unravel_index(np.argmax(error_values), error_values.shape)
    # Extract the lambda and degree resulting in the highest accuracy.
    degree_best = degrees[best_deg]
    lambda_best = lambdas[best_lmbda]
    return lambda_best, degree_best


PRI_JET_NUM_INDEX = 22
SEED = 2019
POSSIBLE_LAMBDA_VALUES = [1e-6, 5e-5, 2.5e-5, 1e-5, 7.5e-4, 5e-4, 2.5e-4, 1e-4, 0]
POSSIBLE_LAMBDA_LOG = [0, ]
POSSIBLE_LAMBDA_SVM = [1e-2, ]
POSSIBLE_DEGREES = np.arange(5, 14)
ALGORITHMS = ("ridge", "logistic", "svm")
grid_shape = (4, len(ALGORITHMS), len(POSSIBLE_DEGREES), len(POSSIBLE_LAMBDA_VALUES), 2)

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

    # Initialize a grid for cross-validation of each algorithm,
    # to record mean and std of train and validation error for each parameter combination
    train_accuracy_matrix = np.zeros(grid_shape)
    validation_accuracy_matrix = np.zeros(grid_shape)

    # Perform cross-validation on each of the 4 subsets,
    # to find the optimal values for the lambda and polynomial degree parameters
    # The train data is first preprocessed then split into 5 cross-folds
    for jet_num, (train_classes_split, train_data_split) in enumerate(zip(train_classes_jet_num_splits, train_data_jet_num_splits)):
        k_indices = k_fold_indices(train_data_split.shape[0], 5, SEED)
        for i, deg in enumerate(POSSIBLE_DEGREES):
            train_data, _ = preprocessing_pipeline(train_data_split, degree=deg)
            train_set_folds = k_fold_cross_split_data(train_classes_split, train_data, k_indices)

            for j, lambda_ in enumerate(POSSIBLE_LAMBDA_VALUES):
                folds_train_accuracy = []
                folds_validation_accuracy = []

                # Train a Ridge Regression model on each fold
                for x_train, y_train, x_test, y_test in train_set_folds:
                    w, train_loss = ridge_regression(y_train, x_train, lambda_)
                    folds_train_accuracy.append(compute_accuracy(predict_labels(w, x_train), y_train))
                    folds_validation_accuracy.append(compute_accuracy(predict_labels(w, x_test), y_test))
                train_accuracy_matrix[jet_num, 0, i, j] = \
                    (np.mean(folds_train_accuracy), np.std(folds_train_accuracy))
                validation_accuracy_matrix[jet_num, 0, i, j] = \
                    (np.mean(folds_validation_accuracy), np.std(folds_validation_accuracy))

            train_data_log_svm = preprocessing_pipeline(train_data_split, degree=deg, norm_first=False)
            train_set_folds = k_fold_cross_split_data(train_classes_split, train_data_log_svm, k_indices)

            for j, lambda_ in enumerate(POSSIBLE_LAMBDA_LOG):
                folds_train_accuracy = []
                folds_validation_accuracy = []

                # Train a Regularized Ridge Regression model on each fold
                for x_train, y_train, x_test, y_test in train_set_folds:
                    initial_w = np.zeros((x_train.shape[1],))
                    try:
                        w, train_loss = reg_logistic_regression(y_train, x_train, lambda_, initial_w, 350, 3e-1, 1)
                        folds_train_accuracy.append(compute_accuracy(predict_labels(w, x_train), y_train))
                        folds_validation_accuracy.append(compute_accuracy(predict_labels(w, x_test), y_test))
                    except Exception:
                        pass
                train_accuracy_matrix[jet_num, 1, i, j] = \
                    (np.mean(folds_train_accuracy), np.std(folds_train_accuracy))
                validation_accuracy_matrix[jet_num, 1, i, j] = \
                    (np.mean(folds_validation_accuracy), np.std(folds_validation_accuracy))

            for j, lambda_ in enumerate(POSSIBLE_LAMBDA_SVM):
                folds_train_accuracy = []
                folds_validation_accuracy = []

                # Train a SVM model on each fold
                for x_train, y_train, x_test, y_test in train_set_folds:
                    initial_w = np.zeros((x_train.shape[1],))
                    try:
                        w, train_loss = svm(y_train, x_train, lambda_, initial_w, 500, 1e-5, 1e-1)
                        folds_train_accuracy.append(compute_accuracy(predict_labels(w, x_train), y_train))
                        folds_validation_accuracy.append(compute_accuracy(predict_labels(w, x_test), y_test))
                    except Exception:
                        pass
                train_accuracy_matrix[jet_num, 2, i, j] = \
                    (np.mean(folds_train_accuracy), np.std(folds_train_accuracy))
                validation_accuracy_matrix[jet_num, 2, i, j] = \
                    (np.mean(folds_validation_accuracy), np.std(folds_validation_accuracy))

    # Find the best parameter combination for each algorithm for each of the 4 subsets and store them to local files
    # The best combination is the one with minimum mean validation loss
    ridge_best_params = \
        [find_best_hyperparameters(validation_accuracy_matrix[jet_num, 0, :, :, 0], POSSIBLE_LAMBDA_VALUES, POSSIBLE_DEGREES)
         for jet_num in range(4)]
    np.save("results/ridge_best_params", ridge_best_params)
    logistic_best_params = \
        [(*find_best_hyperparameters(validation_accuracy_matrix[jet_num, 1, :, :, 0], POSSIBLE_LAMBDA_LOG, POSSIBLE_DEGREES), 3e-1)
         for jet_num in range(4)]
    np.save("results/logistic_best_params", logistic_best_params)
    svm_best_params = \
        [find_best_hyperparameters(validation_accuracy_matrix[jet_num, 2, :, :, 0], POSSIBLE_LAMBDA_SVM, POSSIBLE_DEGREES)
         for jet_num in range(4)]
    np.save("results/svm_best_params", svm_best_params)
