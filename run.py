import numpy as np
import matplotlib.pyplot as plt

from preprocessing import load_csv_data, remove_na_columns, remove_low_variance_features, \
    remove_correlated_features, z_normalize_data, augment_features_polynomial_basis, split_data_by_categorical_column

from helpers import compute_loss_nll, predict_labels, compute_accuracy, compute_loss, create_csv_submission

from implementations import reg_logistic_regression, svm, ridge_regression


def preprocessing(data, *, nan_value=-999., low_var_threshold=0.1, corr_threshold=0.85, degree=3):
    data = data.copy()
    data[data == nan_value] = np.nan
    data = remove_na_columns(data)
    data = remove_low_variance_features(data, low_var_threshold)
    data = remove_correlated_features(data, corr_threshold)
    data = z_normalize_data(data)
    data[np.isnan(data)] = 0.
    data = augment_features_polynomial_basis(data, degree)
    return data


PRI_JET_NUM_INDEX = 22
SEED = 2019

if __name__ == "__main__":
    # Load the test dataset using the provided helper function load_csv_data
    test_classes, test_data, test_ids = load_csv_data("data/test.csv")

    # Split the test data into 4 subsets based on the value of the PRI_JET_NUM categorical feature
    # In the dataset, whether some of the other features are defined (are not NaN) depends on the value of this feature
    test_classes_jet_num_splits, test_data_jet_num_splits, test_ids_jet_num_splits = \
        split_data_by_categorical_column(test_classes,
                                         test_data,
                                         test_ids,
                                         PRI_JET_NUM_INDEX)

    # Load previously computed optimal hyperparameters and weights for ridge regression
    ridge_best_params = np.load("ridge_best_params.npy", allow_pickle=True)
    ridge_models = np.load("ridge_models.npy", allow_pickle=True)

    # Calculate the predictions for each of the 4 subsets using the ridge regression weights and then combine them
    ridge_results = None
    for (w, _), (_, deg), test_classes_split, test_data_split, test_ids_split in zip(ridge_models, ridge_best_params,
                                                                                     test_classes_jet_num_splits,
                                                                                     test_data_jet_num_splits,
                                                                                     test_ids_jet_num_splits):
        test_data_split = preprocessing(test_data_split, degree=deg)
        pred = predict_labels(w, test_data_split)
        out = np.stack((test_ids_split, pred), axis=-1)
        ridge_results = out if ridge_results is None else np.vstack((ridge_results, out))

    # Create the ridge regression submission
    create_csv_submission(ridge_results[:, 0], ridge_results[:, 1], 'ridge_submission.csv')
