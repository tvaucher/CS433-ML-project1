""" Main script - training the best model on the train set using the best hyperparameters and using the test set to make predictions for the submission """

import numpy as np

from preprocessing import load_csv_data, preprocessing_pipeline, split_data_by_categorical_column

from helpers import predict_labels, create_csv_submission, compute_accuracy

from implementations import reg_logistic_regression

PRI_JET_NUM_INDEX = 22
SEED = 2019
ALGORITHMS = ("ridge", "logistic", "svm")

if __name__ == "__main__":

    # Load the train and test datasets using the provided helper function load_csv_data
    train_classes, train_data, train_ids = load_csv_data("data/train.csv")
    test_classes, test_data, test_ids = load_csv_data("data/test.csv")

    # Split the train and test data into 4 subsets based on the value of the PRI_JET_NUM categorical feature
    # In the dataset, whether some of the other features are defined (are not NaN) depends on the value of this feature
    train_classes_jet_num_splits, train_data_jet_num_splits, train_ids_jet_num_splits = \
        split_data_by_categorical_column(train_classes,
                                         train_data,
                                         train_ids,
                                         PRI_JET_NUM_INDEX)
    test_classes_jet_num_splits, test_data_jet_num_splits, test_ids_jet_num_splits = \
        split_data_by_categorical_column(test_classes,
                                         test_data,
                                         test_ids,
                                         PRI_JET_NUM_INDEX)

    # We achieved our best results using Regularized Logistic Regression,
    # so we only load only those previously computed optimal params to generate the submission
    logistic_best_params = np.load("results/logistic_best_params.npy", allow_pickle=True)
    logistic_best_models = []

    for (lambda_, deg, gamma), train_classes_split, train_data_split in \
            zip(logistic_best_params, train_classes_jet_num_splits, train_data_jet_num_splits):
        data_split, columns_to_remove, mean, std = preprocessing_pipeline(train_data_split, degree=np.int(deg),
                                                                          cross_term=True, norm_first=False)
        initial_w = np.zeros((data_split.shape[1],))
        w, loss = reg_logistic_regression(train_classes_split, data_split, lambda_, initial_w, 500, gamma, 1)
        print(f'Loss: {loss:.3f} Accuracy : {compute_accuracy(predict_labels(w, data_split), train_classes_split)}')
        logistic_best_models.append((w, loss, columns_to_remove, mean, std))

    # Calculate the predictions for each of the 4 subsets using the weights and then combine them
    results = None
    for (w, _, col_to_rm, mean, std), (_, deg, _), test_classes_split, test_data_split, test_ids_split in \
            zip(logistic_best_models, logistic_best_params,
                test_classes_jet_num_splits, test_data_jet_num_splits, test_ids_jet_num_splits):
        test_data_split, _, _, _ = preprocessing_pipeline(test_data_split, degree=np.int(deg),
                                                          columns_to_remove=col_to_rm,
                                                          cross_term=True, norm_first=False, mean=mean, std=std)
        pred = predict_labels(w, test_data_split)
        out = np.stack((test_ids_split, pred), axis=-1)
        results = out if results is None else np.vstack((results, out))

    # Create the submission
    create_csv_submission(results[:, 0], results[:, 1], 'results/logistic_submission.csv')
