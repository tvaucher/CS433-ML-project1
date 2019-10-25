import numpy as np
import matplotlib.pyplot as plt

from preprocessing import load_csv_data, preprocessing_pipeline, split_data_by_categorical_column

from helpers import predict_labels, create_csv_submission


PRI_JET_NUM_INDEX = 22
SEED = 2019
ALGORITHMS = ("ridge", "logistic", "svm")

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
    for alg in ALGORITHMS:
        # Load previously computed optimal hyperparameters and weights for each algorithm
        alg_best_params = np.load(alg + "_best_params.npy", allow_pickle=True)
        alg_models = np.load(alg + "_models.npy", allow_pickle=True)

        # Calculate the predictions for each of the 4 subsets using the weights and then combine them
        alg_results = None
        for (w, _), (_, deg), test_classes_split, test_data_split, test_ids_split in zip(alg_models, alg_best_params,
                                                                                         test_classes_jet_num_splits,
                                                                                         test_data_jet_num_splits,
                                                                                         test_ids_jet_num_splits):
            test_data_split = preprocessing_pipeline(test_data_split, degree=deg.astype(int))
            if alg == "logistic":
                # For logistic regression the predictions will be in the interval [0, 1] instead of [-1, 1],
                # so 0.5 is used as the class border value instead of 0
                pred = np.dot(test_data_split, w)
                pred[np.where(pred <= 0.5)] = -1
                pred[np.where(pred > 0.5)] = 1
            else:
                pred = predict_labels(w, test_data_split)
            out = np.stack((test_ids_split, pred), axis=-1)
            alg_results = out if alg_results is None else np.vstack((alg_results, out))

        # Create the submission for each algorithm
        create_csv_submission(alg_results[:, 0], alg_results[:, 1], alg + '_submission.csv')
