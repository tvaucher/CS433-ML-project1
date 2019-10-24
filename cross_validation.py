import numpy as np

def k_fold_cross_split_data(y, x, k, seed):
    """
    Helper function that splits data samples randomly into k folds, to be used afterwards for cross-validation

    :param y: vector of target values, numpy array with dimensions (N, 1)
    :param x: data matrix, numpy ndarray with shape with shape (N, D),
              where N is the number of samples and D is the number of features
    :param k: number of folds, positive integer
    :param seed: seeding value for the random generator, integer

    :returns: splits, list of tuples of the form (x_train, y_train, x_test, y_test),
              where each tuple contains samples and targets in train set, followed by samples and targets in test set
    """

    # Set the seed
    np.random.seed(seed)

    # Calculate the number of samples in each fold as an integer by rounding if necessary
    n = x.shape[0]
    fold_size = int(n / k)

    # Shuffle and split the sample indices into k folds
    indices = np.random.permutation(n)
    k_indices = [indices[i * fold_size: (i + 1) * fold_size] for i in range(k)]

    # Generate the k different train-test splits of the data using the indices
    # in each of the k folds as test indices and the rest as train indices
    splits = []
    for i, k_index in enumerate(k_indices):
        y_test, x_test = y[k_index], x[k_index]
        k_indices_without_test = k_indices[:i] + k_indices[i + 1:]
        train_indices = [index for k_index in k_indices_without_test for index in k_index]
        y_train, x_train = y[train_indices], x[train_indices]
        splits.append((x_train, y_train, x_test, y_test))
    return splits
