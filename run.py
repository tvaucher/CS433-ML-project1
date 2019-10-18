# Module containing the code to reproduce the results on aicrowd

from helpers import load_csv_data

if __name__ == "__main__":

    # Load the train and test datasets using the provided helper function load_csv_data
    train_classes, train_data, train_ids = load_csv_data("data/train.csv")
    test_classes, test_data, test_ids = load_csv_data("data/test.csv")
