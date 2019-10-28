# CS-433 Project 1
## Authors (team: Indigo-Vanguard)
- Louis Amaudruz
- Andrej Janchevski
- Timoté Vaucher

## File structure
- cross_validation.py
    - Module containing functions for splitting data for cross validation
- helpers.py
    - Module that contains various helper functions for the project
- implementations.py
    - Module containing all implementations of ML techniques required for the project
- preprocessing.py
    - Module containing functions to load and preprocess the data
- training.ipynb
    - Script where the training set is used to find the best hyperparameters using k-fold cross-validation
- run.py
    - Main script - training the best model on the train set using the best hyperparameters and using the test set to make predictions for the submission
- results/
    - Folder containing the results from the cross-validation, model training and submission
- plot.ipynb + plots/ 
    - Notebook with the plotting functions + folder containing the resulting plots

## How to reproduce our results
We assume that the repository is already downloaded and extracted, that the [data](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs-2019/dataset_files) is downloaded and extracted in a data/ folder at the root of the program. We further assume that Anaconda is already installed.

### Create the environment
```shell
conda create -n cs433proj1 python=3.7 numpy=1.16 matplotlib
conda activate cs433proj1
```

### Run the code
From the root folder of the project

```shell
python run.py
```

