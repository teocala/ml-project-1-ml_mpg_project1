"""
Main script to work and get results from the dataset
"""

# Libraries importation

import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import *
from utilities import *


""" MAIN """

if __name__ == '__main__':

    # Data loading

    # TODO: In the same directory of your repository folder, create a folder "data" where you place the data downloaded from
    #  1) https://github.com/epfml/ML_course/tree/master/projects/project1/data
    #  or
    #  2) https://www.aicrowd.com/challenges/epfl-machine-learning-higgs

    DATA_TRAIN_PATH = '../data/train.csv'
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    N = len(y) # training set cardinality
    D = tX.shape[1] # number of parameters ("dimensionality")


    DATA_TEST_PATH = '../data/test.csv'
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)


    weights = np.zeros(D)
    tX, D, cols_deleted = missing_values(tX)
    cols_kept = list(set(range(D)).difference(set(cols_deleted)))
    tX = standardize_tX(tX)
    alpha = 0.1
    tX = eliminate_outliers(tX, alpha)
    initial_w = np.zeros(D)
    maxiter = 1000
    gamma = 0.1
    print(tX.shape)
    print(y.shape)
    loss, w_hat = logistic_regression(y, tX, initial_w, maxiter, gamma)
    weights[cols_kept] = w_hat

    
    # Creation of the submission file
    OUTPUT_PATH = '../data/submission.csv'
    y_pred = predict_labels(weights, tX_test)
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
