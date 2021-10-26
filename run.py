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

    """ Acquisition of train and test data """
    DATA_TRAIN_PATH = '../data/train.csv'
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    DATA_TEST_PATH = '../data/test.csv'
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    N = len(y) # training set cardinality
    D = tX.shape[1] # number of parameters ("dimensionality")



    """ Parameters assignment """
    alpha = 0.1
    maxiter = 5000
    gamma = 0.00001
    degree = 3



    """ Correction of the training data """
    y [y < 0] = 0
    tX, D_del, cols_deleted, cols_kept = missing_values_elimination(tX)
    tX = standardize_tX(tX)
    tX = eliminate_outliers(tX, alpha)
    tX = phi(tX, degree)



    """ Logistic regression on the training set """
    weights = np.zeros(D * degree + 1)
    initial_w = least_squares(y,tX)[0]
    loss, weights_hat = logistic_regression(y, tX, initial_w, maxiter, gamma)
    new_cols_kept = [0] ### where to put the intercept???
    for i in range(0,degree):
        cols_new = np.array(cols_kept) + (i * D + 1)
        cols_new = cols_new.tolist()
        new_cols_kept = new_cols_kept + cols_new
    weights[new_cols_kept] = weights_hat
    
    # """ Logistic regression with SGD on the training set """
    # weights = np.zeros(D)
    # initial_w = least_squares(y,tX)[0]
    # loss, weights_hat = logistic_regression_SGD(y, tX, initial_w, maxiter, gamma)
    # weights[cols_kept] = weights_hat

    # """ Regularized logistic regression """
    # initial_w = least_squares(y,tX)[0]
    # lambda_ = choose_lambda_logistic(y, tX, initial_w, maxiter, gamma)
    # maxiter = 5000
    # weights = np.zeros(D)
    # initial_w = least_squares(y,tX)[0]
    # loss, weights_hat = reg_logistic_regression(y, tX, lambda_, initial_w, maxiter, gamma)
    # weights[cols_kept] = weights_hat

    """ Correction of the test data """
    tX_test = missing_values_correction(tX_test)
    tX_test = standardize_tX(tX_test)
    tX_test = phi(tX_test, degree)



    """ Creation of the submission file """
    OUTPUT_PATH = '../data/submission.csv'
    y_pred = predict_labels(weights, tX_test)
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
