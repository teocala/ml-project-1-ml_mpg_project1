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
    DATA_TRAIN_PATH = 'data/train.csv'
    y_train, tX_train, ids = load_csv_data(DATA_TRAIN_PATH)
    DATA_TEST_PATH = 'data/test.csv'
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    N = len(y_train) # training set cardinality
    D = tX_train.shape[1] # number of parameters ("dimensionality")

    # Inizialization of a vector which stores the final prediction
    y_pred = np.zeros(tX_test.shape[0])

    """ Parameters assignment """
    alpha = 0.1
    opt_degrees = [2,2,2,1]


    """ Splitting of training and test data according to the categorical feature PRI_jet_num """


    jet_0 = get_subset_PRI_jet_num(tX_train, 0)
    jet_1 = get_subset_PRI_jet_num(tX_train, 1)
    jet_2 = get_subset_PRI_jet_num(tX_train, 2)
    jet_3 = get_subset_PRI_jet_num(tX_train, 3)

    jets = [jet_0,jet_1,jet_2,jet_3]

    jet_0_te = get_subset_PRI_jet_num(tX_test, 0)
    jet_1_te = get_subset_PRI_jet_num(tX_test, 1)
    jet_2_te = get_subset_PRI_jet_num(tX_test, 2)
    jet_3_te = get_subset_PRI_jet_num(tX_test, 3)

    jets_te = [jet_0_te,jet_1_te,jet_2_te,jet_3_te]

    # We will not need the 22-th feature in the dataset anymore
    tX_train = np.delete(tX_train, 22, axis = 1)
    tX_test = np.delete(tX_test, 22, axis = 1)


    for num_jet in range(4):
        j = jets[num_jet]
        j_test = jets_te[num_jet]
        tX = tX_train[j]
        y = y_train[j]
        tX_jt = tX_test[j_test]
        deg = opt_degrees[num_jet]


        """ Correction of the training data """
        y [y < 0] = 0
        tX = preprocessing(tX) # Preprocessing transformations described in the report (see utilities.py for details)
        tX = eliminate_outliers(tX, alpha) # Elimination of (1-alpha) outliers
        tX = build_poly_with_roots(tX, deg) # Polynomial augmentation: addition of polynomials until degree deg, square and cubic roots and pairwise products


        """ Choice of the logistic regression regularization parameter"""
        # To be used for a cross-validation over lambda parameters (warning: it requires a long time)
        #initial_w = np.zeros(tX.shape[1])
        #lambda_ = choose_lambda_logistic(y, tX, initial_w, maxiter=100, gamma=0.1)


        """ Regularized logistic regression """
        initial_w = np.zeros(tX.shape[1])
        loss, weights = fista(y, tX, initial_w, max_iters = 500, gamma = 0.1, lambda_ = 0.001)
        # Otherwise, l1_logistic_regression in implementations.py could be used for the L1/lasso regularization
        # (i.e. the standard subgradient method) but this is slower.


        """ Correction of the test data """
        tX_jt = preprocessing(tX_jt)
        tX_jt = build_poly_with_roots(tX_jt, deg)

        """ Prection for the current jet_num """
        y_pred[j_test] = predict_labels(weights, tX_jt)
        # Otherwise k_nearest in utilities.py could be used, warning: it requires a long time



    """ Accuracy of the result """

    DATA_SOL_PATH = 'data/true_solutions.csv'

    y = np.genfromtxt(DATA_SOL_PATH, delimiter=",", skip_header=1, dtype=str, usecols=-3)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1
    y_true = yb[-len(y_pred):]

    accuracy = np.count_nonzero(y_true == y_pred)/len(y_pred)
    print("Accuracy =", accuracy)

    """ Creation of the submission file """
    OUTPUT_PATH = 'data/submission.csv'
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
