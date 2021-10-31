# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 09:32:35 2021

@author: giuli
"""

import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from proj1_helpers import *
from parameter_selection_utilities import *

DATA_TRAIN_PATH = 'data/train.csv'
y_train, tX_train, ids = load_csv_data(DATA_TRAIN_PATH)

N = len(y_train) # training set cardinality
D = tX_train.shape[1] # number of parameters ("dimensionality")
    
degrees = [1,2,3,4,5,6,7]
lambdas = np.logspace(-5,0,10)
alpha = 0.1

seed = 1

""" Optimal parameters for Ridge Regression """

jet_0 = get_subset_PRI_jet_num(tX_train, 0)
jet_1 = get_subset_PRI_jet_num(tX_train, 1)
jet_2 = get_subset_PRI_jet_num(tX_train, 2)
jet_3 = get_subset_PRI_jet_num(tX_train, 3)

jets = [jet_0,jet_1,jet_2,jet_3]

J = len(jets)
degrees_ridge = np.zeros(J)
lambdas_ridge = np.zeros(J)
accs_ridge = np.zeros(J)
k_fold = 3


# We will not need the 22-th feature in the dataset anymore
tX_train = np.delete(tX_train, 22, axis = 1)

for i in range(J):
    j = jets[i]
    tX = tX_train[j]
    y = y_train[j]


    tX = preprocessing(tX)
    tX = eliminate_outliers(tX, alpha)

    degrees_ridge[i],lambdas_ridge[i],accs_ridge[i] = choose_parameters_ridge_regression(degrees, lambdas, k_fold, y, tX, seed)

""" Optimal parameters for Lasso Regression """



""" Optimal parameters for Logistic Regression """
