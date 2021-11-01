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

# Loading train data

DATA_TRAIN_PATH = 'data/train.csv'
y_train, tX_train, ids = load_csv_data(DATA_TRAIN_PATH)

N = len(y_train) # training set cardinality
D = tX_train.shape[1] # number of parameters ("dimensionality")

# Candidates for degree and lambdas 
degrees = [1,2,3,4,5,6]
lambdas = np.logspace(-6,-3,10)
alpha = 0.1

# Division according to PRI_jet_num
jet_0 = get_subset_PRI_jet_num(tX_train, 0)
jet_1 = get_subset_PRI_jet_num(tX_train, 1)
jet_2 = get_subset_PRI_jet_num(tX_train, 2)
jet_3 = get_subset_PRI_jet_num(tX_train, 3)

jets = [jet_0,jet_1,jet_2,jet_3]

J = len(jets)

degrees_ridge = np.zeros(J)
lambdas_ridge = np.zeros(J)
accs_ridge = np.zeros(J)

degrees_l1 = np.zeros(J)
lambdas_l1 = np.zeros(J)
accs_l1 = np.zeros(J)

# We will not need the 22-th feature in the dataset anymore
tX_train = np.delete(tX_train, 22, axis = 1)

"""
Here you can find code to identify the best parameters with cross validation 
for Ridge Regression and L1-Regularized Logistic Regression. Since the computational time
is pretty high, we suggest to comment the part of the code which is not needed.
"""


""" Optimal parameters for Ridge Regression """
# for i in range(J):
#     j = jets[i]
#     tX = tX_train[j]
#     y = y_train[j]


#     tX = preprocessing(tX)
#     tX = eliminate_outliers(tX, alpha)

#     degrees_ridge[i],lambdas_ridge[i],accs_ridge[i] = choose_parameters_ridge_regression(degrees, lambdas, k_fold, y, tX, seed)

""" Optimal parameters for Lasso Regression """
# for i in range(J):
#     j = jets[i]
#     tX = tX_train[j]
#     y = y_train[j]


#     tX = preprocessing(tX)
#     tX = eliminate_outliers(tX, alpha)

#     degrees_l1[i],lambdas_l1[i],accs_l1[i] = choose_parameters_l1_regression(degrees, lambdas, k_fold, y, tX, seed)

"""
Having found the optimal degrees, we report here the code to generate the plot included in the report,
with the trend of the test accuracy wrt to different values of lambda
"""
accuracies = np.zeros(len(lambdas))
degrees = [2,2,2,1]

j = jets[2]
tX = tX_train[j]
y = y_train[j]
deg = degrees[2]


tX = preprocessing(tX)
tX = eliminate_outliers(tX, alpha)
accuracies = choose_parameters_l1_regression(y, tX, [deg], lambdas, k_fold = 3, seed = 1)[:,2]
    
plt.semilogx(lambdas, accuracies, marker='*', color = 'g', label="Accuracy - Jet N°3" )
plt.xlabel("lambda")
plt.ylabel("Accuracy")
plt.title("L1 - regularized logistic Regression")
leg = plt.legend(loc='best', shadow=True)
leg.draw_frame(False)
plt.savefig("accuracies_l1.png")
