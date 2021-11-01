# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 09:30:30 2021

@author: giuli

"""

import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from utilities import *
from proj1_helpers import *

def build_k_indices(y, k_fold, seed):
    """
    Build k indices for k-fold
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def choose_parameters_ridge_regression(degrees, lambdas, k_fold, y, tx, seed):
    """
    Returns the hyper-parameters among the ones passed as input which maximize the accuracy predicted with cross validation
    """
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    comparison = []

    for degree in degrees:
        for lamb in lambdas:
            print('Degree: ',degree)
            print('Lambda: ',lamb)
            accs_test = []
            for k in range(k_fold):
                acc_test = cross_validation_ridge(y, tx, k_indices, k, degree, lamb)[1]
                accs_test.append(acc_test)
            comparison.append([degree,lamb,np.mean(accs_test)])

    comparison = np.array(comparison)
    ind_best =  np.argmax(comparison[:,2])
    best_deg = comparison[ind_best,0]
    best_l = comparison[ind_best,1]
    acc = comparison[ind_best,2]

    #plot_train_test_ridge(comparison[:,2], lambdas, degrees)

    return best_deg, best_l, acc

def cross_validation_ridge(y, x, k_indices, k, degree, lambda_):
    """return the loss of ridge regression."""

    # get k'th subgroup in test, others in train:
    ind = k_indices[k,:]
    ind_tr = np.delete(k_indices, (k), axis = 0)
    x_te = x[ind]
    x_tr = np.vstack(x[ind_tr])
    y_te = y[ind]
    y_tr = np.hstack(y[ind_tr])

    # form data with polynomial degree:
    basis_tr = build_poly_with_roots(x_tr, degree)
    basis_te = build_poly_with_roots(x_te, degree)

    # ridge regression:
    w = ridge_regression(y_tr, basis_tr, lambda_)[0]

    # calculate the accuracy for train and test data:
    y_tr_pred = predict_labels(w, basis_tr)
    y_te_pred = predict_labels(w, basis_te)

    acc_train = compute_accuracy(y_tr_pred, y_tr)
    acc_test = compute_accuracy(y_te_pred, y_te)

    return acc_train, acc_test


def choose_parameters_l1_regression(y, tx, degrees, lambdas, k_fold, seed):
    """
    Returns the hyper-parameters among the ones passed as input which maximize the accuracy predicted with cross validation
    """
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    comparison = []

    for degree in degrees:
        for lamb in lambdas:
            print('Degree: ',degree)
            print('Lambda: ',lamb)
            accs_test = []
            for k in range(k_fold):
                acc_test = cross_validation_l1(y, tx, k_indices, k, degree, lamb)[1]
                accs_test.append(acc_test)
            comparison.append([degree,lamb,np.mean(accs_test)])
    comparison = np.matrix(comparison)

    return comparison

def cross_validation_l1(y, x, k_indices, k, degree, lambda_):
    """return the loss of lasso regression."""

    gamma = 0.0001
    max_iters = 500

    # get k'th subgroup in test, others in train:
    ind = k_indices[k,:]
    ind_tr = np.delete(k_indices, (k), axis = 0)
    x_te = x[ind]
    x_tr = np.vstack(x[ind_tr])
    y_te = y[ind]
    y_tr = np.hstack(y[ind_tr])

    # form data with polynomial degree:
    basis_tr = build_poly_with_roots(x_tr, degree)
    basis_te = build_poly_with_roots(x_te, degree)

    # l1 regularized logistic regression:
    initial_w = np.zeros(basis_tr.shape[1])
    w = fista(y_tr, basis_tr, initial_w, max_iters, gamma, lambda_)[1]

    # calculate the accuracy for train and test data:
    y_tr_pred = predict_labels(w, basis_tr)
    y_te_pred = predict_labels(w, basis_te)

    acc_train = compute_accuracy(y_tr_pred, y_tr)
    acc_test = compute_accuracy(y_te_pred, y_te)


    return acc_train, acc_test



def plot_accuracies_l1(accuracies, lambdas):
    """
    train_errors, test_errors and lambdas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    """
    for i in range(accuracies.shape[1]):
        i_plt = str(i+1)
        plt.semilogx(lambdas, accuracies[:,i], marker='*', label="Accuracy - Jet N° " + i_plt)
    plt.xlabel("lambda")
    plt.ylabel("Accuracy")
    plt.title("L1 - regularized logistic Regression")
    leg = plt.legend(loc='best', shadow=True)
    leg.draw_frame(False)
    plt.savefig("accuracies_l1.png")



def plot_train_test_ridge(accuracies, lambdas, degrees):
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a logistic regression on the train set
    * test_errors[0] = RMSE of the parameter found by logistic regression applied on the test set
    """

    plt.semilogx(lambdas, accuracies, color='g', marker='*', label="Accuracies")
    plt.xlabel("lambda")
    plt.ylabel("Accuracy")
    plt.title("Ridge Regression")
    leg = plt.legend(loc=8, shadow=True)
    leg.draw_frame(False)
    plt.semilogx(degrees, accuracies, color='m', marker='*', label="Accuracies")
    plt.title("Ridge Regression")
    plt.xlabel("degree")
    plt.ylabel("Accuracy")
    leg = plt.legend(loc=8, shadow=True)
    leg.draw_frame(False)
