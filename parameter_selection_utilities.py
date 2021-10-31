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

def choose_parameters_logistic_regression(degrees, lambdas, k_fold, y, tx, seed):
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
                acc_test = cross_validation_logistic(y, tx, k_indices, k, degree, lamb)[1]
                accs_test.append(acc_test)
            comparison.append([degree,lamb,np.mean(accs_test)])
    
    comparison = np.array(comparison)
    ind_best =  np.argmax(comparison[:,2])      
    best_deg = comparison[ind_best,0]
    best_l = comparison[ind_best,1]
    acc = comparison[ind_best,2]
   
    return best_deg, best_l, acc

def cross_validation_logistic(y, x, k_indices, k, degree, lambda_):
    """return the loss of logistic regression."""

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
    w = logistic_regression(y_tr, basis_tr, lambda_)[0]

    # calculate the accuracy for train and test data:
    y_tr_pred = predict_labels(w, basis_tr)
    y_te_pred = predict_labels(w, basis_te)
        
    acc_train = compute_accuracy(y_tr_pred, y_tr)
    acc_test = compute_accuracy(y_te_pred, y_te)

    return acc_train, acc_test

def choose_parameters_lasso_regression(degrees, lambdas, k_fold, y, tx, seed):
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
                acc_test = cross_validation_lasso(y, tx, k_indices, k, degree, lamb)[1]
                accs_test.append(acc_test)
            comparison.append([degree,lamb,np.mean(accs_test)])
    
    comparison = np.array(comparison)
    ind_best =  np.argmax(comparison[:,2])      
    best_deg = comparison[ind_best,0]
    best_l = comparison[ind_best,1]
    acc = comparison[ind_best,2]
   
    return best_deg, best_l, acc

def cross_validation_lasso(y, x, k_indices, k, degree, lambda_):
    """return the loss of lasso regression."""

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
    w = n(y_tr, basis_tr, lambda_)[0]

    # calculate the accuracy for train and test data:
    y_tr_pred = predict_labels(w, basis_tr)
    y_te_pred = predict_labels(w, basis_te)
        
    acc_train = compute_accuracy(y_tr_pred, y_tr)
    acc_test = compute_accuracy(y_te_pred, y_te)

    return acc_train, acc_test

def plot_train_test_logistic(train_errors, test_errors, accuracies, lambdas):
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a logistic regression on the train set
    * test_errors[0] = RMSE of the parameter found by logistic regression applied on the test set
    """
    plt.semilogx(lambdas, train_errors, color='b', marker='*', label="Train error")
    plt.semilogx(lambdas, test_errors, color='r', marker='*', label="Test error")
    plt.semilogx(lambdas, accuracies, color='g', marker='*', label="Accuracies")
    plt.xlabel("hyper-parameter")
    plt.ylabel("RMSE")
    plt.title("Regularized Logistic Regression")
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    
def plot_train_test_ridge(train_errors, test_errors, accuracies, lambdas):
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a logistic regression on the train set
    * test_errors[0] = RMSE of the parameter found by logistic regression applied on the test set
    """
    
    plt.semilogx(lambdas, train_errors, color='b', marker='*', label="Train error")
    plt.semilogx(lambdas, test_errors, color='r', marker='*', label="Test error")
    plt.semilogx(lambdas, accuracies, color='g', marker='*', label="Accuracies")
    plt.xlabel("hyper-parameter")
    plt.ylabel("RMSE")
    plt.title("Ridge Regression")
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)

def choose_lambda_logistic(y,tX, initial_w, maxiter, gamma):
    """
    Returns the optimal lambda obtained with cross-validation for logistic regression
    """
    seed = 1
    k_fold = 4
    lambdas = np.logspace(-6, 0, 20)

    # splitting data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    rmse_tr = []
    rmse_te = []
    accuracies = []

    for i in range(len(lambdas)):
        lambda_ = lambdas[i]
        tr_loss = 0
        te_loss = 0
        accuracy_i = 0
        for k in range(k_fold):
            loss_tr, loss_te, accuracy = cross_validation_logistic(y, tX, k_indices, k, lambda_, initial_w, maxiter, gamma)[1:]
            tr_loss = tr_loss + loss_tr
            te_loss = te_loss + loss_te
            accuracy_i = accuracy_i + accuracy
        rmse_tr.append(np.sqrt(2 * tr_loss/k_fold))
        rmse_te.append(np.sqrt(2 * te_loss/k_fold))
        accuracies.append(accuracy_i/k_fold)
        print ("lambda = ", lambdas[i], ' - accuracy = ', accuracy_i/k_fold)
    plot_train_test_logistic(rmse_tr, rmse_te, accuracies, lambdas)
    return lambdas[np.argmin(rmse_te)]

def choose_degree_logistic(y,tX, maxiter, gamma):
    """
    Returns the optimal degree obtained with cross-validation for logistic regression
    """
    seed = 1
    k_fold = 4
    degrees = [1,2,3,4,5,6,7]

    # splitting data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    rmse_tr = []
    rmse_te = []

    for i in range(len(degrees)):
        deg = degrees[i]
        tr_loss = 0
        te_loss = 0
        for k in range(k_fold):
            initial_w = np.zeros(tX.shape[1])
            loss_tr, loss_te = cross_validation_logistic_degree(y, tX, k_indices, deg, k, maxiter, gamma)[1:]
            tr_loss = tr_loss + loss_tr
            te_loss = te_loss + loss_te
        rmse_tr.append(np.sqrt(2 * tr_loss/k_fold))
        rmse_te.append(np.sqrt(2 * te_loss/k_fold))
    plot_train_test_logistic(rmse_tr, rmse_te, degrees)
    return degrees[np.argmin(rmse_te)]

def choose_lambda_ridge(y,tX, initial_w, maxiter, gamma):
    """
    Returns the optimal lambda obtained with cross-validation for ridge regression
    """
    seed = 1
    k_fold = 4
    lambdas = np.logspace(-6, 0, 20)

    # splitting data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    rmse_tr = []
    rmse_te = []

    for i in range(len(lambdas)):
        lambda_ = lambdas[i]
        tr_loss = 0
        te_loss = 0
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation(y, tX, k_indices, k, lambda_)[1:]
            tr_loss = tr_loss + loss_tr
            te_loss = te_loss + loss_te
        rmse_tr.append(np.sqrt(2 * tr_loss/k_fold))
        rmse_te.append(np.sqrt(2 * te_loss/k_fold))
    plot_train_test_ridge(rmse_tr, rmse_te, lambdas)
    return lambdas[np.argmin(rmse_te)]

def choose_degree_ridge(y,tX, maxiter, gamma):
    """
    Returns the optimal degree obtained with cross-validation for ridge regression
    """
    seed = 1
    k_fold = 4
    degrees = [1,2,3,4,5,6,7]
    lambda_ = 1e-6 # which resulted optimum

    # splitting data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    rmse_tr = []
    rmse_te = []

    for i in range(len(degrees)):
        deg = degrees[i]
        tr_loss = 0
        te_loss = 0
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation_degree(y, tX, k_indices, k, lambda_, deg)[1:]
            tr_loss = tr_loss + loss_tr
            te_loss = te_loss + loss_te
        rmse_tr.append(np.sqrt(2 * tr_loss/k_fold))
        rmse_te.append(np.sqrt(2 * te_loss/k_fold))
    print(rmse_te)
    print(rmse_tr)
    plot_train_test_ridge(rmse_tr, rmse_te, degrees)
    return degrees[np.argmin(rmse_te)]