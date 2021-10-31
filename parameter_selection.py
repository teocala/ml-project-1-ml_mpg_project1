# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 09:30:30 2021

@author: giuli

"""

import numpy as np
import matplotlib.pyplot as plt
from implementations import *
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


def cross_validation_ridge_lambda(y, x, k_indices, k, lambda_):
    """
    Utility for the cross validation on lambda, for Ridge Regression
    """
    # get k'th subgroup in test, others in train:
    ind = k_indices[k,:]
    x_te = x[ind]
    y_te = y[ind]
    ind_tr = np.delete(k_indices, (k), axis = 0)
    x_tr = np.vstack(x[ind_tr])
    y_tr = np.hstack(y[ind_tr])
    # ridge regression:
    w, loss_tr = ridge_regression(y_tr, x_tr, lambda_)
    # calculate the loss for test data:
    e_te = y_te - x_te.dot(w)
    loss_te = 1/(2*len(y_te)) * np.transpose(e_te).dot(e_te)

    return w, loss_tr, loss_te

def cross_validation_ridge_degree(y, x, k_indices, k, lambda_, degree):
    """
    Utility for the cross validation on degree, for Ridge Regression
    """
    # get k'th subgroup in test, others in train:
    ind = k_indices[k,:]
    x_te = x[ind]
    y_te = y[ind]
    ind_tr = np.delete(k_indices, (k), axis = 0)
    x_tr = np.vstack(x[ind_tr])
    y_tr = np.hstack(y[ind_tr])
    x_tr = build_poly(x_tr,degree)
    x_te = build_poly(x_te,degree)
    # ridge regression:
    w, loss_tr = ridge_regression(y_tr, x_tr, lambda_)
    # calculate the loss for test data:
    e_te = y_te - x_te.dot(w)
    loss_te = 1/(2*len(y_te)) * np.transpose(e_te).dot(e_te)

    return w, loss_tr, loss_te

def cross_validation_logistic_lambda(y, x, k_indices, k, lambda_, initial_w, max_iters, gamma):
    """
    Utility for the cross validation on lambda, for Logistic Regression
    """
    # get k'th subgroup in test, others in train:
    ind = k_indices[k,:]
    x_te = x[ind]
    y_te = y[ind]
    ind_tr = np.delete(k_indices, (k), axis = 0)
    ind_tr = np.hstack(ind_tr)
    x_tr = x[ind_tr]
    y_tr = y[ind_tr]
    # l1 regression with fista
    loss_tr, w = fista(y_tr, x_tr, initial_w, max_iters, gamma, lambda_)
    # calculate the loss for test data:
    loss_tr = compute_loss_logistic(y_tr, x_tr, w)
    loss_te = compute_loss_logistic(y_te, x_te, w)
    y_pred = predict_labels(w, x_te)
    y_pred[y_pred<0]=0
    accuracy = compute_accuracy(y_te, y_pred)
    return w, loss_tr, loss_te, accuracy

def cross_validation_logistic_degree(y, x, k_indices, deg, k, max_iters, gamma):
    """
    Utility for the cross validation on degree, for Logistic Regression
    """
    # get k'th subgroup in test, others in train:
    ind = k_indices[k,:]
    x_te = x[ind]
    y_te = y[ind]
    ind_tr = np.delete(k_indices, (k), axis = 0)
    ind_tr = np.hstack(ind_tr)
    x_tr = x[ind_tr]
    y_tr = y[ind_tr]
    x_tr = build_poly(x_tr,deg)
    x_te = build_poly(x_te,deg)
    # logistic regression:
    initial_w = np.zeros(x_tr.shape[1])
    loss_tr, w = logistic_regression(y_tr, x_tr, initial_w, max_iters, gamma)
    # calculate the loss for test data:
    loss_tr = compute_loss_logistic(y_tr, x_tr, w)
    loss_te = compute_loss_logistic(y_te, x_te, w)
    return w, loss_tr, loss_te

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