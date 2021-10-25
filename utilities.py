import numpy as np
import matplotlib.pyplot as plt
from implementations import *

def missing_values_elimination(X):
    """
    Deletion of features with more than 70% missing values and imposition of the mean in the remaining features
    """
    N, D = X.shape
    missing_data = np.zeros(D)
    cols_to_delete = []
    for i in range(D):
        missing_data[i] = np.count_nonzero(X[:,i]==-999)/N

        if missing_data[i]>0.7:
            cols_to_delete.append(i)

        elif missing_data[i]>0:
            X_feature = X[:,i]
            median = np.median(X_feature[X_feature != -999])
            X[:,i] = np.where(X[:,i]==-999, median, X[:,i])

    cols_to_keep = list(set(range(D)).difference(set(cols_to_delete)))
    X = np.delete(X, cols_to_delete, axis = 1)
    D_del = X.shape[1]
    return X,D_del,cols_to_delete,cols_to_keep

def missing_values_correction(X):
    """
    Correction of the missing terms (=-999) with the imposition of the mean value
    """
    N, D = X.shape
    missing_data = np.zeros(D)
    for i in range(D):
        missing_data[i] = np.count_nonzero(X[:,i]==-999)
        if missing_data[i]>0:
            X_feature = X[:,i]
            median = np.median(X_feature[X_feature != -999])
            X[:,i] = np.where(X[:,i]==-999, median, X[:,i])
    return X


def normalize(X):
    """
    Normalization of the features values by division by the maximum value for each feature
    """
    D = X.shape[1]
    for i in range(D):
        maximum = np.max(abs(X[:,i]))
        if (maximum != 0.0) : X[:,i] = X[:,i]/maximum
    return X

def standardize(x, mean=None, std=None):
    """
    Standardization of a vector: mean is subtracted, then division by the standard deviation
    """
    if mean is None:
        mean = np.mean(x, axis=0)
    x = x - mean
    if std is None:
        std = np.std(x, axis=0)
    x = x[:, std > 0] / std[std > 0]

    return x

def standardize_tX(X):
    """
    Standardization of the whole training dataset
    """
    N = X.shape[0]
    D = X.shape[1]
    for i in range(D):
        X[:,i] = np.reshape(standardize(X[:,i]),(N,))

    return X

def eliminate_outliers(X, a):
    """
    Given the quantile of order a, the upper and lower tail of the data are cut, by imposing the value of the 1-a and a quantile,   respcetively
    """
    D = X.shape[1]
    for i in range(D):
        X[:,i][ X[:,i]<np.quantile(X[:,i],a) ] = np.quantile(X[:,i],a)
        X[:,i][ X[:,i]>np.quantile(X[:,i],1-a) ] = np.quantile(X[:,i],1-a)

    return X


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_):
    """return the loss of ridge regression."""
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

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    N = x.shape[0]
    basis = np.zeros([N,degree+1])
    for n in range(N):
        for i in range(degree+1):
            basis[n,i] = x[n]**i
    return basis

def cross_validation_logistic(y, x, k_indices, k, lambda_, initial_w, max_iters, gamma):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train:
    ind = k_indices[k,:]
    x_te = x[ind]
    y_te = y[ind]
    ind_tr = np.delete(k_indices, (k), axis = 0)
    x_tr = np.vstack(x[ind_tr])
    y_tr = np.hstack(y[ind_tr])
    # ridge regression:
    loss_tr, w = reg_logistic_regression(y_tr, x_tr, lambda_,initial_w, max_iters, gamma)
    # calculate the loss for test data:
    loss_te = compute_loss_logistic(y_te, x_te, w, lambda_)
    return w, loss_tr, loss_te

def plot_train_test(train_errors, test_errors, lambdas):
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a logistic regression on the train set
    * test_errors[0] = RMSE of the parameter found by logistic regression applied on the test set
    
    degree is just used for the title of the plot.
    """
    plt.semilogx(lambdas, train_errors, color='b', marker='*', label="Train error")
    plt.semilogx(lambdas, test_errors, color='r', marker='*', label="Test error")
    plt.xlabel("lambda")
    plt.ylabel("RMSE")
    plt.title("Regularized Logistic Regression")
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    plt.savefig("reg_logistic_regression")

def choose_lambda_logistic(y,tX, initial_w, maxiter, gamma):
    seed = 1
    k_fold = 3
    lambdas = np.logspace(-4, 0, 30)

    # splitting data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    rmse_tr = []
    rmse_te = []

    for i in range(len(lambdas)):
        lambda_ = lambdas[i]
        tr_loss = 0
        te_loss = 0
        for k in range(k_fold): 
            loss_tr, loss_te = cross_validation_logistic(y, tX, k_indices, k, lambda_, initial_w, maxiter, gamma)[1:]
            tr_loss = tr_loss + loss_tr
            te_loss = te_loss + loss_te
        rmse_tr.append(np.sqrt(2 * tr_loss/k_fold))
        rmse_te.append(np.sqrt(2 * te_loss/k_fold))
    print(rmse_te)
    print(rmse_tr)
    plot_train_test(rmse_tr, rmse_te, lambdas) 
    return lambdas[np.argmin(rmse_te)]
