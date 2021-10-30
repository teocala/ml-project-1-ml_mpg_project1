import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from proj1_helpers import *

def missing_values_elimination(X):
    """
    Deletion of features with more than 70% missing values and imposition of the median in the remaining features
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

    X[:,cols_to_delete]=0

    return X

def normalize(X):
    """
    Normalization of the features values by division by the maximum value for each feature (not used in the final version)
    """
    D = X.shape[1]
    for i in range(D):
        maximum = np.max(abs(X[:,i]))
        if (maximum != 0.0) : X[:,i] = X[:,i]/maximum
    return X

def standardize(x, mean=None, std=None):
    """
    Standardization: mean is subtracted, then division by the standard deviation
    """
    if mean is None:
        mean = np.mean(x, axis=0)
    x = x - mean
    if std is None:
        std = np.std(x, axis=0)
    x = x[:, std > 0] / std[std > 0]

    return x

def eliminate_outliers(X, a):
    """
    Given the quantile of order a, the upper and lower tail of the data are cut, by imposing the value of the 1-a and a quantile, respcetively
    """
    D = X.shape[1]
    for i in range(D):
        X[:,i][ X[:,i]<np.quantile(X[:,i],a) ] = np.quantile(X[:,i],a)
        X[:,i][ X[:,i]>np.quantile(X[:,i],1-a) ] = np.quantile(X[:,i],1-a)

    return X


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

def build_poly(x, degree):
    """
    Polynomial basis functions for input data x, for j=0 up to j=degree
    """
    N, D = x.shape
    poly_basis = np.zeros(shape = (N, 1+D*(degree)))

    poly_basis[:,0] = np.ones(N)

    for deg in range(1,degree+1):
        for i in range(D):
            poly_basis[:, 1+D*(deg-1)+i ] = np.power(x[:,i],deg)

    return poly_basis

def root(x,t):
    """ 
    Computes the th-square of each element of a matrix
    """
    N, D = x.shape
    r = np.zeros([N,D])
    for i in range(N):
        for j in range(D):
            if x[i,j]>0:
                r[i,j] = x[i,j]**(1/t)
            else:
                r[i,j] = -(-x[i,j])**(1/t)
    return r

def build_poly_with_roots(x, degree):
    """
    Polynomial basis of degree M=degree for input data x, with square roots, cubic roots and pairwise products
    """
    N, D = x.shape
    temp_dict = {}
    count = 0

    for i in range(D):
        for j in range(i+1,D):
            temp = x[:,i]*x[:,j]
            temp_dict[count] = [temp]
            count = count + 1

    poly_basis = np.zeros(shape = (N, 1+D*(degree + 2) + count))

    poly_basis[:,0] = np.ones(N)

    for deg in range(1,degree+1):
        for i in range(D):
            poly_basis[:, 1+D*(deg-1)+i ] = np.power(x[:,i],deg)

    for m in range(count):
        poly_basis[:,1+2*D+m] = temp_dict[m][0]

    for i in range(D):
        poly_basis[:, 1+D*degree+count + i] = np.abs(x[:,i])**0.5

    poly_basis[:,1+D*degree+count+D:] = root(x,3)

    return poly_basis

def compute_accuracy(y_pred, y):
    """
    Computes accuracy
    """
    total = 0
    for i, y_val in enumerate(y):
        if y_val == y_pred[i]:
            total = total + 1

    return total / len(y)

def get_subset_PRI_jet_num(x, num_jet):
    """ 
    Returns the rows whose PRI_jet_num (feature in col 22) is equal to num_jet
    """
    return np.where(x[:,22] == num_jet)

def k_nearest(x, y, x_test, k):
    """
    Assigns the label y by using the K-nearest algorithm
    """
    # warnings:
    # 1) x and x_test must be corrected from missing data and standardized (other l^2 norm is unbalanced)
    # 2) k must be odd (otherwise doubt choice when k/2 vs k/2)
    # 3) y must be made of {0,1}

    n = x_test.shape[0]
    y_test = np.zeros(n)

    for i in range(n):
        norms = np.linalg.norm(x-x_test[i,:], axis=1)
        nearest_ids = norms.argsort()[:k]
        nearest_mean = y[nearest_ids].mean()
        if (nearest_mean >= 0.5):
            y_test[i] = 1
        else:
            y_test[i] = -1

        print ('Step ', i, ' of ', n) #it's indeed veeeery loooong

    return y_test

def log_transform(x):
    """ 
    Logaritmic transformation for positive features x, substitute x with log(1+x)
    """
    # The indexes of positive features are identified by plot analysis
    idx = [0,1,2,5,7,9,10,13,16,19,21,22,25]
    x_t1 = np.log1p(x[:, idx])
    x = np.hstack((x, x_t1))

    return x

def symmetric_transform(x):
    """
    Absolute value of symmetrical features
    """
    # The indexes of symmetrical features are identified by plot analysis
    # To avoid, eta parameters are intentionally symmetrical and the absolute value would lose their meaning
    idx = [14,17,23,26]
    x[:,idx]= abs(x[:,idx])

    return x

def angle_transform(x):
    """
    Tranformation for angles features
    """
    # Physically, these features are measures of angles. Thus, they need to be transformed before the regression
    idx = [15,18,20,24,27]
    x[:,idx] = np.cos(x[:,idx])

    return x

def PCA (tX):
    # Only indicative, seems that the last column is the only one negligible
    Z = tX-np.mean(tX)
    Z = Z / np.std(Z)
    Z = np.dot(Z.T, Z)
    eigenvalues, eigenvectors = np.linalg.eig(Z)
    D = np.diag(eigenvalues)
    Z_new = np.dot(Z, eigenvectors)
    fig, ax1 = plt.subplots(1,1)
    print ('PCA eigenvalues = ', eigenvalues)

def preprocessing(x):
    """
    Pre-processing of the given tensor x by applying the above implemented techniques
    """
    x = missing_values_elimination(x)
    x = log_transform(x)
    x = angle_transform(x)
    x = standardize(x)
    x = np.delete(x, [15,16,18,20], 1)
    return x
