import numpy as np
from implementations import *

def missing_values(X):
    """
    Deletion of features with more than 70% missing values and imposition of the mean in the remaining features
    """
    N, D = X.shape
    missing_data = np.zeros(D)
    cols_todelete = [] 
    for i in range(D):
        missing_data[i] = np.count_nonzero(X[:,i]==-999)/N
      
        if missing_data[i]>0.7: 
            cols_todelete.append(i)
           
        elif missing_data[i]>0:
            X_feature = X[:,i]
            mean = np.mean(X_feature[X_feature != -999])
            X[:,i] = np.where(X[:,i]==-999, mean, X[:,i]) 
                    
    X = np.delete(X, cols_todelete, axis = 1)
    D = X.shape[1]    
    return X,D

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
    #print(ind_tr.shape)
    x_tr = np.vstack(x[ind_tr])
    y_tr = np.hstack(y[ind_tr])
    # ridge regression:
    #print(y_tr.shape)
    #print(x_tr.shape)
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
