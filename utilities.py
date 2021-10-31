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
    x = symmetric_transform(x)
    x = standardize(x)
    x = np.delete(x, [15,16,18,20], 1)
    return x
