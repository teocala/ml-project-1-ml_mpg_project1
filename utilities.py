import numpy as np

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
