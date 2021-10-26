import numpy as np
import matplotlib.pyplot as plt

# PLEASE, CHECK ALL THE IMPLEMENTATIONS
# Functions needed for the project (see "Step 2" in "project1_description.pdf")


# Methods from LAB 2

def compute_loss_MSE(y, tx, w):
    
    """
    Computes the loss function using the Mean Squared Error as Cost
    INPUTS: y = target, tx = sample matrix, w = weights vector
    OUTPUT: evaluation of the MSE given the inputs
    """
    
    e = y - tx @ w
    N = len(y)
    return (e**2).sum()/(2*N)


def compute_gradient_MSE(y, tx, w):
    
    """
    Computes the gradient of the Loss function with MSE
    INPUTS: y = target, tx = sample matrix, w = weights vector
    OUTPUT: g = gradient of the MSE with respect to w
    """
    e = y - tx @ w
    N = len(y)
    g = -(1/N)*(tx.T @ e)
    return g


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    
    """
    Gradient Descent algorithm  using MSE as Cost 
    INPUTS: y = target, tx = sample matrix, w = intial guess for the weights vector, max_iters = maximum number of iterations, gamma = learning rate
    OUTPUT: w = weight vector computed with GD after max_iters iterations, loss = loss evaluation at w
    """
    
    w = initial_w
    for n_iter in range(max_iters):
        w = w - gamma*compute_gradient_MSE(y,tx,w)
    loss = compute_loss_MSE(y, tx, w)
    return w,loss

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    
    # Auxiliary function for least_squares_SGD
    
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]



def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size = 1):
    
    """
    Stochastic Gradient Descent algorithm
    INPUTS: y = target, tx = sample matrix, initial_w = intial guess for the weights vector, batch_size = number of samples on which the new gradient is computed (by default = 1), max_iters = maximum number of iterations, gamma = learning rate
    OUTPUT: w = weights vector at last iteration, loss = MSE evaluation at last iteration
    """
    w = initial_w
    g = 0
    for n_iter in range(max_iters):
        for b_y, b_x in batch_iter(y, tx, batch_size): #batch_size is chosen 1 if no parameter is passed
            g = gamma * compute_gradient_MSE(b_y, b_x, w)
        w = w - g
    loss = compute_loss_MSE(y, tx, w)
    return w,loss




# Methods from LAB 3

def least_squares(y, tx):
    
    """
    Computation of the weights vector by solving the normal equations for linear regression
    INPUTS: y = target, tx = sample matrix
    OUTPUTS: w = weights vector, loss = corresponding MSE evaluation
    """
    
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = compute_loss_MSE(y,tx,w)
    return w,loss


def ridge_regression(y, tx, lambda_):
    
    """
    Computation of the weights vector by solving the L2-regularized normal equations for linear regression
    INPUTS: y = target, tx = sample matrix, lambda_ = regularization parameter
    OUTPUTS: w = weights vector, loss = corresponding MSE evaluation
    """
    
    N = len(y)
    D = tx.shape[1]
    I = np.eye(D)
    w = np.linalg.solve(tx.T @ tx + 2*N*lambda_*I, tx.T @ y)
    loss = compute_loss_MSE(y, tx, w)
    return w,loss





# Methods from LAB 5

sigmoid = lambda x : 1/(1+np.exp(-x))
"""Sigmoid function for logistic regression"""


def compute_loss_logistic(y, tx, w):
    """
    Computation of the logistic loss (negative log likelihood)
    INPUTS: y = target, tx = sample matrix, w = weights vector
    OUTPUTS: evaluation of the loss
    """
    n = len(y)
    return (-y.T.dot(np.log(sigmoid(tx.dot(w)))) - (1-y).T.dot(np.log(1 - sigmoid(tx.dot(w)))))/n


def compute_gradient_logistic(y, tx, w):
    """
    Computation of the gradient of the logistic loss (negative log likelihood)
    INPUTS: y = target, tx = sample matrix, w = weights vector
    OUTPUTS: evaluation of the gradient of the loss
    """
    n = len(y)
    return -tx.T.dot(y-sigmoid(tx.dot(w)))/n


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic Gradient Descent algorithm
    INPUTS: y = target, tx = sample matrix, initial_w = intial guess for the weights vector, max_iters = maximum number of iterations, gamma = learning rate
    OUTPUT: w = weights vector at last iteration, loss = logistic loss evaluation at last iteration
    """
    w = initial_w
    for iter in range(max_iters):
        loss = compute_loss_logistic(y,tx,w)
        g = compute_gradient_logistic(y,tx,w)
        w = w - gamma*g
    print(loss)
    return loss, w


def compute_loss_reg_logistic(y, tx, w, lambda_):
    """
    Computation of the regularized logistic loss (negative log likelihood)
    INPUTS: y = target, tx = sample matrix, w = weights vector
    OUTPUTS: evaluation of the loss
    """
    return compute_loss_logistic(y,tx,w) + lambda_*np.linalg.norm(w,2)**2


def compute_gradient_reg_logistic(y, tx, w, lambda_):
    """
    Computation of the gradient of the regularized logistic loss (negative log likelihood)
    INPUTS: y = target, tx = sample matrix, w = weights vector
    OUTPUTS: evaluation of the gradient of the loss
    """
    return compute_gradient_logistic(y,tx,w) + 2*lambda_*w


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized Logistic Gradient Descent algorithm
    INPUTS: y = target, tx = sample matrix, lambda_ = regularization parameter, initial_w = intial guess for the weights vector, max_iters = maximum number of iterations, gamma = learning rate
    OUTPUT: w = weights vector at last iteration, loss = logistic loss evaluation at last iteration
    """
    w = initial_w
    for iter in range(max_iters):
        loss = compute_loss_reg_logistic(y,tx,w, lambda_)
        g = compute_gradient_reg_logistic(y,tx,w, lambda_)
        w = w - gamma*g
    return loss, w


