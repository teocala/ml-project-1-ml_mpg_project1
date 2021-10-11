import numpy as np
import matplotlib.pyplot as plt

# PLEASE, CHECK ALL THE IMPLEMENTATIONS
# Functions needed for the project (see "Step 2" in "project1_description.pdf")


# Methods from LAB 2

def compute_loss_MSE(y, tx, w):
    e = y - tx @ w
    N = len(y)
    return (e**2).sum()/(2*N)


def compute_gradient_MSE(y, tx, w):
    e = y - tx @ w
    N = len(y)
    g = -(1/N)*(tx.T @ e)
    return g


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        w = w - gamma*compute_gradient_MSE(y,tx,w)
        loss = compute_loss_MSE(y, tx, w)
    return w,loss


def least_squares_SGD(y, tx, initial_w, batch_size = 1, max_iters, gamma):
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
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = compute_loss_MSE(y,tx,w)
    return w,loss


def ridge_regression(y, tx, lambda_):
    N = len(y)
    D = tx.shape[1]
    I = np.eye(D)
    w = np.linalg.solve(tx.T @ tx + 2*N*lambda_*I, tx.T @ y)
    loss = compute_loss_MSE(y, tx, w)
    return w,loss

