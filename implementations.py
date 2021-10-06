import numpy as np
import matplotlib.pyplot as plt

# PLEASE, CHECK ALL THE IMPLEMENTATIONS
# Functions needed for the project (see "Step 2" in "project1_description.pdf")

# Methods taken from lab 2, Matteo

def compute_gradient_MSE(y, tx, w):
    e = y - tx @ w
    N = y.shape[0]
    g0 = -(1/N)*(e.T @ tx[:,0])
    g1 = -(1/N)*(e.T @ tx[:,1])
    return np.array([g0, g1])


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        w = w - gamma*compute_gradient_MSE(y,tx,w)
    loss = compute_loss(y, tx, w)
    return w,loss


def stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    g = 0
    for n_iter in range(max_iters):
        #batch_size is chosen 1
        for b_y, b_x in batch_iter(y, tx, 1):  # since n_batches is by default = 1, this loop is done only once
            g = gamma * compute_gradient_MSE(b_y, b_x, w)
        w = w - g
    loss = compute_loss(y, tx, w)
    return w,loss