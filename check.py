# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 16:36:25 2021

@author: giuli
"""
import numpy as np

DATA_TRAIN_PATH = '../data/true_solutions.csv'

y = np.genfromtxt(DATA_TRAIN_PATH, delimiter=",", skip_header=1, dtype=str, usecols=-3)

# convert class labels from strings to binary (-1,1)
yb = np.ones(len(y))
yb[np.where(y == 'b')] = -1

y_vere = yb[-568238:]

accuracy = np.count_nonzero(y_vere == y_pred)/len(y_pred)
