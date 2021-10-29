# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 23:25:58 2021

@author: giuli

"""

import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import *
from utilities import *

def distributionsPlot(y,tX,featuresNames):
    savetX = tX
    savey = y
    alphaQuantile = 0

    for i in range(len(featuresNames)):

        y =  savey[(savetX[:,i] != - 999.0)]
        tX = savetX[(savetX[:,i] != - 999.0),:]
        
        if tX.shape[0]!=0:

            idPositive = [y==1][0]
            idNegative = [y==-1][0]

            plt.hist(tX[idPositive,i] ,100, histtype ='step',color='g',label='y == 1',density=True)      
            plt.hist(tX[idNegative,i] ,100, histtype ='step',color='m',label='y == -1',density=True)  
            plt.legend(loc = "upper right")
            plt.title("{name}, feature: {id}/{tot}".format(name=featuresNames[i],id=i,tot=len(featuresNames)-1), fontsize=12)
            plt.show()
            
def get_jet_masks(x):
    """
    Returns 3 masks corresponding to the rows of x where the feature 22 'PRI_jet_num'
    is equal to 0, 1 and  2 or 3 respectively.
    """
    return {
        0: x[:, 22] == 0,
        1: x[:, 22] == 1,
        2: x[:, 22] == 2,
        3: x[:, 22] == 3
    }


DATA_TRAIN_PATH = '../data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

featuresNames = np.genfromtxt('../data/train.csv', delimiter=",", dtype=str,max_rows=1)[2:]

msk_jets_train = get_jet_masks(tX)
for idx in range(len(msk_jets_train)):
        
        y_idx = y[msk_jets_train[idx]]
        x_idx = tX[msk_jets_train[idx],:]
        
        print('Jet',idx)
        print('Number of 1:', list(y_idx).count(1))
        print('Number of -1:', list(y_idx).count(-1))

        distributionsPlot(y_idx, x_idx, featuresNames)

        print('-------------------------------------------')
        print('-------------------------------------------')