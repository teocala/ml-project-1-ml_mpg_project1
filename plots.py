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

def FeaturesPlot(y,tX,features):

    for i in range(len(features)):

        y =  y[(tX[:,i] != - 999.0)]
        tX = tX[(tX[:,i] != - 999.0),:]
        
        if tX.shape[0]!=0:

            idPositive = [y==1][0]
            idNegative = [y==-1][0]

            plt.hist(tX[idPositive,i] ,100, histtype ='step',color='g',label='y == 1',density=True)      
            plt.hist(tX[idNegative,i] ,100, histtype ='step',color='m',label='y == -1',density=True)  
            plt.legend(loc = "upper right")
            plt.title("{name}, feature: {id}/{tot}".format(name=features[i],id=i,tot=len(features)-1), fontsize=12)
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

def labels_in_training(y,tX):
    msk_jets_train = get_jet_masks(tX)

    ax = plt.subplot(111)
    colors = ['g','m','y','b']
    legend = ['num_jet = 0','num_jet = 1','num_jet = 2','num_jet = 3']
    ind = np.array([-1,  1])
    w = 0.25
    for i in range(len(msk_jets_train)):
        y_i = y[msk_jets_train[i]]
        count_prediction = {-1:  np.count_nonzero(y_i == -1), 1:  np.count_nonzero(y_i == 1)}
        ax.bar(ind+w*i, count_prediction.values(), width=w, color=colors[i],align='center')

    ax.set_ylabel('Number of training data')
    ax.set_xticks(ind+0.25)
    ax.set_xticklabels( ('-1', '1') )
    ax.legend(legend)
    ax.plot()

DATA_TRAIN_PATH = '../data/train.csv'
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

features = np.genfromtxt('../data/train.csv', delimiter=",", dtype=str,max_rows=1)[2:]

num_jets_train = get_jet_masks(tX)
for i in range(len(num_jets_train)):
        
        y_i = y[num_jets_train[i]]
        x_i = tX[num_jets_train[i],:]
        
        print('Jet',i)
        print('Number of 1:', list(y_i).count(1))
        print('Number of -1:', list(y_i).count(-1))

        FeaturesPlot(y_i, x_i, features)
        
labels_in_training(y,tX)
