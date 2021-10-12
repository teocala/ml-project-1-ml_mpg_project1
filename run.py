"""
Main script to work and get results from the dataset
"""


"""Libraries importation"""

import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import *



""" MAIN """

if __name__ == '__main__':

    """ Data loading (lines from ML_course/projects/project1/scripts/project1.ipynb"""

    # TODO: In the same directory of your repository folder, create a folder "data" where you place the data downloaded from
    #  1) https://github.com/epfml/ML_course/tree/master/projects/project1/data
    #  or
    #  2) https://www.aicrowd.com/challenges/epfl-machine-learning-higgs

    DATA_TRAIN_PATH = '../data/train.csv'
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    N = len(y) # training set cardinality
    D = tX.shape[1] # number of parameters ("dimensionality")
