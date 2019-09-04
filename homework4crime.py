#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 21:22:17 2018

@author: shirlynn
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#  square error  cost function
def computeCost(X1,Y,w,reg):
    """
    input: features matrix X1(with column of bias) and object vector Y, weight vector w
    X1: m x n
    Y: m x 1
    w: n x 1
    """
    m = len(Y)
    cost = (np.linalg.norm((np.dot(X1,w)-Y)))**2 
    return cost
# data_reading function
def read_input(path):
    return pd.read_table(path)