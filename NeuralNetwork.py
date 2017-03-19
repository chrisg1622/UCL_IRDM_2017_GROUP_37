# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 21:59:22 2017

@author: Chris
"""

from sklearn.metrics import mean_squared_error
import numpy as np
import Utils as f
from sklearn.model_selection import train_test_split
import tensorflow as tf

#RMSE function
def RMSE(y,y_pred):
    return mean_squared_error(y, y_pred)**0.5

#function to cap predictions on a scale of 1 to 3
def post_process_preds(y_pred):
    return np.array([max(min(y[0],3.0),1.0) for y in y_pred])
    
    
print('Attempting to load pre-processed data...')
try:
    #load pre-processed training data
    X_train = f.load_obj('input_clean/X_train')
    X_test = f.load_obj('input_clean/X_test')
    Y_train = f.load_obj('input_clean/Y_train')
    print('Pre-processed data successfully loaded...')
except:
    print('Pre-processed data failed to load, ensure the working directory is correct...')
    
#split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.7, random_state = 1)

#parameters
N_feats = X_train.shape[1]
BATCH_SIZE = 64

#palceholders for X and Y
X = tf.placeholder(tf.float64, shape = (BATCH_SIZE, N_feats))
Y = tf.placeholder(tf.float64, shape = (BATCH_SIZE, 1)

#input layer
tf.layers.layer











