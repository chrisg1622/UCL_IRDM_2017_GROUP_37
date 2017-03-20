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
import pandas as pd
from time import time

#RMSE function
def RMSE(y,y_pred):
    return mean_squared_error(y, y_pred)**0.5

#function to cap predictions on a scale of 1 to 3
def post_process_preds(y_pred):
    return np.array([max(min(y[0],3.0),1.0) for y in y_pred])
    
    
print('Attempting to load pre-processed data...')
try:
    #load pre-processed training data
    X = f.load_obj('input_clean/X_train')
    Y = f.load_obj('input_clean/Y_train')
    print('Pre-processed data successfully loaded...')
except:
    print('Pre-processed data failed to load, ensure the working directory is correct...')
    
#dataframe to store random search results
try: 
    results_df = pd.read_csv('output/NeuralNetworkOptimization.csv',index_col='Round')
    print('Previous optimization results successfully loaded...')
except:
    results_df = pd.DataFrame(columns = ['batch_size',
                                         'h1_units',
                                         'h1_reg',
                                         'h2_units', 
                                         'h2_reg',
                                         'h3_units', 
                                         'h3_reg',
                                         'learning_rate'])
    print('Previous optimization results failed to load, starting new optimization experiment...')

#split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.7, random_state = 1)

#number of features in data
N_feats = X_train.shape[1]

#parameters
BATCH_SIZE_ = [64,128,256,512]
h1_units_ = [100,200,300,400,500]
h1_reg_ = [0.1,0.2,0.4,0.8,1.6]
h2_units_ = [50,100,150,200,250]
h2_reg_ = [0.1,0.2,0.4,0.8,1.6]
h3_units_ = [25,50,75,100,125]
h3_reg_ = [0.1,0.2,0.4,0.8,1.6]
learning_rate_ = [0.01,0.001,0.0001]

#palceholders for X and Y
x = tf.placeholder(tf.float64, shape = (None, N_feats))
y = tf.placeholder(tf.float64, shape = (None, 1))

for i in range(10):
    #randomly sample params from given ranges
    BATCH_SIZE = np.random.choice(BATCH_SIZE_,size=1)[0]
    h1_units = np.random.choice(h1_units_,size=1)[0]
    h1_reg = np.random.choice(h1_reg_,size=1)[0]
    h2_units = np.random.choice(h2_units_,size=1)[0]
    h2_reg = np.random.choice(h2_reg_,size=1)[0]
    h3_units = np.random.choice(h3_units_,size=1)[0]
    h3_reg = np.random.choice(h3_reg_,size=1)[0]
    learning_rate = np.random.choice(learning_rate_,size=1)[0]
        
    
    with tf.variable_scope("Scope"+str(i)) as varscope:
        #hidden layer 1
        hidden_1 = tf.layers.dense(x, units = h1_units, 
                                          activation = tf.nn.sigmoid, 
                                          use_bias = True, 
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(h1_reg),
                                          bias_regularizer=tf.contrib.layers.l2_regularizer(h1_reg),
                                          name = 'hidden_1')
        
        #hidden layer 2
        hidden_2 = tf.layers.dense(hidden_1, units = h2_units, 
                                          activation = tf.nn.sigmoid, 
                                          use_bias = True, 
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(h2_reg),
                                          bias_regularizer=tf.contrib.layers.l2_regularizer(h2_reg),
                                          name = 'hidden_2')
        
        #hidden layer 3
        hidden_3 = tf.layers.dense(hidden_2, units = h3_units, 
                                          activation = tf.nn.sigmoid, 
                                          use_bias = True, 
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(h3_reg),
                                          bias_regularizer=tf.contrib.layers.l2_regularizer(h3_reg),
                                          name = 'hidden_3')
        
        #output (i.e. y)
        y_pred = tf.layers.dense(hidden_3, units = 1, activation = None, name = 'output')
    
    #loss
    loss = tf.reduce_sum(tf.losses.mean_squared_error(y, y_pred))
    
    #optimizer
    opt_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    
    #run max 20 epochs
    with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            #find number of stories
            n = X_train.shape[0]
            
            min_test_rmse_score = 2.0
            min_test_rmse_epoch = 0
            min_test_rmse_time = 0.0
            for epoch in range(20):
                print('----- Epoch', epoch, '-----')
                total_loss = 0
                #start timer for timing training
                t0 = time()
                #sample batches randomly
                for i in range(n // BATCH_SIZE):
                    indices = np.random.choice(n,BATCH_SIZE)
                    feed_dict = {x: X_train.iloc[indices], 
                                 y: Y_train.iloc[indices]}
                    #compute loss at current batch
                    _, current_loss = sess.run([opt_op, loss], feed_dict=feed_dict)
                    #update total loss
                    total_loss += current_loss
                print(' Train loss:', total_loss / n)
                t1 = time()
                
                #define feed_dict for training data            
                train_feed_dict = {x: X_train,
                                   y: Y_train}
                #predict on training data
                Y_train_pred = sess.run(y_pred, feed_dict = train_feed_dict)
                #postprocess predictions
                Y_train_pred_pp = post_process_preds(Y_train_pred)
                #compute training rmse score
                train_rmse = RMSE(Y_train, Y_train_pred_pp)
                print(' Train rmse:', round(train_rmse,5))
                
                #define feed_dict for test data            
                test_feed_dict = {x: X_test,
                                   y: Y_test}
                #predict on training data
                Y_test_pred = sess.run(y_pred, feed_dict = test_feed_dict)
                #postprocess predictions
                Y_test_pred_pp = post_process_preds(Y_test_pred)
                #compute training rmse score
                test_rmse = RMSE(Y_test, Y_test_pred_pp)
                print(' Test rmse:', round(test_rmse,5))
                #if necessary, update the min rmse score
                if test_rmse < min_test_rmse_score:
                    min_test_rmse_score = test_rmse
                    min_test_rmse_epoch = epoch
                    min_test_rmse_time = round(t1-t0)
            #store current results
            current_results = pd.DataFrame({'batch_size':[BATCH_SIZE],
                                            'h1_units':[h1_units],
                                            'h1_reg':[h1_reg],
                                            'h2_units':[h2_units], 
                                            'h2_reg':[h2_reg],
                                            'h3_units':[h3_units], 
                                            'h3_reg':[h3_reg],
                                            'learning_rate':[learning_rate],             
                                            'best_test_rmse':[min_test_rmse_score],             
                                            'best_test_rmse_epoch':[min_test_rmse_epoch],             
                                            'training_time (on best epoch)':['{}s'.format(min_test_rmse_time)]})              
            print('Current Results: \n', current_results)
            #append results to dataframe
            results_df = pd.concat([results_df,current_results])
            results_df = results_df.reset_index(drop=True)
            #save results
            results_df.to_csv('output/NeuralNetworkOptimization.csv', index_label = 'Round')







