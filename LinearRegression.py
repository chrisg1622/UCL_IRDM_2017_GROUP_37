# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 21:59:22 2017

@author: Chris
"""

from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import Utils as f
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
import metrics as m

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

#dict to store CV results
CV_results_df = pd.DataFrame()
#create 10-fold CV model
folds = KFold(n_splits=10, shuffle=True)
#lists to store metric values at each fold
rmse_scores = []
exp_var_scores = []
mean_abs_error_scores = []
mean_sq_error_scores = []
r2_scores = []
#carry out 10-fold CV on the data
fold = 1
print('Starting 10-fold cross validation')
for train_index, test_index in folds.split(X_train):
    
    #retrieve train - test split for current fold
    x_train = X_train.iloc[train_index]
    x_test = X_train.iloc[test_index]
    y_train = Y_train.iloc[train_index]
    y_test = Y_train.iloc[test_index]
    #create model
    model = LinearRegression()
    #train model
    model.fit(x_train, y_train)    
    #predict on the test split
    y_pred = model.predict(x_test)
    #rescale predictions
    y_pred_pp = post_process_preds(y_pred)
    #evaluate predictions and append score to lists
    exp_var,mean_abs_err,mean_sq_err,rmse,r2_sc = m.metrics_regress(y_pred_pp, y_test)
    #append scores to lists
    rmse_scores.append(rmse)
    exp_var_scores.append(exp_var)
    mean_abs_error_scores.append(mean_abs_err)
    mean_sq_error_scores.append(mean_sq_err)
    r2_scores.append(r2_sc)
    print('Fold: {}, \t RMSE: {:4f}'.format(fold, rmse_scores[-1]))
    for tup in [(rmse_scores,'rmse'),(exp_var_scores,'expl_var'),(mean_abs_error_scores,'mae'),(mean_sq_error_scores,'mse'),(r2_scores,'R2')]:
        scores, metric = tup   
        CV_results_df.loc[metric,'Fold '+str(fold)] = scores[-1]
    fold += 1

#compute mean and standard deviation of each metric, and append these to the results df
for tup in [(rmse_scores,'rmse'),(exp_var_scores,'expl_var'),(mean_abs_error_scores,'mae'),(mean_sq_error_scores,'mse'),(r2_scores,'R2')]:
    scores, metric = tup        
    #compute mean and std deviation of the scores
    mean = np.mean(scores)
    std = np.std(scores)
    print('10 fold CV '+metric+' mean: {:4f}'.format(mean))
    print('10 fold CV '+metric+' standard deviation: {:4f}'.format(std))
    CV_results_df.loc[metric,'mean'] = mean
    CV_results_df.loc[metric,'std'] = std

CV_results_df.to_csv('output/LinearRegressionCVResults.csv')

print('Training new model on training dataset...')
#train a new model on the whole training data
model = LinearRegression()
model.fit(X_train,Y_train)
print('Predicting on test dataset...')
#predict using the trained model
predictions = model.predict(X_test)
#rescale predictions to a 1-3 scale
predictions = post_process_preds(predictions)
#add index to predictions
predictions = pd.DataFrame(predictions, index = X_test.index)
print('Saving predictions to output folder...')
#save predictions
predictions.to_csv('output/LinearRegressionPredictions.csv')