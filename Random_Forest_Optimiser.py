# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 21:59:22 2017

@author: Chris
"""

from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import Utils as f
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#RMSE function
def RMSE(y,y_pred):
    return mean_squared_error(y, y_pred)**0.5

#function to cap predictions on a scale of 1 to 3
def post_process_preds(y_pred):
    return np.array([max(min(y,3.0),1.0) for y in y_pred])
    
    
print('Attempting to load pre-processed data...')
try:
    #load pre-processed training data
    X = f.load_obj('input/X_train')
    Y = f.load_obj('input/Y_train')
    print('Pre-processed data successfully loaded...')
except:
    print('Pre-processed data failed to load, ensure the working directory is correct...')
    
#split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, random_state = 1)

#define possible params for random search
params = {
            'n_estimators': [10,20,40,80,160,320],
            'max_features': [0.6,0.8,1.0,'auto','sqrt','log2'],
            'max_depth': [4,8,16,32,64],
            'n_jobs': [3]
          }

#dataframe to store random search results
results_df = pd.DataFrame(columns = ['n_estimators','max_features','max_depth','val_rmse'])

#perform the random search several times
for i in range(2):
    print('-- Starting search iteration {} --'.format(i+1))
    n_estimators = np.random.choice(params['n_estimators'],size=1)[0]
    max_features = np.random.choice(params['max_features'],size=1)[0]
    max_depth = np.random.choice(params['max_depth'],size=1)[0]
    print('Training Random Forest...')    
    #define model
    model = RandomForestRegressor(n_estimators = n_estimators,
                                  max_features = max_features,
                                  max_depth = max_depth)
    #train model on train split
    model.fit(X_train, Y_train)
    print('Predicting on test data and evaluating results...')
    #predict on test split
    Y_test_preds = model.predict(X_test)
    #evaluate predictions
    rmse = RMSE(Y_test, Y_test_preds)
    #store current results
    current_results = pd.DataFrame({'n_estimators': n_estimators,
                                'max_features':max_features,
                                'max_depth':max_depth,
                                'test_rmse':rmse})
    print('Current Results: \n', current_results)
    #append results to dataframe
    results_df.append(current_results)
    
