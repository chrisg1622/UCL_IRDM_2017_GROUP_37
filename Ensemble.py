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

models=['LinearRegression',
          'Lasso',
          'Ridge',
          'ElasticNet',
          'KNN',
          'XGBoost',
          'RandomForest',
          'NeuralNetwork']

CV_results = {}
predictions = {}
print('Attempting to load CV Results...')
for model in models:
    print('Reading {} CV Results...'.format(model))
    CVResults_df = pd.read_csv(str('output/'+model+'CVResults.csv'),index_col='Unnamed: 0')
    preds_df = pd.read_csv(str('output/'+model+'Predictions.csv'),index_col='id')
    CV_results[model] = CVResults_df.loc['rmse','mean']
    predictions[model] = preds_df
    
    