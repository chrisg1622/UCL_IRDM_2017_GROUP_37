
# coding: utf-8

# In[77]:

# All the necessary libs
import pandas as pd
import numpy as np
import Utils as f
from time import time

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor as xgb


# In[73]:

# Load training data
X = f.load_obj('input/X_train')
y = f.load_obj('input/Y_train')

clf = xgb()

optimization_df = pd.DataFrame(columns=['rank', 'time', 'rmse', 'gamma', 'learning_rate', 
                                    'reg_lambda', 'reg_alpha', 'max_depth',
                                    'colsample_bytree', 'colsample_bylevel',
                                    'subsample', 'min_child_weight'])
optimized_list = []
optimized_dict = {}

# Utility function to report best scores
def report(results, n_top=50):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            rmse = np.sqrt(abs(results['mean_test_score'][candidate]))
            optimized_dict = {'rank': i, 'time': results['mean_fit_time'][candidate], 'rmse': rmse,
                             'gamma': results['param_gamma'].data[candidate],
                              'learning_rate': results['param_learning_rate'].data[candidate],
                              'reg_lambda': results['param_reg_lambda'].data[candidate],
                              'reg_alpha': results['param_reg_alpha'].data[candidate],
                              'max_depth': results['param_max_depth'].data[candidate],
                              'colsample_bytree': results['param_colsample_bytree'].data[candidate],
                              'colsample_bylevel': results['param_colsample_bylevel'].data[candidate],
                              'subsample': results['param_subsample'].data[candidate],
                              'min_child_weight': results['param_min_child_weight'].data[candidate]}
            optimized_list.append(optimized_dict)
            print("Model with rank: {0}".format(i))
            print("Mean rmse score: {0:.3f})".format(
                  np.sqrt(abs(results['mean_test_score'][candidate]))))
            print("Parameters: {0}".format(results['params'][candidate]))            

# specify parameters and distributions to sample from
params = {
            'learning_rate': [0.01,0.015,0.025,0.05,0.1],
            'gamma': [0.05,0.1,0.3,0.5,0.7,0.9,1.0],
            'max_depth': [3,5,7,9,12,15,17,25],
            'min_child_weight':[1,3,5,7],
            'subsample':[0.6,0.7,0.8,0.9,1.0],
            'colsample_bytree':[0.6,0.7,0.8,0.9,1.0],
            'colsample_bylevel':[0.6,0.7,0.8,0.9,1.0],
            'reg_lambda': [0.001,0.01,0.1,1.0],
            'reg_alpha': [0,0.1,0.5,1.0]
          }

k_fold = KFold(3, random_state=1)

# run randomized search
n_iter_search = 50
random_search = RandomizedSearchCV(clf, param_distributions=params,scoring='neg_mean_squared_error', cv=k_fold,
                                   n_iter=n_iter_search)
start = time()
random_search.fit(X, y)
print("Randomized Search CV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)

# Save params
optimization_df = optimization_df.append(optimized_list)
optimization_df.to_csv('output/XGBoostOptimization.csv')


# In[76]:

# Save notebook as py 
get_ipython().system(u'jupyter nbconvert --to=python XGBoost_Optimization.ipynb')


# In[ ]:



