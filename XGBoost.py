<<<<<<< HEAD

# coding: utf-8

# In[1]:

# Import necessary libraries
import pandas as pd
import numpy as np
import Utils as f
import time

import xgboost
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import eval_metrics as m


# In[2]:

# load cleaned training data
X_train = f.load_obj('input_clean/X_train')
X_test = f.load_obj('input_clean/X_test')
Y_train = f.load_obj('input_clean/Y_train')


# In[3]:

# Evaluate all metrics
store = []
def metric_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    exp_var,mean_abs_err,mean_sq_err,rms,r2_sc = m.metrics_regress(y, y_pred)
    global store
    store.append(m.metrics_regress(y, y_pred))
    return rms


# In[4]:

# Optimal parameters
param_df = pd.DataFrame()
param_df = param_df.from_csv('output/XGBoostOptimization.csv')
opt_params = param_df.iloc[0][3:]
opt_params_df = pd.DataFrame(opt_params).T
opt_params_df


# In[5]:

# 10 fold CV with XGBoost

# Best parameters already found by tuning
xgb = xgboost.XGBRegressor(gamma=opt_params_df.gamma[0],
                           colsample_bytree=opt_params_df.colsample_bytree[0],
                           colsample_bylevel=opt_params_df.colsample_bylevel[0],
                           learning_rate=opt_params_df.learning_rate[0],
                           max_depth=int(opt_params_df.max_depth[0]),
                           min_child_weight=opt_params_df.min_child_weight[0],
                           subsample=opt_params_df.subsample[0],
                           reg_alpha=opt_params_df.reg_alpha[0],
                           reg_lambda=opt_params_df.reg_lambda[0])

k_fold = KFold(10, random_state=1)

start = time.time()
print('Started 10 Fold cross validation')

xgb_scores = cross_val_score(xgb, X_train, Y_train, cv=k_fold, scoring=metric_scorer, n_jobs=1)

print('Finished Cross validation in {} mins'.format(round(((time.time() - start)/60),2)))


# In[6]:

# CV results
cv_results_df = pd.DataFrame(store)
cv_results_df.columns = ['expl_var','mae','mse','rmse','r2']
cv_results_df = cv_results_df.T
cv_results_df.columns = ['Fold '+str(i+1) for i in cv_results_df.columns]
means = []
stds = []
for i in range(0,5):
    means.append(np.mean(cv_results_df.iloc[i].values))
    stds.append(np.std(cv_results_df.iloc[i].values))
cv_results_df['mean'] = means
cv_results_df['std'] = stds
cv_results_df.to_csv('output/XGBoostCVResults.csv')
cv_results_df


# In[9]:

plt.figure()
rmse=xgb_scores.tolist()
plt.errorbar(np.arange(1,k_fold+1,1), rmse)
plt.xlabel('CV Fold')
plt.ylabel('RMSE score')
plt.axis('tight')
plt.show()


# In[10]:

# Fit model and save output results
xgb.fit(X_train,Y_train)
y_pred= xgb.predict(X_test)
output = pd.DataFrame(y_pred)
output = output.set_index(X_test.index)
output.to_csv('output/XGBoostPredictions.csv')


# In[11]:

# Save notebook as py 
get_ipython().system(u'jupyter nbconvert --to=python XGBoost.ipynb')


# In[ ]:



=======

# coding: utf-8

# In[1]:

# Import necessary libraries
import pandas as pd
import numpy as np
import Utils as f
import time

import xgboost
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import eval_metrics as m


# In[2]:

# load cleaned training data
X_train = f.load_obj('input_clean/X_train')
X_test = f.load_obj('input_clean/X_test')
Y_train = f.load_obj('input_clean/Y_train')


# In[3]:

# Evaluate all metrics
store = []
def metric_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    exp_var,mean_abs_err,mean_sq_err,rms,r2_sc = m.metrics_regress(y, y_pred)
    global store
    store.append(m.metrics_regress(y, y_pred))
    return rms


# In[4]:

# Optimal parameters
param_df = pd.DataFrame()
param_df = param_df.from_csv('output/XGBoostOptimization.csv')
opt_params = param_df.iloc[0][3:]
opt_params_df = pd.DataFrame(opt_params).T
opt_params_df


# In[5]:

# 10 fold CV with XGBoost

# Best parameters already found by tuning
xgb = xgboost.XGBRegressor(gamma=opt_params_df.gamma[0],
                           colsample_bytree=opt_params_df.colsample_bytree[0],
                           colsample_bylevel=opt_params_df.colsample_bylevel[0],
                           learning_rate=opt_params_df.learning_rate[0],
                           max_depth=int(opt_params_df.max_depth[0]),
                           min_child_weight=opt_params_df.min_child_weight[0],
                           subsample=opt_params_df.subsample[0],
                           reg_alpha=opt_params_df.reg_alpha[0],
                           reg_lambda=opt_params_df.reg_lambda[0])

k_fold = KFold(10, random_state=1)

start = time.time()
print('Started 10 Fold cross validation')

xgb_scores = cross_val_score(xgb, X_train, Y_train, cv=k_fold, scoring=metric_scorer, n_jobs=1)

print('Finished Cross validation in {} mins'.format(round(((time.time() - start)/60),2)))


# In[6]:

# CV results
cv_results_df = pd.DataFrame(store)
cv_results_df.columns = ['expl_var','mae','mse','rmse','r2']
cv_results_df = cv_results_df.T
cv_results_df.columns = ['Fold '+str(i+1) for i in cv_results_df.columns]
means = []
stds = []
for i in range(0,5):
    means.append(np.mean(cv_results_df.iloc[i].values))
    stds.append(np.std(cv_results_df.iloc[i].values))
cv_results_df['mean'] = means
cv_results_df['std'] = stds
cv_results_df.to_csv('output/XGBoostCVResults.csv')
cv_results_df


# In[9]:

plt.figure()
rmse=xgb_scores.tolist()
plt.errorbar(np.arange(1,k_fold+1,1), rmse)
plt.xlabel('CV Fold')
plt.ylabel('RMSE score')
plt.axis('tight')
plt.show()


# In[10]:

# Fit model and save output results
xgb.fit(X_train,Y_train)
y_pred= xgb.predict(X_test)
output = pd.DataFrame(y_pred)
output = output.set_index(X_test.index)
output.to_csv('output/XGBoostPredictions.csv')


# In[11]:

# Save notebook as py 
get_ipython().system(u'jupyter nbconvert --to=python XGBoost.ipynb')


# In[ ]:



>>>>>>> b87fc3811e42ab51738dd141760a7221e7902d47
