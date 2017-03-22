
# coding: utf-8

# In[5]:

# Import necessary libraries
import pandas as pd
import numpy as np
import Utils as f
import time

from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import eval_metrics as m
from sklearn.linear_model import ElasticNet


# In[6]:

# load cleaned training data
X_train = f.load_obj('input_clean/X_train')
X_test = f.load_obj('input_clean/X_test')
Y_train = f.load_obj('input_clean/Y_train')


# In[7]:

# Evaluate all metrics
store = []

def metric_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    exp_var,mean_abs_err,mean_sq_err,rms,r2_sc = m.metrics_regress(y, y_pred)
    global store
    store.append(m.metrics_regress(y, y_pred))
    return rms


# In[29]:

# Optimal parameters
param_df = pd.DataFrame()
param_df = param_df.from_csv('output/ElasticNetOptimization.csv')
opt_params = param_df.iloc[0][3:]
opt_params_df = pd.DataFrame(opt_params).T
opt_params_df


# In[ ]:

en = ElasticNet(alpha=opt_params_df.alpha[0],
             fit_intercept=opt_params_df.fit_intercept [0],
             normalize=opt_params_df.normalize[0],
             l1_ratio=opt_params_df.l1_ratio[0],
             tol=opt_params_df.tol[0])

k_fold = KFold(10, random_state=1)

start = time.time()
print('Started 10 Fold cross validation')

en_scores = cross_val_score(en, X_train, Y_train, cv=k_fold, scoring=metric_scorer, n_jobs=1)

print('Finished Cross validation in {} mins'.format(round(((time.time() - start)/60),2)))


# In[8]:

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
cv_results_df.to_csv('output/ElasticNetCVResults.csv')
cv_results_df


# In[9]:

# Fit model and save output results
en.fit(X_train,Y_train)
y_pred= en.predict(X_test)
output = pd.DataFrame(y_pred)
output = output.set_index(X_test.index)
output.to_csv('output/ElasticNetPredictions.csv')


# In[10]:

# Save notebook as py 
get_ipython().system(u'jupyter nbconvert --to=python ElasticNet.ipynb')





