<<<<<<< HEAD

# coding: utf-8

# In[20]:

# Import necessary libraries
import pandas as pd
import numpy as np
import Utils as f
import time

from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import eval_metrics as m
from sklearn.linear_model import Ridge


# In[21]:

# load cleaned training data
X_train = f.load_obj('input_clean/X_train')
X_test = f.load_obj('input_clean/X_test')
Y_train = f.load_obj('input_clean/Y_train')


# In[22]:

#function to cap predictions on a scale of 1 to 3
def post_process_preds(y_pred):
    return np.array([max(min(y,3.0),1.0) for y in y_pred])


# In[23]:

# Evaluate all metrics
store = []

def metric_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    y_ppred = post_process_preds(y_pred)
    exp_var,mean_abs_err,mean_sq_err,rms,r2_sc = m.metrics_regress(y, y_pred)
    global store
    store.append(m.metrics_regress(y, y_ppred))
    return rms


# In[24]:

# Optimal parameters
param_df = pd.DataFrame()
param_df = param_df.from_csv('output/RidgeOptimization.csv')
opt_params = param_df.iloc[0][3:]
opt_params_df = pd.DataFrame(opt_params).T
opt_params_df


# In[25]:

ridge = Ridge(alpha=opt_params_df.alpha[0],
             fit_intercept=opt_params_df.fit_intercept[0],
             normalize=opt_params_df.normalize[0],
             solver=opt_params_df.solver[0],
             tol=opt_params_df.tol[0])

k_fold = KFold(10, random_state=1)

start = time.time()
print('Started 10 Fold cross validation')

ridge_scores = cross_val_score(ridge, X_train, Y_train, cv=k_fold, scoring=metric_scorer, n_jobs=1)

print('Finished Cross validation in {} mins'.format(round(((time.time() - start)/60),2)))


# In[26]:

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
cv_results_df.to_csv('output/RidgeCVResults.csv')
cv_results_df


# In[ ]:

# Fit model and save output results
ridge.fit(X_train,Y_train)
y_pred= ridge.predict(X_test)
y_ppred = post_process_preds(y_pred)
output = pd.DataFrame(y_ppred)
output = output.set_index(X_test.index)
output.columns = ['relevance']
output.to_csv('output/RidgePredictions.csv')


# In[ ]:

# Save notebook as py 
get_ipython().system(u'jupyter nbconvert --to=python Ridge.ipynb')


# In[ ]:



=======

# coding: utf-8

# In[2]:

# Import necessary libraries
import pandas as pd
import numpy as np
import Utils as f
import time

from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import eval_metrics as m
from sklearn.linear_model import Ridge


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
param_df = param_df.from_csv('output/RidgeOptimization.csv')
opt_params = param_df.iloc[0][3:]
opt_params_df = pd.DataFrame(opt_params).T
opt_params_df


# In[5]:

ridge = Ridge(alpha=opt_params_df.alpha[0],
             fit_intercept=opt_params_df.fit_intercept[0],
             normalize=opt_params_df.normalize[0],
             solver=opt_params_df.solver[0],
             tol=opt_params_df.tol[0])

k_fold = KFold(10, random_state=1)

start = time.time()
print('Started 10 Fold cross validation')

ridge_scores = cross_val_score(ridge, X_train, Y_train, cv=k_fold, scoring=metric_scorer, n_jobs=1)

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
cv_results_df.to_csv('output/RidgeCVResults.csv')
cv_results_df


# In[7]:

# Fit model and save output results
ridge.fit(X_train,Y_train)
y_pred= ridge.predict(X_test)
output = pd.DataFrame(y_pred)
output = output.set_index(X_test.index)
output.to_csv('output/RidgePredictions.csv')


# In[ ]:

# Save notebook as py 
get_ipython().system(u'jupyter nbconvert --to=python Ridge.ipynb')

>>>>>>> b87fc3811e42ab51738dd141760a7221e7902d47
