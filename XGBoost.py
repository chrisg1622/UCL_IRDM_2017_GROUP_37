
# coding: utf-8

# In[1]:

# Import necessary libraries
import pandas as pd
import numpy as np
import Utils as f
import time

import xgboost
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# In[2]:

# load cleaned training data
X_train = f.load_obj('input_clean/X_train')
X_test = f.load_obj('input_clean/X_test')
Y_train = f.load_obj('input_clean/Y_train')


# In[3]:

# 10 fold CV with XGBoost
xgb = xgboost.XGBRegressor()
k_fold = KFold(10)

start = time.time()
print('Started 10 Fold cross validation')

xgb_scores = cross_val_score(xgb, X_train, Y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=1)
rmse = np.sqrt(abs(xgb_scores))

print('Finished Cross validation in {} mins'.format(round(((time.time() - start)/60),2)))


# In[4]:

# CV results
print('CV rmse scores:',rmse.tolist())
print('CV rmse mean: {:4f}'.format(rmse.mean()))
print('CV rmse standard deviation: {:4f}'.format(rmse.std()))

plt.figure()
plt.errorbar(np.arange(1,k_fold.get_n_splits()+1,1), rmse.tolist())
plt.xlabel('K in KFold')
plt.ylabel('RMSE score')
plt.axis('tight')
plt.show()


# In[5]:

# Fit model and save output results
xgb.fit(X_train,Y_train)
y_pred= xgb.predict(X_test)
output = pd.DataFrame(y_pred)
output = output.set_index(X_test.index)
output.to_csv('output/XGBoost.csv')


# In[6]:

# Save notebook as py 
get_ipython().system(u'jupyter nbconvert --to=python XGBoost.ipynb')


# In[ ]:



