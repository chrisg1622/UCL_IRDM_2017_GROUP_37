# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 21:50:13 2017

@author: Chris
"""
import pandas as pd
import numpy as np
import Utils as f
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.rcParams['xtick.labelsize'] = 15
matplotlib.rcParams['ytick.labelsize'] = 15
matplotlib.rcParams['axes.titlesize'] = 15
matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['axes.labelsize'] = 17

print('Attempting to load pre-processed data...')
try:
    #load pre-processed training data
    X_train = f.load_obj('input_clean/X_train')
    X_test = f.load_obj('input_clean/X_test')
    Y_train = f.load_obj('input_clean/Y_train')
    print('Pre-processed data successfully loaded...')
except:
    print('Pre-processed data failed to load, ensure the working directory is correct...')

#%%

#split the data by relevance value and view the difference in means for each feature
#retrieve feature importance scores, calculated by 
scores = []
for col in X_train.columns:
    less_rel_score = X_train.loc[Y_train['relevance']<2.0][col].describe()['mean']
    more_rel_score = X_train.loc[Y_train['relevance']>=2.0][col].describe()['mean']
    print('Feature: {} \n Mean for relevance >= 2.0: {} \n Mean for relevance < 2.0 {}'.format(col,
          more_rel_score,
          less_rel_score))
    scores.append(abs(more_rel_score - less_rel_score)/min(less_rel_score,more_rel_score))

#%%

#number the attributes from 1 to 68, store these as k,v pairs
feature_importance_scores = [(i+1,scores[i]) for i in range(len(scores))]
#sort the attributes by importance, descending
sorted_feats = sorted(feature_importance_scores, key = lambda x:x[1], reverse=True)
#retrieves sorted keys and values
sorted_keys = [feat[0] for feat in sorted_feats]
sorted_vals = [feat[1] for feat in sorted_feats]
sorted_names = [X_train.columns[i-1] for i in sorted_keys]

#number of top/bottom features to retrive
num_feats = 5
#create figure for plot
fig, (ax1, ax2) = plt.subplots(1,2,sharey=True)
#plot 10 most important features
sns.barplot(x = sorted_keys[:num_feats],y = sorted_vals[:num_feats], order = sorted_keys[:num_feats], ax=ax1)
#plot 10 least important features
sns.barplot(x = sorted_keys[-num_feats:],y = sorted_vals[-num_feats:], order = sorted_keys[-num_feats:], ax=ax2)
ax1.set_title(str(num_feats)+' Most Varying Features')
ax1.set_xlabel('Feature number')
ax1.set_ylabel('Relative difference')
ax2.set_xlabel('Feature number')
ax2.set_ylabel('Relative difference')
ax2.set_title(str(num_feats)+' Least Varying Features')

fig.savefig('output/FeatureImportance_UsingRelativeDifference')

feature_importance = pd.DataFrame({'Feature':sorted_names,'% Importance':sorted_vals})
feature_importance.to_csv('output/FeatureImportance_UsingRelativeDifference.csv')