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
folds = KFold(n_splits=10, shuffle=True, random_state=1)
#lists to store metric values at each fold
rmse_scores = []
exp_var_scores = []
mean_abs_error_scores = []
mean_sq_error_scores = []
r2_scores = []
#lists to store metric values at each fold
rmse_scores_pp = []
exp_var_scores_pp = []
mean_abs_error_scores_pp = []
mean_sq_error_scores_pp = []
r2_scores_pp = []
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
    #evaluate predictions and append score to lists
    exp_var,mean_abs_err,mean_sq_err,rmse,r2_sc = m.metrics_regress(y_pred, y_test)
    #append scores to lists
    rmse_scores.append(rmse)
    exp_var_scores.append(exp_var)
    mean_abs_error_scores.append(mean_abs_err)
    mean_sq_error_scores.append(mean_sq_err)
    r2_scores.append(r2_sc)
    
    #rescale predictions
    y_pred_pp = post_process_preds(y_pred)
    #evaluate predictions and append score to lists
    exp_var_pp,mean_abs_err_pp,mean_sq_err_pp,rmse_pp,r2_sc_pp = m.metrics_regress(y_pred_pp, y_test)
    #append scores to lists
    rmse_scores_pp.append(rmse_pp)
    exp_var_scores_pp.append(exp_var_pp)
    mean_abs_error_scores_pp.append(mean_abs_err_pp)
    mean_sq_error_scores_pp.append(mean_sq_err_pp)
    r2_scores_pp.append(r2_sc_pp)
    
    print('Fold: {}, \t RMSE: {:4f} \t Post_Processed RMSE: {:4f}'.format(fold, rmse_scores[-1], rmse_scores_pp[-1]))
    fold += 1

#compute mean and standard deviation of each metric, and append these to the results df
CV_results_df2 = pd.DataFrame()
for i,tup in enumerate([(rmse_scores,rmse_scores_pp,'rmse'),
            (exp_var_scores,exp_var_scores_pp,'expl_var'),
            (mean_abs_error_scores,mean_abs_error_scores_pp,'mae'),
            (mean_sq_error_scores,mean_sq_error_scores_pp,'mse'),
            (r2_scores,r2_scores_pp,'R2')]):
    scores, scores_pp, metric = tup        
    #compute mean and std deviation of the scores
    mean = np.mean(scores)
    std = np.std(scores)
    print('10 fold CV '+metric+' mean: {:4f}'.format(mean))
    print('10 fold CV '+metric+' standard deviation: {:4f}'.format(std))
    CV_results_df.loc[2*i,'10-fold mean'] = mean
    CV_results_df.loc[2*i,'10-fold std'] = std
    CV_results_df.loc[2*i,'Post-processed'] = 'No'
    CV_results_df.loc[2*i,'Metric'] = metric

    mean_pp = np.mean(scores_pp)
    std_pp = np.std(scores_pp)
    print('Post-processed predictions, 10 fold CV '+metric+' mean: {:4f}'.format(mean_pp))
    print('Post-processed predictions, 10 fold CV '+metric+' standard deviation: {:4f}'.format(std_pp))
    CV_results_df.loc[2*i + 1,'10-fold mean'] = mean_pp
    CV_results_df.loc[2*i + 1,'10-fold std'] = std_pp
    CV_results_df.loc[2*i + 1,'Post-processed'] = 'Yes'
    CV_results_df.loc[2*i + 1,'Metric'] = metric

    CV_results_df2.loc[i, 'Metric'] = metric
    CV_results_df2.loc[i, '10-Fold Mean Not Post-Processed'] = mean
    CV_results_df2.loc[i, '10-Fold STD Not Post-Processed'] = std
    CV_results_df2.loc[i, '10-Fold Mean Post-Processed'] = mean_pp
    CV_results_df2.loc[i, '10-Fold STD Post-Processed'] = std_pp
    CV_results_df2.loc[i, 'Change in 10-Fold Mean'] = mean_pp - mean

CV_results_df2.to_csv('output/PostProcessingTestResults.csv')



#%%

#plot results and save figure
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.rcParams['xtick.labelsize'] = 15
matplotlib.rcParams['ytick.labelsize'] = 15
matplotlib.rcParams['axes.titlesize'] = 15
matplotlib.rcParams['legend.fontsize'] = 15
matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['axes.labelsize'] = 17


plot = sns.factorplot(x = 'Metric', 
                      y='Change in 10-Fold Mean', 
                      data=CV_results_df2, 
                      kind="bar", 
                      size=4,
                      aspect=2,
                      color='Green')
plot.savefig('output/PostProcessingTestResults.png')

plot = sns.factorplot(x = 'Metric', 
                      y="10-fold mean", 
                      data=CV_results_df, 
                      hue = 'Post-processed',
                      kind="bar", 
                      size=4, 
                      aspect=2,
                      color='Green')
plot.savefig('output/PostProcessingTestResults2.png')




