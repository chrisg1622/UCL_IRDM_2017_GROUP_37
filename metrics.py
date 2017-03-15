def metrics_regress(predicts,ground_truth):
    #Explained variance score
    exp_var = metrics.explained_variance_score(predicts,ground_truth) # 0 to 1, the higher the better
    #Mean Absolute Error
    mean_abs_err = metrics.mean_absolute_error(predicts,ground_truth) # the lower the better
    #Mean Squared Error
    mean_sq_err = metrics.mean_squared_error(predicts,ground_truth)
    #Root Mean Squared Error
    rms = sqrt(mean_sq_err)
    #R2 coefficient
    r2_sc = metrics.r2_score(predicts,ground_truth)
    
    return exp_var,mean_abs_err,mean_sq_err,rms,r2_sc
    
    

def evaluate(predicts,ground_truth):
    exp_var,mean_abs_err,mean_sq_err,rms,r2_sc = metrics_regress(predicts,ground_truth)
    print("Explained variance score: {}".format(exp_var))
    print("Mean Absolute Error: {}".format(mean_abs_err))
    print("Root Mean Square Error: {}".format(rms))
    print("Coefficient of determination (R2): {}".format(r2_sc))
    