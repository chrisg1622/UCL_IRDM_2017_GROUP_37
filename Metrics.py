def evaluate(predicts,ground_truth):
    exp_var,mean_abs_err,mean_sq_err,rms,r2_sc = metrics_regress(predicts,ground_truth)
    print("Explained variance score: {}".format(exp_var))
    print("Mean Absolute Error: {}".format(mean_abs_err))
    print("Root Mean Square Error: {}".format(rms))
    print("Coefficient of determination (R2): {}".format(r2_sc))
    
    

def evaluate(predicts,ground_truth):
    exp_var,mean_abs_err,mean_sq_err,rms,r2_sc = metrics_regress(predicts,ground_truth)
    print("Explained variance score: {}".format(exp_var))
    print("Mean Absolute Error: {}".format(mean_abs_err))
    print("Root Mean Square Error: {}".format(rms))
    print("Coefficient of determination (R2): {}".format(r2_sc))
    