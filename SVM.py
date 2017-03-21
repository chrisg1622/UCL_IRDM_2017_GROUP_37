from sklearn import svm
import pickle as p
import pandas as pd
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn import metrics
from numpy import sqrt
import numpy as np
import metrics as m

def post_process_preds(y_pred):
    return np.array([max(min(y[0],3.0),1.0) for y in y_pred])

def load_data():
    #Loads data and outputs teaining and test sets
    print("Loading dataset...")
    try:
        train_x = p.load(open('input_clean/X_train.pkl', 'rb'))
        train_y = p.load(open('input_clean/Y_train.pkl', 'rb'))
        test_x = p.load(open('input_clean/X_test.pkl', 'rb'))
    except:
        print("Loading failed")
        return
    print("Dataset successfully loaded!")
    return train_x, train_y, test_x

def score_rmse(model,x,y):
    pre_predicts = model.predict(x)
    predicts = post_process_preds(pre_predicts)
    mean_sq_err = metrics.mean_squared_error(y, predicts)
    rmse = sqrt(mean_sq_err)
    return rmse

def predict_to_csv(model,x_test):
    predicts = model.predict(x_test)
    df = pd.DataFrame(predicts, index=x_test.index)
    df.to_csv('output/SVMPredictions.csv')

    return

def cross_validation(x,y,model,n_folds=10): # Not finished
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(x_train,y_train)
        preditcs = model.predict(x_test)





def random_grid_search(x, y,n_iter = 10,kernel = 'rbf', n_splits_cv = 3):

    #Performs a Randomized Grid Search over the parameters defined in 'parameters' variable.

    model = svm.SVR()
    print("Starting grid search with {} runs".format(n_iter))
    parameters_all = [{'kernel': 'rbf', 'gamma': [0.1, 0.5, 'auto', 1], 'C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50,100]},  #Parameters for a complete GridSearch
                  {'kernel': 'linear', 'C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100]},
                  {'kernel': 'poly', 'degree': [2, 3, 5], 'C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100]}
                  ]
    parameters_linear = {'kernel': ['linear'], 'C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100]} # RandomizedSearchCV doesn't allow for lists of dictionaries
    parameters_rbf = {'kernel': ['rbf'], 'gamma': [0.1, 0.5, 'auto', 1], 'C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50,100]}
    parameters_poly = {'kernel': ['poly'], 'degree': [2, 3, 5], 'C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100]}

    if kernel == 'rbf':
        parameters = parameters_rbf
    elif kernel == 'linear':
        parameters = parameters_linear
    elif kernel == 'poly':
        parameters = parameters_poly

    else:
        print("Kernel not valid")
        return

    kf = KFold(n_splits=n_splits_cv, shuffle=True, random_state=1)
    grid_search = RandomizedSearchCV(model, param_distributions=parameters, verbose=3,n_iter=n_iter, cv=kf, scoring=score_rmse)
    grid_search.fit(x, y.values.ravel())
    print("Grid Search finished")
    return grid_search


def main():
    train_x, train_y, test_x = load_data()
    results = random_grid_search(train_x, train_y, kernel='rbf',n_iter = 10)
    df_results = pd.DataFrame(results)

    print(df_results)
    df_results.to_csv("output/SVMOptimization.csv")

    return

if __name__ == "__main__":
    main()




