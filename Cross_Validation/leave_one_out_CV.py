import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, recall_score, roc_curve, f1_score
from sklearn.preprocessing import MinMaxScaler
from itertools import product

 
def expand_grid(dictionary):

    return pd.DataFrame([row for row in product(*dictionary.values())], columns = dictionary.keys())

 
def leave_one_out_CV(X, Y, data_to_score, model):

    ## This function applies leave-one-out cross validation on a couple of common 
    ## classifiers using the a grid of hyper-parameters. It retuns probabilities
    ## for each combination of hyper-parameters..
    
    ## X: denotes the input variables
    ## Y: denotes the target variable in the test dataset (Y is expected to be a binary variable)
    ## data_to_score: denotes the dataset to be scored
    ## model: model to be considered

    if (model == 'RF'):
        
    
    if (model == 'Ada'):
        
    
    if (model == 'GB'):
        
        
    if (model = 'svm'):

    ###############################################
    ## Defining hyer-parameters to be considered ##
    ###############################################

    ## Number of trees in random forest
    n_estimators = [500, 1000, 2000]

    ## Number of features to consider at every split
    max_features = [3, 5]

    ## Maximum number of levels in tree
    max_depth = [5, 7]

    ## Minimum number of samples required to split a node
    min_samples_split = [10, 15]

    ## Minimum number of samples required at each leaf node
    min_samples_leaf = [5, 7]

    ## Creating the dictionary of hyper-parameters
    param_grid = {'n_estimators': n_estimators,
                  'max_features': max_features,
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf}

    param_grid = expand_grid(param_grid)

    ## Adding accuracy and recall columns
    param_grid['cutoff'] = np.nan
    param_grid['accuracy'] = np.nan
    param_grid['recall'] = np.nan
    param_grid['f1'] = np.nan

    for i in range(param_grid.shape[0]):

        ## Applying leave-one-out cross-validation
        RF_md = RF_leave_one_out_CV_help(X, Y,
                                         n_estimators = param_grid['n_estimators'][i],
                                         max_features = param_grid['max_features'][i],
                                         max_depth = param_grid['max_depth'][i],
                                         min_samples_split = param_grid['min_samples_split'][i],
                                         min_samples_leaf = param_grid['min_samples_leaf'][i])

        ## Computing performance measures
        param_grid.iloc[i, 5] = RF_md[0]
        param_grid.iloc[i, 6] = RF_md[1]
        param_grid.iloc[i, 7] = RF_md[2]
        param_grid.iloc[i, 8] = RF_md[3]

    param_grid = param_grid.sort_values(by = ['f1', 'recall'], ascending = [False, False])

    ## Fitting the best model
    RF_md = RandomForestClassifier(n_estimators = param_grid['n_estimators'][0],
                                   max_features = param_grid['max_features'][0],
                                   max_depth = param_grid['max_depth'][0],
                                   min_samples_split = param_grid['min_samples_split'][0],
                                   min_samples_leaf = param_grid['min_samples_leaf'][0])

    RF_md.fit(X, Y)

    ## Predicting on the test, baseline, S1, and S2 datasets
    X = RF_md.predict_proba(X_test)[:, 1]
    baseline_preds = RF_md.predict_proba(baseline)[:, 1]
    S1_preds = RF_md.predict_proba(S1)[:, 1]
    S2_preds = RF_md.predict_proba(S2)[:, 1]
    S4_preds = RF_md.predict_proba(S4)[:, 1]

    return [X, baseline_preds, S1_preds, S2_preds, S4_preds]

 
def leave_one_out_CV_help(X, Y, model):

    ## Creating loocv procedure
    cv = LeaveOneOut()

    ## Declaring lists to store results
    y_true, y_pred = list(), list()

    for train_ix, test_ix in cv.split(X):

        ## Splitting the data
        X_train, X_test = X.loc[train_ix, :], X.loc[test_ix, :]
        Y_train, Y = Y[train_ix], Y[test_ix]

        ## Fitting the model
        RF_md = RandomForestClassifier(n_estimators = n_estimators,
                                       max_features = max_features,
                                       max_depth = max_depth,
                                       min_samples_split = min_samples_split,
                                       min_samples_leaf = min_samples_leaf)

        RF_md.fit(X_train, Y_train)

        ## Predicting on the test data
        yhat = RF_md.predict_proba(X_test)[:, 1]

        ## Storing results
        y_true.append(Y)
        y_pred.append(yhat)

    ## Extracting False-Positive, True-Positive and optimal cutoff
    False_Positive_Rate, True_Positive_Rate, cutoff = roc_curve(y_true, y_pred)

    ## Finding optimal cutoff (the one that maximizes True-Positive and minimizes False-Positive)
    to_select = np.argmax(True_Positive_Rate - False_Positive_Rate)
    opt_cutoff = cutoff[to_select]

    ## Changing to 0-1
    Y_hat = np.where(y_pred <= opt_cutoff, 0, 1)

    ## Computing performance measures
    accuracy = accuracy_score(y_true, Y_hat)
    recall = recall_score(y_true, Y_hat, average = 'macro')
    f1 = f1_score(y_true, Y_hat)

    return [opt_cutoff, accuracy, recall, f1]