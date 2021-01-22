###########################
## Classification Models ##
###########################


import pandas as pd

import numpy as np

from sklearn.model_selection import StratifiedKFold

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, recall_score, roc_curve

from itertools import product


## Defining the grid of hyper-parameters


def expand_grid(dictionary):
    return pd.DataFrame([row for row in product(*dictionary.values())], columns=dictionary.keys())


#################################################################


## Cross-Validation to select best hyper-parameter combination ##


#################################################################


def CV_Classifier(X, Y, CV, numb_folds, model):
    ## This function performs cross-validation (given CV and numb_folds)

    ## with the goal of selection best hyper-parameter combination

    ## X: denotes the input variables

    ## Y: denotes the target variable

    ## Defining list to store results

    results = []

    for i in range(CV):
        ## Appending results

        results.append(CV_Classifier_help(X, Y, numb_folds, model))

    return pd.concat(results).reset_index(drop=True)


def CV_Classifier_help(X, Y, numb_folds, model):
    ## This function performs cross-validation (given numb_folds)

    ## with the goal of selection best hyper-parameter combination

    ## X: denotes the input variables

    ## Y: denotes the target variable

    ## Defining list to store results

    results = []

    ## Defining the stratified folds

    skf = StratifiedKFold(n_splits=numb_folds, shuffle=True, random_state=None)

    ## Looping through the indexes

    for train_index, test_index in skf.split(X, Y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]

        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        ## Applying the random forest classifier and appending results

        results.append(Classifier(X_train, Y_train, X_test, Y_train, model))

    return pd.concat(results).reset_index(drop=True)


def Classifier(X_train, Y_train, X_test, Y_test, model):
    ## This function applies a classification model using

    ## the a grid of hyper-parameters. It retuns probabilities

    ## for each combination of hyper-parameters for the give model.

    ## X_train: denotes the input variables in the train dataset

    ## Y_train: denotes the target variable in the train dataset (Y is expected to be a binary variable)

    ## X_test: denotes the input variables in the test dataset

    ## Y_test: denotes the target variable in the test dataset (Y is expected to be a binary variable)

    ## model: model to be considered

    ###################

    ## Random Forest ##

    ###################

    if (model == 'RF'):

        ###############################################

        ## Defining hyer-parameters to be considered ##

        ###############################################

        ## Number of trees in random forest

        n_estimators = [100, 500, 1000, 1500, 2000]

        ## Number of features to consider at every split

        max_features = [3, 5, 7]

        ## Maximum number of levels in tree

        max_depth = [5, 7, 10]

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

        param_grid['accuracy'] = np.nan

        param_grid['recall'] = np.nan

        for i in range(param_grid.shape[0]):
            print(i)

            ## Fitting the model (using the ith combination of hyper-parameters)

            RF_md = RandomForestClassifier(n_estimators=param_grid['n_estimators'][i],

                                           max_features=param_grid['max_features'][i],

                                           max_depth=param_grid['max_depth'][i],

                                           min_samples_split=param_grid['min_samples_split'][i],

                                           min_samples_leaf=param_grid['min_samples_leaf'][i])

            RF_md.fit(X_train, Y_train)

            ## Predicting on the test dataset

            preds = RF_md.predict_proba(X_test)[:, 1]

            ## Extracting False-Positive, True-Positive and optimal cutoff

            False_Positive_Rate, True_Positive_Rate, cutoff = roc_curve(Y_test, preds)

            ## Finding optimal cutoff (the one that maximizes True-Positive and minimizes False-Positive)

            to_select = np.argmax(True_Positive_Rate - False_Positive_Rate)

            opt_cutoff = cutoff[to_select]

            ## Changing to 0-1

            Y_hat = np.where(preds <= opt_cutoff, 0, 1)

            ## Computing accuracy and recall

            param_grid.iloc[i, 5] = opt_cutoff

            param_grid.iloc[i, 6] = accuracy_score(Y_test, Y_hat)

            param_grid.iloc[i, 7] = recall_score(Y_test, Y_hat, average='macro')

        return param_grid

    ##############

    ## AdaBoost ##

    ##############

    if (model == 'Ada'):

        ###############################################

        ## Defining hyer-parameters to be considered ##

        ###############################################

        ## Number of trees

        n_estimators = [100, 500, 1000, 1500, 2000]

        ## Maximum number of levels in tree

        max_depth = [5, 7, 10]

        ## Learning rate

        learning_rate = [0.001, 0.01, 0.1, 0.5, 1]

        ## Creating the dictionary of hyper-parameters

        param_grid = {'n_estimators': n_estimators,

                      'max_depth': max_depth,

                      'learning_rate': learning_rate}

        param_grid = expand_grid(param_grid)

        ## Adding accuracy and recall columns

        param_grid['accuracy'] = np.nan

        param_grid['recall'] = np.nan

        for i in range(param_grid.shape[0]):
            ## Fitting the model (using the ith combination of hyper-parameters)

            Ada_md = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=param_grid['max_depth'][i]),

                                        n_estimators=param_grid['n_estimators'][i],

                                        learning_rate=param_grid['learning_rate'][i])

            Ada_md.fit(X_train, Y_train)

            ## Predicting on the test dataset

            preds = Ada_md.predict_proba(X_test)[:, 1]

            ## Extracting False-Positive, True-Positive and optimal cutoff

            False_Positive_Rate, True_Positive_Rate, cutoff = roc_curve(Y_test, preds)

            ## Finding optimal cutoff (the one that maximizes True-Positive and minimizes False-Positive)

            to_select = np.argmax(True_Positive_Rate - False_Positive_Rate)

            opt_cutoff = cutoff[to_select]

            ## Changing to 0-1

            Y_hat = np.where(preds <= opt_cutoff, 0, 1)

            ## Computing accuracy and recall

            param_grid.iloc[i, 3] = opt_cutoff

            param_grid.iloc[i, 4] = accuracy_score(Y_test, Y_hat)

            param_grid.iloc[i, 5] = recall_score(Y_test, Y_hat, average='macro')

        return param_grid

    #######################

    ## Gradient Boosting ##

    #######################

    if (model == 'GB'):

        ###############################################

        ## Defining hyer-parameters to be considered ##

        ###############################################

        ## Number of trees

        n_estimators = [100, 500, 1000, 1500, 2000]

        ## Learning rate

        learning_rate = [0.001, 0.01, 0.1, 0.5, 1]

        ## Number of features to consider at every split

        max_features = [3, 5, 7]

        ## Maximum number of levels in tree

        max_depth = [5, 7, 10]

        ## Minimum number of samples required to split a node

        min_samples_split = [10, 15]

        ## Minimum number of samples required at each leaf node

        min_samples_leaf = [5, 7]

        ## Creating the dictionary of hyper-parameters

        param_grid = {'n_estimators': n_estimators,

                      'learning_rate': learning_rate,

                      'max_features': max_features,

                      'max_depth': max_depth,

                      'min_samples_split': min_samples_split,

                      'min_samples_leaf': min_samples_leaf}

        param_grid = expand_grid(param_grid)

        ## Adding accuracy and recall columns

        param_grid['accuracy'] = np.nan

        param_grid['recall'] = np.nan

        for i in range(param_grid.shape[0]):
            ## Fitting the model (using the ith combination of hyper-parameters)

            GB_md = GradientBoostingClassifier(n_estimators=param_grid['n_estimators'][i],

                                               learning_rate=param_grid['learning_rate'][i],

                                               max_features=param_grid['max_features'][i],

                                               max_depth=param_grid['max_depth'][i],

                                               min_samples_split=param_grid['min_samples_split'][i],

                                               min_samples_leaf=param_grid['min_samples_leaf'][i])

            GB_md.fit(X_train, Y_train)

            ## Predicting on the test dataset

            preds = GB_md.predict_proba(X_test)[:, 1]

            ## Extracting False-Positive, True-Positive and optimal cutoff

            False_Positive_Rate, True_Positive_Rate, cutoff = roc_curve(Y_test, preds)

            ## Finding optimal cutoff (the one that maximizes True-Positive and minimizes False-Positive)

            to_select = np.argmax(True_Positive_Rate - False_Positive_Rate)

            opt_cutoff = cutoff[to_select]

            ## Changing to 0-1

            Y_hat = np.where(preds <= opt_cutoff, 0, 1)

            ## Computing accuracy and recall

            param_grid.iloc[i, 6] = opt_cutoff

            param_grid.iloc[i, 7] = accuracy_score(Y_test, Y_hat)

            param_grid.iloc[i, 8] = recall_score(Y_test, Y_hat, average='macro')

        return param_grid

    #########

    ## SVM ##

    #########

    if (model == 'svm'):

        ###############################################

        ## Defining hyer-parameters to be considered ##

        ###############################################

        ## Kernel

        kernel = ['rbf', 'poly', 'sigmoid']

        ## Regularization parameter

        C = [0.01, 0.1, 1, 10]

        ## Gamma

        gamma = [0.001, 0.01, 0.1, 1]

        ## Creating the dictionary of hyper-parameters

        param_grid = {'kernel': kernel,

                      'C': C,

                      'gamma': gamma}

        param_grid = expand_grid(param_grid)

        ## Adding accuracy and recall columns

        param_grid['accuracy'] = np.nan

        param_grid['recall'] = np.nan

        for i in range(param_grid.shape[0]):
            print(i)

            ## Fitting the model (using the ith combination of hyper-parameters)

            SVM_md = SVC(kernel=param_grid['kernel'][i],

                         C=param_grid['C'][i],

                         gamma=param_grid['gamma'][i],

                         probability=True)

            SVM_md.fit(X_train, Y_train)

            ## Predicting on the test dataset

            preds = SVM_md.predict_proba(X_test)[:, 1]

            ## Extracting False-Positive, True-Positive and optimal cutoff

            False_Positive_Rate, True_Positive_Rate, cutoff = roc_curve(Y_test, preds)

            ## Finding optimal cutoff (the one that maximizes True-Positive and minimizes False-Positive)

            to_select = np.argmax(True_Positive_Rate - False_Positive_Rate)
            ## Classification Models ##

            ###########################

            import pandas as pd

            import numpy as np

            from sklearn.model_selection import StratifiedKFold

            from sklearn.tree import DecisionTreeClassifier

            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

            from sklearn.svm import SVC

            from sklearn.metrics import accuracy_score, recall_score, roc_curve

            from itertools import product

            ## Defining the grid of hyper-parameters

            def expand_grid(dictionary):

                return pd.DataFrame([row for row in product(*dictionary.values())], columns=dictionary.keys())

            #################################################################

            ## Cross-Validation to select best hyper-parameter combination ##

            #################################################################

            def CV_Classifier(X, Y, CV, numb_folds, model):

                ## This function performs cross-validation (given CV and numb_folds)

                ## with the goal of selection best hyper-parameter combination

                ## X: denotes the input variables

                ## Y: denotes the target variable

                ## Defining list to store results

                results = []

                for i in range(CV):
                    ## Appending results

                    results.append(CV_Classifier_help(X, Y, numb_folds, model))

                return pd.concat(results).reset_index(drop=True)

            def CV_Classifier_help(X, Y, numb_folds, model):

                ## This function performs cross-validation (given numb_folds)

                ## with the goal of selection best hyper-parameter combination

                ## X: denotes the input variables

                ## Y: denotes the target variable

                ## Defining list to store results

                results = []

                ## Defining the stratified folds

                skf = StratifiedKFold(n_splits=numb_folds, shuffle=True, random_state=None)

                ## Looping through the indexes

                for train_index, test_index in skf.split(X, Y):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

                    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

                    ## Applying the random forest classifier and appending results

                    results.append(Classifier(X_train, Y_train, X_test, Y_train, model))

                return pd.concat(results).reset_index(drop=True)

            def Classifier(X_train, Y_train, X_test, Y_test, model):

                ## This function applies a classification model using

                ## the a grid of hyper-parameters. It retuns probabilities

                ## for each combination of hyper-parameters for the give model.

                ## X_train: denotes the input variables in the train dataset

                ## Y_train: denotes the target variable in the train dataset (Y is expected to be a binary variable)

                ## X_test: denotes the input variables in the test dataset

                ## Y_test: denotes the target variable in the test dataset (Y is expected to be a binary variable)

                ## model: model to be considered

                ###################

                ## Random Forest ##

                ###################

                if (model == 'RF'):

                    ###############################################

                    ## Defining hyer-parameters to be considered ##

                    ###############################################

                    ## Number of trees in random forest

                    n_estimators = [100, 500, 1000, 1500, 2000]

                    ## Number of features to consider at every split

                    max_features = [3, 5, 7]

                    ## Maximum number of levels in tree

                    max_depth = [5, 7, 10]

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

                    param_grid['accuracy'] = np.nan

                    param_grid['recall'] = np.nan

                    for i in range(param_grid.shape[0]):
                        print(i)

                        ## Fitting the model (using the ith combination of hyper-parameters)

                        RF_md = RandomForestClassifier(n_estimators=param_grid['n_estimators'][i],

                                                       max_features=param_grid['max_features'][i],

                                                       max_depth=param_grid['max_depth'][i],

                                                       min_samples_split=param_grid['min_samples_split'][i],

                                                       min_samples_leaf=param_grid['min_samples_leaf'][i])

                        RF_md.fit(X_train, Y_train)

                        ## Predicting on the test dataset

                        preds = RF_md.predict_proba(X_test)[:, 1]

                        ## Extracting False-Positive, True-Positive and optimal cutoff

                        False_Positive_Rate, True_Positive_Rate, cutoff = roc_curve(Y_test, preds)

                        ## Finding optimal cutoff (the one that maximizes True-Positive and minimizes False-Positive)

                        to_select = np.argmax(True_Positive_Rate - False_Positive_Rate)

                        opt_cutoff = cutoff[to_select]

                        ## Changing to 0-1

                        Y_hat = np.where(preds <= opt_cutoff, 0, 1)

                        ## Computing accuracy and recall

                        param_grid.iloc[i, 5] = opt_cutoff

                        param_grid.iloc[i, 6] = accuracy_score(Y_test, Y_hat)

                        param_grid.iloc[i, 7] = recall_score(Y_test, Y_hat, average='macro')

                    return param_grid

                ##############

                ## AdaBoost ##

                ##############

                if (model == 'Ada'):

                    ###############################################

                    ## Defining hyer-parameters to be considered ##

                    ###############################################

                    ## Number of trees

                    n_estimators = [100, 500, 1000, 1500, 2000]

                    ## Maximum number of levels in tree

                    max_depth = [5, 7, 10]

                    ## Learning rate

                    learning_rate = [0.001, 0.01, 0.1, 0.5, 1]

                    ## Creating the dictionary of hyper-parameters

                    param_grid = {'n_estimators': n_estimators,

                                  'max_depth': max_depth,

                                  'learning_rate': learning_rate}

                    param_grid = expand_grid(param_grid)

                    ## Adding accuracy and recall columns

                    param_grid['accuracy'] = np.nan

                    param_grid['recall'] = np.nan

                    for i in range(param_grid.shape[0]):
                        ## Fitting the model (using the ith combination of hyper-parameters)

                        Ada_md = AdaBoostClassifier(
                            base_estimator=DecisionTreeClassifier(max_depth=param_grid['max_depth'][i]),

                            n_estimators=param_grid['n_estimators'][i],

                            learning_rate=param_grid['learning_rate'][i])

                        Ada_md.fit(X_train, Y_train)

                        ## Predicting on the test dataset

                        preds = Ada_md.predict_proba(X_test)[:, 1]

                        ## Extracting False-Positive, True-Positive and optimal cutoff

                        False_Positive_Rate, True_Positive_Rate, cutoff = roc_curve(Y_test, preds)

                        ## Finding optimal cutoff (the one that maximizes True-Positive and minimizes False-Positive)

                        to_select = np.argmax(True_Positive_Rate - False_Positive_Rate)

                        opt_cutoff = cutoff[to_select]

                        ## Changing to 0-1

                        Y_hat = np.where(preds <= opt_cutoff, 0, 1)

                        ## Computing accuracy and recall

                        param_grid.iloc[i, 3] = opt_cutoff

                        param_grid.iloc[i, 4] = accuracy_score(Y_test, Y_hat)

                        param_grid.iloc[i, 5] = recall_score(Y_test, Y_hat, average='macro')

                    return param_grid

                #######################

                ## Gradient Boosting ##

                #######################

                if (model == 'GB'):

                    ###############################################

                    ## Defining hyer-parameters to be considered ##

                    ###############################################

                    ## Number of trees

                    n_estimators = [100, 500, 1000, 1500, 2000]

                    ## Learning rate

                    learning_rate = [0.001, 0.01, 0.1, 0.5, 1]

                    ## Number of features to consider at every split

                    max_features = [3, 5, 7]

                    ## Maximum number of levels in tree

                    max_depth = [5, 7, 10]

                    ## Minimum number of samples required to split a node

                    min_samples_split = [10, 15]

                    ## Minimum number of samples required at each leaf node

                    min_samples_leaf = [5, 7]

                    ## Creating the dictionary of hyper-parameters

                    param_grid = {'n_estimators': n_estimators,

                                  'learning_rate': learning_rate,

                                  'max_features': max_features,

                                  'max_depth': max_depth,

                                  'min_samples_split': min_samples_split,

                                  'min_samples_leaf': min_samples_leaf}

                    param_grid = expand_grid(param_grid)

                    ## Adding accuracy and recall columns

                    param_grid['accuracy'] = np.nan

                    param_grid['recall'] = np.nan

                    for i in range(param_grid.shape[0]):
                        ## Fitting the model (using the ith combination of hyper-parameters)

                        GB_md = GradientBoostingClassifier(n_estimators=param_grid['n_estimators'][i],

                                                           learning_rate=param_grid['learning_rate'][i],

                                                           max_features=param_grid['max_features'][i],

                                                           max_depth=param_grid['max_depth'][i],

                                                           min_samples_split=param_grid['min_samples_split'][i],

                                                           min_samples_leaf=param_grid['min_samples_leaf'][i])

                        GB_md.fit(X_train, Y_train)

                        ## Predicting on the test dataset

                        preds = GB_md.predict_proba(X_test)[:, 1]

                        ## Extracting False-Positive, True-Positive and optimal cutoff

                        False_Positive_Rate, True_Positive_Rate, cutoff = roc_curve(Y_test, preds)

                        ## Finding optimal cutoff (the one that maximizes True-Positive and minimizes False-Positive)

                        to_select = np.argmax(True_Positive_Rate - False_Positive_Rate)

                        opt_cutoff = cutoff[to_select]

                        ## Changing to 0-1

                        Y_hat = np.where(preds <= opt_cutoff, 0, 1)

                        ## Computing accuracy and recall

                        param_grid.iloc[i, 6] = opt_cutoff

                        param_grid.iloc[i, 7] = accuracy_score(Y_test, Y_hat)

                        param_grid.iloc[i, 8] = recall_score(Y_test, Y_hat, average='macro')

                    return param_grid

                #########

                ## SVM ##

                #########

                if (model == 'svm'):

                    ###############################################

                    ## Defining hyer-parameters to be considered ##

                    ###############################################

                    ## Kernel

                    kernel = ['rbf', 'poly', 'sigmoid']

                    ## Regularization parameter

                    C = [0.01, 0.1, 1, 10]

                    ## Gamma

                    gamma = [0.001, 0.01, 0.1, 1]

                    ## Creating the dictionary of hyper-parameters

                    param_grid = {'kernel': kernel,

                                  'C': C,

                                  'gamma': gamma}

                    param_grid = expand_grid(param_grid)

                    ## Adding accuracy and recall columns

                    param_grid['accuracy'] = np.nan

                    param_grid['recall'] = np.nan

                    for i in range(param_grid.shape[0]):
                        print(i)

                        ## Fitting the model (using the ith combination of hyper-parameters)

                        SVM_md = SVC(kernel=param_grid['kernel'][i],

                                     C=param_grid['C'][i],

                                     gamma=param_grid['gamma'][i],

                                     probability=True)

                        SVM_md.fit(X_train, Y_train)

                        ## Predicting on the test dataset

                        preds = SVM_md.predict_proba(X_test)[:, 1]

                        ## Extracting False-Positive, True-Positive and optimal cutoff

                        False_Positive_Rate, True_Positive_Rate, cutoff = roc_curve(Y_test, preds)

                        ## Finding optimal cutoff (the one that maximizes True-Positive and minimizes False-Positive)

                        to_select = np.argmax(True_Positive_Rate - False_Positive_Rate)

                        opt_cutoff = cutoff[to_select]

                        ## Changing to 0-1

                        Y_hat = np.where(preds <= opt_cutoff, 0, 1)

                        ## Computing accuracy and recall

                        param_grid.iloc[i, 3] = opt_cutoff

                        param_grid.iloc[i, 4] = accuracy_score(Y_test, Y_hat)

                        param_grid.iloc[i, 5] = recall_score(Y_test, Y_hat, average='macro')

                    return param_grid
            opt_cutoff = cutoff[to_select]

            ## Changing to 0-1

            Y_hat = np.where(preds <= opt_cutoff, 0, 1)

            ## Computing accuracy and recall

            param_grid.iloc[i, 3] = opt_cutoff

            param_grid.iloc[i, 4] = accuracy_score(Y_test, Y_hat)

            param_grid.iloc[i, 5] = recall_score(Y_test, Y_hat, average='macro')

        return param_grid