"""
This module provides wrapper functions for classifying texts with LGBM and explaining them with SHAP, based on features extracted by Spacy.
The modules you might need to run first are 'preprocess_spacy.py' module, to first load and preprocess the texts,
'feature_extraction.py' module, to extract interesting textual features, and later 'text_visualise.py', to plot results and their SHAP explanations.

It includes functions to ...

Author: Jeremi K. Ochab
Date: July 04, 2023
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import pickle
import os

from . import shared_module
# Global control of tqdm progress bar instead of providing it in each function separately.
tqdm_notebook = shared_module.tqdm_notebook
tqdm_display = shared_module.tqdm_display

# from sklearn.model_selection import GridSearchCV
import sklearn
from sklearn.metrics import accuracy_score, fbeta_score, matthews_corrcoef, make_scorer, roc_auc_score, recall_score, top_k_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold,StratifiedKFold,GroupKFold, train_test_split
from sklearn.dummy import DummyClassifier
# Define a mapping of CV methods to their default parameters
cv_method_parameters = {
    'KFold': {'n_splits': 5, 'shuffle': False},
    'GroupKFold': {'n_splits': 10},
    'RepeatedKFold': {'n_splits': 5, 'n_repeats': 2},
    'StratifiedKFold': {'n_splits': 5, 'shuffle': False},
    'RepeatedStratifiedKFold': {'n_splits': 5, 'n_repeats': 2},
    'StratifiedGroupKFold': {'n_splits': 5, 'shuffle': False},
    'ShuffleSplit': {'n_splits': 5,'test_size': 0.2,'train_size': 0.2, 'shuffle': False},
    'GroupShuffleSplit': {'n_splits': 5, 'test_size': 0.2},
    'StratifiedShuffleSplit': {'n_splits': 5,'test_size': 0.2,'train_size': 0.2, 'shuffle': False},
    'LeaveOneOut': {},
    'LeavePOut': {'p': 2},
    'LeaveOneGroupOut': {},
    'LeavePGroupsOut': {'n_groups': 2},
    'PredefinedSplit': {'test_fold': [1, 0, 1, 1, 0]},  # Modify with your own test_fold values
    }

from collections import defaultdict

def target_classes(sample_classes,
                   no_classes=0,
                   class_regrouping = [],
                   verbatim = 0):
    """
    Produces class labels based on provided sample names. 

    Parameters
    ----------
    sample_classes : list
        A list of strings representing class names of each sample.
    no_classes : int, optional
        Number of classes to create (default is 0, which produces the maximal number of class labels).
    class_regrouping : list, optional
        A list of lists representing classes. Each inner list contains all class names to be included in a class (default is an empty list, which produces the maximal number of class labels).
    verbose : int, optional
        Verbosity level (0: all messages, 1: warnings, 2: no messages) (default is 0).

    Returns
    -------
    y : ndarray
        An array of class labels corresponding to each sample class.
    yD : dict
        A dictionary where the keys are the class labels and the values are lists of corresponding class names.
    classD : dict
        A dictionary where the keys are the sample class names and the values are the corresponding class labels.

    Notes
    -----
    - `class_regrouping = [['class1_name'],[]]` results in the second class containing all remaining class names.
    - If `class_regrouping` is set, `no_classes` is ignored.
    
    Raises
    ------
    ValueError
        If `sample_classes` is not a list of strings.
        If `no_classes` is not a positive integer.
        If `class_regrouping` is not a list of lists.
        If any class names in `class_regrouping` do not exist in `sample_classes`.
        
    Examples
    --------
    # Default: distinguish between all available class names
    >>> class_regrouping = [['class1_name1'], [], ['class2_name1','class2_name2']]
    >>> sample_classes = ['class1_name1', 'class1_name1', 'class3_name1', 'class3_name2', 'class2_name1', 'class2_name2', 'class2_name2', 'class2_name2']
    >>> y, yD, classD = target_classes(sample_classes,class_regrouping=[])
    >>> print(y)
    [0 1 2 3 4]
    >>> print(yD)
    {0: ['class1_name1'], 1: ['class2_name2'], 2: ['class3_name1'], 3: ['class2_name1'], 4: ['class3_name2']}
    >>> print(classD)
    {'class1_name1': 0, 'class2_name2': 1, 'class3_name1': 2, 'class2_name1': 3, 'class3_name2': 4}

    # Ignore no_classes; produce three classes; class 1 contains all remaining class names.
    >>> class_regrouping = [['class1_name1'], [], ['class2_name1','class2_name2']]
    >>> sample_classes = ['class1_name1', 'class1_name1', 'class3_name1', 'class3_name2', 'class2_name1', 'class2_name2', 'class2_name2', 'class2_name2']
    >>> y, yD, classD = target_classes(y,no_classes=2,class_regrouping=class_regrouping)
    >>> print(y)
    [0 2 1 2 1]
    >>> print(yD)
    {0: ['class1_name1'], 1: ['class3_name1', 'class3_name2'], 2: ['class2_name1', 'class2_name2']}
    >>> print(classD)
    {'class1_name1': 0, 'class3_name1': 1, 'class3_name2': 1, 'class2_name1': 2, 'class2_name2': 2}
    """
    if not isinstance(sample_classes, list) or not all(isinstance(cl, str) for cl in sample_classes):
        raise ValueError("`sample_classes` must be a list of strings")
    if not isinstance(no_classes, int) or no_classes < 0:
        raise ValueError("`no_classes` must be a positive integer")
    if not isinstance(class_regrouping, list) or not all(isinstance(cl_list, list) for cl_list in class_regrouping):
        raise ValueError("`class_regrouping` must be a list of lists")
    for cl_list in class_regrouping:
        if not all(cl in sample_classes for cl in cl_list):
            raise ValueError("All class names in `class_regrouping` must exist in sample_classes")

    classD={}
    if not class_regrouping:
        class_regrouping = [[cl] for cl in list(set(sample_classes))[0:no_classes-1]]
        class_regrouping.append([])
    class_names_set = set(class_name for class_list in class_regrouping for class_name in class_list)

    # Find the index of the first empty list
    empty_list_index = next((i for i, sublst in enumerate(class_regrouping) if sublst == []), None)
    if empty_list_index is not None:
        # Keep the first empty list and remove the rest
        class_regrouping = [sublst for i, sublst in enumerate(class_regrouping) if sublst != [] or i == empty_list_index]
    elif set(sample_classes) != class_names_set:
        class_regrouping.append([])
        if verbatim < 1:
            print("Not all class names in `sample_classes` are present in `class_regrouping`. Adding a class for the remaining ones.")
    if no_classes != len(class_regrouping):
        if verbatim < 1 and no_classes>0:
            print("The number of classes changed.")    
        no_classes = len(class_regrouping)
    if verbatim < 1:
        print("Proceeding with {} classes.".format(no_classes))

    for l, class_names in enumerate(class_regrouping):
        if not class_names:  # If the list is empty
            class_names = set(sample_classes).difference(class_names_set)
        for cl in class_names:
            classD[cl] = l
    yD = {}
    for key, value in classD.items():
        if value in yD:
            yD[value].append(key)
        else:
            yD[value] = [key]
    y = np.array([classD[cl] for cl in sample_classes])

    return y, yD, classD

# # Convert authors to integers
# groups = [doc.doc._.author for doc in sdocs]
# authors = np.unique(groups)
# authorsD = {name:i for name, i in zip(authors,range(0,len(authors)))}
# groupsD = {i:name for name, i in zip(authors,range(0,len(authors)))}
# groups = np.array([authorsD[name] for name in groups])


def cv_classifier(feature_dataframe,
                  class_labels,
                  group_labels = None,
                  # metadata,
                  # class_names = 'subdir',
                  # group_names = '',
                  classifier = 'LGBM',
                  classifier_scheme = {'objective': 'binary', 'learning_rate': 0.5, 'metric': ['auc','binary',"xentropy"],
                                       'nthread': 12, 'boosting':'dart', # https://neptune.ai/blog/lightgbm-parameters-guide
                                        # 'max_bin':,
                                        'num_leaves': 5, 'num_iterations': 100,
                                        # 'min_data_in_leaf': 10,
                                        # 'min_sum_hessian_in_leaf':,
                                        'max_depth': 5, 'bagging_freq': 3, 'bagging_fraction':0.8,
                                        # 'feature_fraction':,
                                        'verbose': -1},
                  # TO DO: zautomatyzować przekazywanie argumentów do StratifiedKFold itp.
                  cv_scheme = {'cv_method':'StratifiedKFold',
                                 'n_repeats':10, # liczba powtórzeń
                                 'n_splits':10, # liczba podziałów wewnątrz *KFold i *ShuffleSplit
                                 'shuffle':True,# KFold, StratifiedKFold,StratifiedGroupKFold
                                 'n_groups': 2, #LeavePGroupsOut
                                 'p': 2, #LeavePOut
                                 'test_fold': [1, 0, 1, 1, 0], #PredefinedSplit
                                 'test_size': 0.2,#GroupShuffleSplit, ShuffleSplit
                                 'train_size': 0.2,# ShuffleSplit
                                 'random_state':None,#KFold,RepeatedKFold, RepeatedStratifiedKFold,ShuffleSplit
                                 'val_fraction':0.25 #train_test_split
                                 },
                  # TO DO: add hyperparamater tuning
                  verbose = 0,
                  precheck = False):
    
    if (cv_scheme['cv_method'] in ['GroupKFold','StratifiedGroupKFold','GroupShuffleSplit']) and cv_scheme['n_splits']>len(np.unique(group_labels)):
#        cv_scheme['n_splits'] = len(np.unique(group_labels))
        if verbose < 1:
            print("The number of distinct groups has to be at least equal to the number of folds. Proceeding with {} folds.".format(cv_scheme['n_splits']))
        
    #=== BEGIN CHOOSING OPTIONS ===
     # uwaga tutaj, jeślibyśmy mieli >2 grupy
    np.random.seed(1) # Reproducibility 
    random_states = np.random.randint(10000, size=cv_scheme['n_repeats']) # random states for generating subsequent data splits
    #=== END CHOOSING OPTIONS ===

    ensemble = []
    feature_names = feature_dataframe.columns.to_list()

    for i in tqdm_notebook(range(cv_scheme['n_repeats'])):
        # Get the default parameters for the chosen CV method
        cv_default_params = cv_method_parameters.get(cv_scheme['cv_method'], {})
        # Override default parameters with user-provided parameters
        cv_params = {**cv_default_params, **{key: value for key, value in cv_scheme.items() if key in cv_default_params.keys()}}
        # Initialize splitter
        # Get the cross-validation function dynamically
        cv_function = getattr(sklearn.model_selection,cv_scheme['cv_method'])
        cv_outer = cv_function(**cv_params)
        # cv_outer = cv_function(n_splits=cv_scheme['n_splits'], shuffle = cv_scheme['shuffle'], random_state=random_states[i]) # tu nie musi być grupowe(?), lepiej jakiś shuffle(?)
        # Iterate over folds
        for fold, (train_val_index, test_index) in  enumerate(cv_outer.split(feature_dataframe,class_labels,group_labels)):

            # Get train_val and test
            x_train_val, x_test = feature_dataframe.iloc[train_val_index,:], feature_dataframe.iloc[test_index,:]
            y_train_val, y_test = class_labels[train_val_index], class_labels[test_index] 
            
            
            if cv_scheme['val_fraction']*y_train_val.shape[0]<1:
                x_train = x_train_val
                y_train = y_train_val
                y_val = np.array([])
                valid_set = None
                early_stopping_rounds = None
                if verbose < 1:
                    print(f"'val_fraction' setting results in {cv_scheme['val_fraction']*y_train_val.shape[0]} validation samples. No nested validation will be performed.")
            else:
                # Partition the train_val set
                x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=cv_scheme['val_fraction'], random_state=0)

            print(f"Train:{y_train.shape[0]}, Val:{y_val.shape[0]}, Test:{test_index.shape[0]}")
            # Prepare dataset in LightGMB format
            y_train = np.squeeze(y_train)
            y_val = np.squeeze(y_val)
            train_set = lgb.Dataset(x_train, y_train, feature_name=feature_names) # Deprecated: silent=True
            if valid_set is not None:
                valid_set = lgb.Dataset(x_val, y_val, feature_name=feature_names)

                # Train the model 
                boosted_tree = lgb.train(
                    params = classifier_scheme,
                    train_set = train_set,
                    valid_sets = valid_set,
                    num_boost_round = 10000, # TO DO: to jest alias num_iterations przekazywany w params
                    early_stopping_rounds =  early_stopping_rounds,# Validation score needs to improve at least every stopping_rounds
                    verbose_eval = False,
                )
            else:
                                # Train the model 
                boosted_tree = lgb.train(
                    params = classifier_scheme,
                    train_set = train_set,
                    num_boost_round = 10000, # TO DO: to jest alias num_iterations przekazywany w params
                )


            # Save the model in the ensemble list
            ensemble.append({'trained_model':boosted_tree,'repeat':i,'fold':fold,'train_val_index':train_val_index,'test_index':test_index,'y_test':y_test,'y_train_val':y_train_val})
    return ensemble

# Setup dający wgląd w F1, AUC, MCC
def auc_av(y_true,y_pred):
    result = roc_auc_score(y_true,y_pred, average='macro')
    return result
def f1(y_true,y_pred):
    result = fbeta_score(y_true,y_pred, beta=1, average='macro')
    return result

dic_to_score = {'acc':{'func':accuracy_score},
                'f1':{'func':fbeta_score, 'params':{'beta':1, 'average':'macro'}},
                'auc':{'func':roc_auc_score,'params':{'average':'macro'}},
                'mcc': {'func':matthews_corrcoef},
                'recall':{'func':recall_score},
                'topk': {'func':top_k_accuracy_score,'params':{'k':1}}}

# TO DO: which metric need rounded y_test_pred
def probs_to_preds(y_test_pred):
    if y_test_pred.dtype == 'float64' and len(y_test_pred.shape)>1:
        return np.argmax(y_test_pred, axis=1), y_test_pred
    elif y_test_pred.dtype == 'float64':
        return y_test_pred.round(), y_test_pred
    elif y_test_pred.dtype == 'int32' or y_test_pred.dtype == 'int64':
        return y_test_pred, None
        
def score_predictions(y_test_pred, y_test, scoring):
    scores = {}
    y_test_pred, y_test_prob = probs_to_preds(y_test_pred)
    for k, v in scoring.items():
        if k == 'auc':
            scores[k] = v['func'](y_test, y_test_pred)
        elif k == 'mcc':
            scores[k] = v['func'](y_test, y_test_pred)
        elif k == 'topk':
            func = v['func']
            params = v.get('params', {})
            scores[k] = func(y_test, y_test_prob, **params)
        else:
            # Extract optional parameters for the function
            func = v['func']
            params = v.get('params', {})
            scores[k] = func(y_test, y_test_pred.round(), **params)
    return scores.copy()

# TODO: problem with functions in default scoring
def collect_scores(ensemble,
                   feature_dataframe,
                   scoring = {'acc':accuracy_score, 'f1':f1},
                   baseline_strategy = 'most_frequent'):#{“most_frequent”, “prior”, “stratified”, “uniform”, “constant”},
    
    if baseline_strategy:
        dummy_clf = DummyClassifier(strategy=baseline_strategy)
    
    scores = Scores(feature_dataframe.index)

    for record in tqdm_notebook(ensemble):
        model = record['trained_model']
        test_index = record['test_index']
        x_test = feature_dataframe.iloc[test_index,:]
        y_test = record['y_test']
        # Make predictions on test and compute performance metrics
        y_test_pred = model.predict(x_test)
        # scores_all.append(score_predictions(y_test_pred,y_test,scoring))
        scores.add_all(y_test_pred,y_test,scoring)        
        scores.add_preds(y_test,y_test_pred,test_index)
        
        if baseline_strategy:        
            train_val_index = record['train_val_index']
            x_train_val = feature_dataframe.iloc[train_val_index,:]
            y_train_val = record['y_train_val']
            dummy_clf.fit(x_train_val, y_train_val)
            y_test_pred = dummy_clf.predict_proba(x_test)

            # scores_base.append(score_predictions(y_test_pred,y_test,scoring))
            scores.add_base(y_test_pred,y_test,scoring)
        scores.add_cv(record['fold'],record['repeat'])
    return scores

def collect_scores_holdout(ensemble,
                   feature_dataframe,
                   y_test,
                   scoring = {'acc':accuracy_score, 'f1':f1}):
    scores = Scores(feature_dataframe.index)
    for record in tqdm_notebook(ensemble):
        model = record['trained_model']
        # Make predictions on test and compute performance metrics
        y_test_pred = model.predict(feature_dataframe)
        # scores_all.append(score_predictions(y_test_pred,y_test,scoring))
        scores.add_all(y_test_pred,y_test,scoring)        
        scores.add_preds(y_test,y_test_pred,feature_dataframe.index)
        scores.add_cv(record['fold'],record['repeat'])
    return scores


class Scores:
    scoreD = {'acc':'Accuracy [0-1, higher better]: \t',
              'f1':'F1 [0-1, higher better]: \t\t',
              'auc':'AUC [0-1, higher better]: \t\t',
              'topk':'Correct prediction in top k results [0-1, higher better]: \t\t',
              'mcc':'Matthews corr. [-1-1, higher better]: \t\t',
             'recall':'Recall [0-1, higher better]: \t'}
    def __init__(self, size):
        self.scores_base = []
        self.scores_all = []
        self.cv = []
        self.predictions_per_cv = dict()
        
        for sample in size:
            self.predictions_per_cv[sample]={}
            self.predictions_per_cv[sample]['y_true'] = np.array([])
            self.predictions_per_cv[sample]['y_pred'] = np.array([])
            self.predictions_per_cv[sample]['y_prob'] = np.array([])
            self.predictions_per_cv[sample]['errors'] = 0

    def add_preds(self,y_test,y_test_pred,test_index):
        y_test_pred, y_test_prob = probs_to_preds(y_test_pred)
        # Store individual sample predictions
        for j, index in enumerate(test_index):
            # next_key = len(predictions_per_cv[index])
            self.predictions_per_cv[index]['y_true']= np.append(self.predictions_per_cv[index]['y_true'],y_test[j])
            self.predictions_per_cv[index]['y_pred']=np.append(self.predictions_per_cv[index]['y_pred'],y_test_pred[j])
            self.predictions_per_cv[index]['y_prob']=np.append(self.predictions_per_cv[index]['y_prob'],y_test_prob[j])
            if y_test[j] != y_test_pred[j].round():
                self.predictions_per_cv[index]['errors']+=1
            
    def add_item(self, container, y_test_pred,y_test,scoring):
        container.append(score_predictions(y_test_pred,y_test,scoring))

    def add_all(self, y_test_pred,y_test,scoring):
        self.add_item(self.scores_all,y_test_pred,y_test,scoring)

    def add_base(self, y_test_pred,y_test,scoring):
        self.add_item(self.scores_base,y_test_pred,y_test,scoring)

    def add_cv(self, fold, repeat):
        self.cv.append({'fold':fold,'repeat':repeat})
    
    def convert_to_df(self):
        self.df_all = pd.DataFrame(self.scores_all)
        if self.scores_base:
            self.df_base = pd.DataFrame(self.scores_base)
        self.df_preds = pd.DataFrame(self.predictions_per_cv).T
        self.df_preds['errors'] = self.df_preds['errors'].astype(int)
        
    def print_scores(self):
        if not hasattr(self, 'df_all'):
            self.convert_to_df()
        if self.scores_base:
            for k in self.df_all.columns:
                print(f"{self.scoreD[k]}{self.df_all[k].mean():.2f} \t Baseline value: {self.df_base[k].mean(): .2f}")
        else:
            for k in self.df_all.columns:
                print(f"{self.scoreD[k]}{self.df_all[k].mean():.2f}")
    def save(self,filename,overwrite = False):
        filename = shared_module._prepare_filename(filename, '.pkl', overwrite)
        # with open("score_"+str(sample_length)+"ts.pkl", "rb") as file:
        with open(filename, "wb") as file:
            pickle.dump(self, file)
        # if self.verbose < 1:
        print(f"Scores saved as {filename}.")

# TO DO: add verbosity of messages for Scores and CVExplanation classes?

def collect_shap(ensemble,
                 feature_dataframe,
                 feature_names,
                 output_names,
                 minimize_data_use = True):
    
    explanations = CVExplanations(feature_dataframe.index)

    # shap_values_per_cv = dict()
    # for sample in feature_dataframe.index:
    #     shap_values_per_cv[sample] = {} 

    for record in tqdm_notebook(ensemble):
        model = record['trained_model']
        test_index = record['test_index']
        x_test = feature_dataframe.iloc[test_index,:]
        explainer = shap.TreeExplainer(model,feature_names=feature_names)
        shap_values = explainer(x_test) # shap_values = explainer.shap_values(x_test)
        shap_values.output_names = output_names
        
        explanations.add_shaps(shap_values,test_index,minimize_data_use=minimize_data_use)
        # # Extract SHAP information per fold per sample 
        # for j, index in enumerate(test_index):
        #     next_key = len(shap_values_per_cv[index])
        #     shap_values_per_cv[index][next_key] = shap_values[j,:,0] # niepotrzebnie składowane wielokrotnie to samo .data
        #     if next_key and minimize_data_use:
        #         # shap_values_per_cv[index][next_key].data = []
        #         shap_values_per_cv[index][next_key].values = shap_values_per_cv[index][next_key].values.astype('float16')
    return explanations

# TO DO: przepisać to tak, żeby shap_cv od razu były shap._explanation.Explanation
# https://github.com/slundberg/shap/blob/master/shap/_explanation.py
class CVExplanations:
    def __init__(self, size):
        self.size = size
        self.shap_cv = dict()
        for sample in self.size:
            self.shap_cv[sample] = {} 
    
    def compute_average(self):
        if hasattr(self, 'shap_average'):
            return
        # Establish lists to keep average Shap values, their Stds, and their min and max
        self.shap_average = []
        self.shap_average_base = []
        self.shap_stds = []
        self.shap_ranges = []
            
        for i in self.size:
            if len(self.shap_cv[i][0].values.shape)>1:
                df_per_obs = pd.DataFrame.from_dict({j:self.shap_cv[i][j].values[:,0] for j in range(self.n_repeats)})
            else:
                df_per_obs = pd.DataFrame.from_dict({j:self.shap_cv[i][j].values for j in range(self.n_repeats)})
                            
            # Get relevant statistics for every sample 
            self.shap_average.append(df_per_obs.mean(axis=1).values) 
            self.shap_average_base.append(np.average([self.shap_cv[i][j].base_values[0] for j in range(self.n_repeats)]))
            self.shap_stds.append(df_per_obs.std(axis=1).values)
            self.shap_ranges.append(df_per_obs.max(axis=1).values-df_per_obs.min(axis=1).values)
        self.shap_average = np.array(self.shap_average)
        self.df_shap_average = pd.DataFrame(self.shap_average,columns=self.shap_cv[0][0].feature_names)

    def compute_group(self,groups):
        if hasattr(self, 'groups'):
            # if (self.groups == groups).all():
            if np.array_equal(self.groups, groups):
                return
            else:
                print("Warning: overwriting previous `shap_group`")
        self.shap_group = []
        self.shap_group_base = []
        self.groups = groups
        # NP. ŚREDNIA PO AUTORZE
        for k in set(groups):
            df_per_obs = pd.DataFrame([self.shap_cv[i][j].values[:,0] for i in np.where(groups == k)[0] for j in range(self.n_repeats)])
            self.shap_group.append(df_per_obs.mean().values)
            self.shap_group_base.append(np.average([self.shap_cv[i][j].base_values[0] for i in np.where(groups == k)[0] for j in range(self.n_repeats)]))
        self.shap_group = np.array(self.shap_group)
        self.df_shap_group = pd.DataFrame(self.shap_group,columns=self.shap_cv[0][0].feature_names)

    def compute_text(self,index):
        text_explanations = shap._explanation.Explanation(self.shap_average[index],
                                                          base_values=self.shap_average_base[index],
                                                          data=self.shap_cv[index][0].data,
                                                          feature_names=self.shap_cv[index][0].feature_names,
                                                          output_names=self.shap_cv[index][0].output_names)
        return text_explanations
    
    def compute_text_group(self,index,feature_dataframe,groups=None):
        if groups is None:
            if hasattr(self, 'groups'):
                groups = self.groups
            else:
                print("Warning: no `groups` provided. Provide as an argument to `compute_text_group` or first call `compute_group(groups)`.")
                return
        else:
            self.compute_group(groups)
        text_explanations = shap._explanation.Explanation(self.shap_group[index],
                                                          base_values=self.shap_group_base[index],
                                                          data=feature_dataframe.groupby(by=groups).mean().iloc[index],
                                                          feature_names=self.shap_cv[index][0].feature_names,
                                                          output_names=self.shap_cv[index][0].output_names)
        return text_explanations
    
    def add_shaps(self,shap_values,test_index,minimize_data_use = True):
        # Extract SHAP information per fold per sample 
        for j, index in enumerate(test_index):
            next_key = len(self.shap_cv [index])
            # self.shap_cv[index][next_key] = shap_values[j,:,0]
            self.shap_cv[index][next_key] = shap_values[j,:]
#             TO DO: nie jestem pewien, czy działa
            if next_key and minimize_data_use:
                # self.shap_cv [index][next_key].data = [] # niepotrzebnie składowane wielokrotnie to samo .data
                self.shap_cv[index][next_key].values = self.shap_cv[index][next_key].values.astype('float16') # domyślnie jest float64
        self.n_repeats = len(self.shap_cv[0])
# TO DO: add saving and loading explanations
    def save(self,filename,overwrite = False):
        filename = shared_module._prepare_filename(filename, '.pkl', overwrite)
        # with open("score_"+str(sample_length)+"ts.pkl", "rb") as file:
        with open(filename, "wb") as file:
            pickle.dump(self, file)
        print(f"Explanations saved as {filename}.")

def save_models(ensemble,filename,overwrite = False):
    filename = shared_module._prepare_filename(filename, '.pkl', overwrite)
    with open(filename, "wb") as file:
        pickle.dump(ensemble, file)
    print(f"Classifiers saved as {filename}.")

    for i, model_dict in enumerate(ensemble):
        model = model_dict['trained_model']
        model.save_model('booster_'+str(i)+'.txt')
    print(f"Classifiers saved as booster_*.txt.")

def load_models(filename,verbose = 0):    
    filename = shared_module._check_filename(filename, '.pkl', verbose)
    if filename:
        with open(filename, "rb") as file:
            ensemble = pickle.load(file)
        print(f"Loaded cross-validated classifiers from {filename}.")
    else:
        ensemble = None
    return ensemble

def load_scores(filename,verbose = 0):    
    filename = shared_module._check_filename(filename, '.pkl', verbose)
    if filename:
        with open(filename, "rb") as file:
            scores = pickle.load(file)
        print(f"Loaded scores from {filename}. Printing scores:")
        scores.print_scores()
    else:
        scores = None
    return scores

def load_exps(filename, verbose = 0):
    filename = shared_module._check_filename(filename, '.pkl', verbose)
    if filename:
        # filename = os.path.splitext(filename)[0]
        # Load the instance from the file using pickle
        with open(filename, "rb") as file:
            exps = pickle.load(file)
        print(f"Loaded explanations from {filename}.")
    else:
        exps = None
    return exps
