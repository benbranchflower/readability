#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:34:24 2020

@author: Ben Branchflower
"""

import sys

# general imports
import numpy as np
import pandas as pd
import statsmodels as sm
import statsmodels.formula.api as smf
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel

# models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import XGBRFClassifier

# metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# constants
try:
    infile = sys.argv[1] # 'ben.tsv'
except IndexError:
    raise IndexError('Include file name for the data')
    
try:
    outfile = sys.argv[2] # 'ben.tsv'
except IndexError:
    raise IndexError('Include file name for the output')
    
try:
    NJOBS = int(sys.argv[3]) # set to -1 for all processors
except IndexError:
    raise IndexError('Include an integer value for number of processors, -1 for all available')
try:
    SEED = int(sys.argv[4])
except IndexError:
    Seed = 123

outcome_label_dict = {'A1':0,'A2':1,'B1':2,'B2':3,'C1':4,'C2':5}
inv_label_dict = {x:y for y, x in outcome_label_dict.items()}
labels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

# helper functions
def report(model, pred, y):
    y_pred = pd.Series(pred).replace(inv_label_dict)
    y_true = y.replace(inv_label_dict)
    
    print(confusion_matrix(y_true, y_pred, labels=labels))
    print(classification_report(y_true, y_pred, labels=labels))


###  Data loading and processing ###
df = pd.read_csv(infile, sep='\t', error_bad_lines=False)

df.level = df.level.replace(outcome_label_dict)

# modelling
# XGBOOST Handles NaN natively so give it the raw form without filling
x_train, x_test, y_train, y_test = train_test_split(df.drop(['filename','level'], axis=1),
                                                    df.level,
                                                    random_state=SEED,
                                                    train_size=0.8)
# xgtrain = xgb.DMatrix(x_train.values, y_train.values, weight_vec[x_train.index])
# xgtest = xgb.DMatrix(x_test.values, y_test.values, weight_vec[x_test.index])
base_xgb_params = {
    'max_depth':10,
    'learning_rate':0.1,
    # 'n_estimators':1000,
    'objective':'multi:softprob',
    'booster':'gbtree',
    'n_jobs':NJOBS,
    'gamma':0,
    'min_child_weight':1,
    'max_delta_step':0,
    'subsample':1,
    'reg_alpha':0,
    'reg_lambda':1,
    'base_score':0.5,
    'random_state':SEED,
    # 'missing':np.nan,
    'verbosity':1,
    'num_class':6
    }
clf = XGBClassifier(**base_xgb_params)
'''
clf.fit(x_train, y_train)

report(clf, clf.predict(x_train), y_train)
report(clf, clf.predict(x_test), y_test)
report(clf, clf.predict(df.drop(['filename','level'],axis=1)),df.level)
'''

cv = StratifiedKFold(n_splits=4)

grid = {'max_depth':[13,14,15,16,17],'learning_rate':[0.2,0.15,0.1,0.05],
        'gamma':np.arange(0,1,.1), #'min_child_wight':np.arange(0,1,.1),
        'max_delta_step':[1,2,3,4,5], 'subsample':np.arange(.5,1,.1),
        'reg_alpha':np.arange(0,1,.1), 'reg_lambda':np.arange(0,1,.1)}
gs = GridSearchCV(clf, grid, n_jobs=NJOBS, cv=cv, verbose=1)
gs.fit(x_train, y_train)

with open(outfile) as f:
    f.write(gs.best_params_)
    f.write('\n')
    f.write(report(clf, gs.predict(x_train), y_train))
    f.write(report(clf, gs.predict(x_test), y_test))
    f.write(report(clf, gs.predict(df.drop(['filename','level'],axis=1)),df.level))



