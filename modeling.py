#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:34:24 2020

@author: Ben Branchflower
"""

# general imports
import joblib as jl
import numpy as np
import pandas as pd
import statsmodels as sm
import statsmodels.formula.api as smf
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import plot_tree

# models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import XGBRFClassifier

# metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# constants
SEED = 123
NJOBS = 6 # set to -1 for all processors
infile = 'udar_features.tsv' # 'ben.tsv'


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

# create numeric outcome label
# df['label'] = df.filename.str.slice(7,9)
# df.label.value_counts()

df.level = df.level.replace(outcome_label_dict)

# handle missing values
data = df.drop('filename',axis=1).fillna(0) # FIXME: This is a just to get it to run for now

X = data.drop('level',axis=1)
y = data.level
# modelling
'''
model = smf.mnlogit(f'label ~ 1 + {"+".join(df.columns[:-1])}', data)
results = model.fit()
print(results)
'''
'''
# get a baseline of what we will be able to do iwth the data
logit = LogisticRegression(max_iter=2000, random_state=SEED, n_jobs=NJOBS) # convergence is an issue with the obs to feature ratio
logit.fit(X, y)

report(logit, model.predict(X), y)
'''
x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=SEED,
                                                    train_size=0.8)
'''
logit = LogisticRegression(max_iter=1000, class_weight='balanced', C=5,
                           solver='saga', penalty='elasticnet', l1_ratio=.5,
                           random_state=SEED, n_jobs=NJOBS) 
logit.fit(x_train, y_train)
#gs_obj = GridSearchCV(logit, {'C':[1,2,3,4,5], 'l1_ratio':[0.2,0.4,0.6,0.8]},
#                      n_jobs=NJOBS, verbose=1, cv=3)
#gs_obj.fit(x_train, y_train)

print(report(logit, model.predict(x_test), y_test))

report(gs_obj.best_estimator_, model.predict(X), y)
'''

rf = RandomForestClassifier(max_features='sqrt', random_state=SEED)

rf.fit(x_train,y_train)
report(rf, rf.predict(x_test), y_test)
report(rf, rf.predict(X), y)
feat_select = SelectFromModel(rf, 0.001, prefit=True)
colnames = X.columns[feat_select.get_support()]
with open('latest/rf.features', 'w') as f:
    print('\n'.join(colnames), file=f)
X_sub = feat_select.transform(X)
print(X_sub.shape)
x_sub_train, x_sub_test, y_train, y_test = train_test_split(X_sub,y,
                                                            random_state=SEED,
                                                            train_size=0.8)

rf.fit(x_sub_train, y_train)
report(rf, rf.predict(x_sub_test), y_test)
report(rf, rf.predict(X_sub), y)
jl.dump(rf, 'latest/rf.joblib')


fig, ax = plt.subplots(figsize=(38, 9))
plot_tree(rf.estimators_[84], max_depth=3, feature_names=colnames,
          filled=True, fontsize=12, impurity=False, proportion=True,
          class_names=labels)
fig.savefig('tree.png')


# XGBOOST Handles NaN natively so give it the raw form without filling
class_weights = df.level.value_counts().min() / df.level.value_counts()
weight_dict = {x:y for x,y in zip(class_weights.index, class_weights)}
weight_vec = df.level.replace(weight_dict)

x_train, x_test, y_train, y_test = train_test_split(df.drop(['filename','level'], axis=1),
                                                    df.level,
                                                    random_state=SEED,
                                                    train_size=0.8)
xgtrain = xgb.DMatrix(x_train.values, y_train.values, weight_vec[x_train.index])
xgtest = xgb.DMatrix(x_test.values, y_test.values, weight_vec[x_test.index])
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
clf.fit(x_train, y_train)

report(clf, clf.predict(x_train), y_train)
report(clf, clf.predict(x_test), y_test)
report(clf, clf.predict(df.drop(['filename','level'],axis=1)),df.level)

with open('latest/xgb.features', 'w') as f:
    print('\n'.join(colnames), file=f)
jl.dump(clf, 'latest/xgb.joblib')
"""
cv = StratifiedKFold(n_splits=4)

grid = {'max_depth':[13,14,15,16,17],'learning_rate':[0.2,0.15,0.1,0.05],
        'gamma':np.arange(0,1,.1), 'min_child_wight':np.arange(0,1,.1),
        'max_delta_step':[1,2,3,4,5], 'subsample':np.arange(.5,1.1,.1),
        'reg_alpha':np.arange(0,1.1,.1), 'reg_lambda':np.arange(0,1.1,.1)}
gs = GridSearchCV(clf, grid, n_jobs=NJOBS, cv=cv)
gs.fit(x_train, y_train)

print(gs.best_params_)

report(clf, gs.predict(x_train), y_train)
report(clf, gs.predict(x_test), y_test)
report(clf, gs.predict(df.drop(['filename','level'],axis=1)),df.level)

jl.dump(clf, "xgb.joblib")
"""

'''
tuning_xgb_params = {
    'max_depth':None
    }
    
num_round=2
bst = xgb.train(base_xgb_params, xgtrain, num_round)
preds = bst.predict(xgtest)

report(bst, preds.argmax(axis=1), y_test)
preds = bst.predict(xgtrain)
report(bst, preds.argmax(axis=1), y_train)
'''


