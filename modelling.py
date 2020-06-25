#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:34:24 2020

@author: Ben Branchflower
"""

# general imports
import pandas as pd
import statsmodels as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
# models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost

# metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# constants
SEED = 123
NJOBS = 4 # set to -1 for all processors
infile = 'udar_features.tsv' # 'ben.tsv'


outcome_label_dict = {'A1':0,'A2':1,'B1':2,'B2':3,'C1':4,'C2':5}
inv_label_dict = {x:y for y, x in outcome_label_dict.items()}
labels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

# helper functions
def report(model, X, y):
    y_pred = pd.Series(model.predict(X)).replace(inv_label_dict)
    y_true = y.replace(inv_label_dict)
    
    print(confusion_matrix(y_true, y_pred, labels=labels))
    print(classification_report(y_true, y_pred, labels=labels))


###  Data loading and processing ###
df = pd.read_csv(infile, sep='\t', error_bad_lines=False)

# create numeric outcome label


df['label'] = df.filename.str.slice(7,9)
df.label.value_counts()

df.label = df.label.replace(outcome_label_dict)

# handle missing values
data = df.drop('filename',axis=1).fillna(0) # FIXME: This is a just to get it to run for now

X = data.drop('label',axis=1)
y = data.label
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

report(logit, X, y)
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

print(report(logit, x_test, y_test))

report(gs_obj.best_estimator_, X, y)
'''

rf = RandomForestClassifier(max_features='sqrt')

rf.fit(x_train,y_train)
report(rf, x_test, y_test)
report(rf, X, y)
feat_select = SelectFromModel(rf, 0.001, prefit=True) 
X_sub = feat_select.transform(X)
print(X_sub.shape)
x_sub_train, x_sub_test, y_train, y_test = train_test_split(X_sub,y,
                                                            random_state=SEED,
                                                            train_size=0.8)

rf.fit(x_sub_train, y_train)
report(rf, x_sub_test, y_test)
report(rf, X_sub, y)

xgboo




