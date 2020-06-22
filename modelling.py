#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:34:24 2020

@author: Ben Branchflower
"""

import pandas as pd
import statsmodels as sm
import statsmodels.formula.api as smf

# models
from sklearn.linear_model import LogisticRegression

# metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# constants
infile = 'udar_features.tsv' # 'ben.tsv'

outcome_label_dict = {'A1':0,'A2':1,'B1':2,'B2':3,'C1':4,'C2':5}
inv_label_dict = {x:y for y, x in outcome_label_dict.items()}
labels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']


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

logit = LogisticRegression(max_iter=2000)
logit.fit(X, y)

y_pred = pd.Series(logit.predict(X)).replace(inv_label_dict)
y_true = y.replace(inv_label_dict)

print(confusion_matrix(y_true, y_pred, labels=labels))
print(classification_report(y_true, y_pred, labels=labels))