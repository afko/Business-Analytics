# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 10:21:04 2018

BA_03292018
classification

@author: Justin
"""

import pandas as pd
import numpy as np

vote = pd.read_csv(r"C:\Users\김성제\Google Drive\공부\4학년 1학기\Business Analytics\Data set\Week4\\US_County_Level_Presidential_Results_12-16.csv", index_col=0)
county = pd.read_csv(r"C:\Users\김성제\Google Drive\공부\4학년 1학기\Business Analytics\Data set\Week4\\county_facts.csv")
# = pd.read_csv(r"C:\Users\김성제\Google Drive\공부\4학년 1학기\Business Analytics\Data set\Week4\\county_facts_dictionary.csv")


# two data set 을 merge 하기
# ALaska의 경우

# 대소문자는 다른 ASCII를 사용하기 때문에 완전히 다르다.
merge = pd.merge(vote, county, left_on = 'FIPS', right_on = 'fips',  how ='inner') # how로 어떻게 join 할 것인지 결정한다.

merge['target'] = (merge['votes_dem_2016']>merge['votes_gop_2016'])*1

merge['target'].value_counts()

ak = merge[merge['state_abbr']=='AK']
ak_mean = ak.mean().to_frame().T

data = pd.concat((merge[merge['state_abbr']!='AK'], ak_mean)) # concat => concatenate: 사슬같이 잇다

vars = [x for x in data.columns if 'RHI' in x]


X = data[vars]
y = data['target']

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X,y)

clf.coef_
clf.intercept_
clf.score(X,y)

y_pred = clf.predict(X)

from sklearn.metrics import confusion_matrix # class name order를 제공한다.

y1 = [2, 0, 2, 2, 0, 1] # true value
y2 = [0, 0, 2, 2, 0, 2] # prediction value 

confusion_matrix(y1, y2) # 3x3 matrix 
# row - true case
# column - model case

clf.score(X,y)

from sklearn.metrics  import recall_score, precision_score, f1_score
from sklearn.metrics import accuracy_score

recall_score(y, y_pred) # accuracy에 비해 낮다. true positive rate가 낮다.
precision_score(y, y_pred)
f1_score(y, y_pred)
accuracy_score(y, y_pred) # 왜 recall_score가 낮은데 accuracy가 높냐? 
# size의 차이가 있으면 accuracy가 비정상적으로 높을 수 있다.



Cs = np.logspace(-2, 2, 5) # log scale을 준다. (start, end, # of sample)

from sklearn.model_selection import StratifiedKFold

k = 5

skfold = StratifiedKFold(n_splits = k)

accs = []
recall = []
precision = []
f1s = []

X = X.values
y = y.values

for train_set, valid_set in skfold.split(X,y):
    for C in Cs:
        clf.C = C
        clf.fit(X[train_set], y[train_set])
        y_pred = clf.predict(X[valid_set])
        acc = accuracy_score(y[valid_set],y_pred)
        r = recall_score(y[valid_set], y_pred)
        p = precision_score(y[valid_set], y_pred)
        f1 = f1_score(y[valid_set], y_pred)
        accs.append(acc)
        recall.append(r)
        precision.append(p)
        f1s.append(f1)

accs = np.reshape(accs, (5,5))
accs.mean(0) # column ~~
recall = np.reshape(recall, (5,5))
recall.mean(0)


from sklearn.metrics import roc_curve

y_prob = clf.predict_proba(X) # logistic regression 은 predict probability를 제공

roc_curve(y, y_prob[:,1], pos_label = 1)

fpr, tpr, thres = roc_curve(y, y_prob[:,1], pos_label = 1)

import matplotlib.pyplot as plt
plt.plot(fpr, tpr)

from sklearn.metrics import roc_auc_score
roc_auc_score(y, y_prob[:,1])

from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

clf.fit(X,y)

y_pred = clf.predict(X)

# multiclass 일때
recall_score(y,  y_pred, average = 'macro')
recall_score(y,  y_pred, average = 'micro')
recall_score(y,  y_pred, average = 'weighted')


