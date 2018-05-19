# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 20:14:55 2018

Assignment3

"""

import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

vote = pd.read_csv(r"C:\Users\김성제\Google Drive\공부\4학년 1학기\Business Analytics\Data set\Week4\\US_County_Level_Presidential_Results_12-16.csv", index_col=0)
county = pd.read_csv(r"C:\Users\김성제\Google Drive\공부\4학년 1학기\Business Analytics\Data set\Week4\\county_facts.csv")

#preprocessing for AK
merge = pd.merge(vote, county, left_on = 'FIPS', right_on = 'fips',  how ='inner')
merge['target'] = (merge['votes_dem_2016']>merge['votes_gop_2016'])*1
merge['target'].value_counts()
ak = merge[merge['state_abbr']=='AK']
ak_mean = ak.mean().to_frame().T
data = pd.concat((merge[merge['state_abbr']!='AK'], ak_mean))

# function of getting result
def analysis_result(list_):
    X = data[list_]
    y = data['target']
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    clf.fit(X,y)
    clf.coef_
    clf.intercept_
    clf.score(X,y)
    y_pred = clf.predict(X)
    
    from sklearn.metrics  import recall_score, precision_score, f1_score
    from sklearn.metrics import accuracy_score
    
    recall_score(y, y_pred) 
    precision_score(y, y_pred)
    f1_score(y, y_pred)
    accuracy_score(y, y_pred)
    
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
        clf.C = 1
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
    a = sum(accs)/len(accs)
    b = sum(recall)/len(recall)
    c = sum(precision)/len(precision)
    d = sum(f1s)/len(f1s)
    
    y_prob = clf.predict_proba(X) 
    roc_curve(y, y_prob[:,1], pos_label = 1)
    fpr, tpr, thres = roc_curve(y, y_prob[:,1], pos_label = 1)
    
    return a,b,c,d,plt.plot(fpr, tpr),roc_auc_score(y, y_prob[:,1])
    
# for looking keyword uniquely
# Before running this code, return value of function should be modified. (remain only a,b,c,d)
county_dic = pd.read_csv(r"C:\Users\김성제\Google Drive\공부\4학년 1학기\Business Analytics\Data set\Week4\\county_facts_dictionary.csv")
category_list = []
for i in range(1,len(county_dic['column_name'])):
    a = county_dic['column_name'][i][0]
    b = county_dic['column_name'][i][1]
    c = county_dic['column_name'][i][2]
    category_list.append(a+b+c)    
category_list = list(set(category_list))
category_list.remove("RHI") # We have to remove this.

df = pd.DataFrame(columns = ["acc", "recall", "precision", "f1"], index = category_list)
for i in category_list:    
    vars = [x for x in data.columns if 'RHI' in x]
    [vars.append(x) for x in data.columns if i in x]
    df.loc[i] = analysis_result(vars)

    
    

vars = [x for x in data.columns if 'RHI' in x]
[vars.append(x) for x in data.columns if 'HSG' in x]  
[vars.append(x) for x in data.columns if 'PST' in x]

analysis_result(vars)  
    
