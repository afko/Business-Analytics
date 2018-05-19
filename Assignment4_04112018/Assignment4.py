# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 22:22:26 2018

Assignment4
 
"""
import pandas as pd
import matplotlib.pyplot as plt


vote = pd.read_csv(r"C:\Users\김성제\Google Drive\공부\4학년 1학기\Business Analytics\Data set\Week4\\US_County_Level_Presidential_Results_12-16.csv", index_col=0)
county = pd.read_csv(r"C:\Users\김성제\Google Drive\공부\4학년 1학기\Business Analytics\Data set\Week4\\county_facts.csv")

merge = pd.merge(vote, county, left_on = 'FIPS', right_on = 'fips',  how ='inner')

merge['target'] = (merge['votes_dem_2016']>merge['votes_gop_2016'])*1
merge['target'].value_counts()

ak = merge[merge['state_abbr']=='AK']
ak_mean = ak.mean().to_frame().T

data = pd.concat((merge[merge['state_abbr']!='AK'], ak_mean)) 
vars = [x for x in data.columns if "RHI" in x]

X = data[vars]
y = data['target']

###############################################################################

def logistic_(li_X, li_y):
    X = li_X
    y = li_y
    
    # Logistic Regression
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    clf.fit(X,y)
    clf.coef_
    clf.intercept_
    clf.score(X,y)
    y_pred = clf.predict(X)

    from sklearn.metrics  import recall_score, precision_score, f1_score, accuracy_score 
    
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
    
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve

    y_prob = clf.predict_proba(X) 
    roc_curve(y, y_prob[:,1], pos_label = 1)
    fpr, tpr, thres = roc_curve(y, y_prob[:,1], pos_label = 1)
    
    return a,b,c,d,plt.plot(fpr, tpr),roc_auc_score(y, y_prob[:,1])
    
def svc_(li_X, li_y):
    X = li_X
    y = li_y
    
    # Support Vector Machine
    from sklearn.svm import SVC
    
    svc = SVC(kernel = 'linear', probability = True)
    svc.fit(X,y)
    svc.score(X,y)
    y_pred = svc.predict(X)

    from sklearn.metrics  import recall_score, precision_score, f1_score, accuracy_score 
    
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
        svc.fit(X[train_set], y[train_set])
        y_pred = svc.predict(X[valid_set])
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
    
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve

    y_prob = svc.predict_proba(X) 
    roc_curve(y, y_prob[:,1], pos_label = 1)
    fpr, tpr, thres = roc_curve(y, y_prob[:,1], pos_label = 1)
    return a,b,c,d,plt.plot(fpr, tpr),roc_auc_score(y, y_prob[:,1])
    

###############################################################################

# Original
logistic_(X,y)
svc_(X,y)


# Over Sampling    
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
over_X, over_y = ros.fit_sample(X,y) 
logistic_(pd.DataFrame(over_X),pd.DataFrame(over_y))
svc_(pd.DataFrame(over_X),pd.DataFrame(over_y))

# Under Sampling
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(return_indices = True) 
under_X, under_y, inds = rus.fit_sample(X,y)
logistic_(pd.DataFrame(under_X), pd.DataFrame(under_y))
svc_(pd.DataFrame(under_X),pd.DataFrame(under_y))

# SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(k_neighbors = 10)
smote_X, smote_y = smote.fit_sample(X,y)
logistic_(pd.DataFrame(smote_X), pd.DataFrame(smote_y))
svc_(pd.DataFrame(smote_X), pd.DataFrame(smote_y)) # time spend

# Tomek Link
from imblearn.under_sampling import TomekLinks
tl = TomekLinks(return_indices = True)
tomek_X , tomek_y, inds = tl.fit_sample(X,y)
logistic_(pd.DataFrame(tomek_X), pd.DataFrame(tomek_y))
svc_(pd.DataFrame(tomek_X), pd.DataFrame(tomek_y))

# One-sided selection
from imblearn.under_sampling import OneSidedSelection
oss = OneSidedSelection(n_neighbors=1, n_seeds_S=1)
os_X, os_y = oss.fit_sample(X,y)
logistic_(pd.DataFrame(os_X), pd.DataFrame(os_y))
svc_(pd.DataFrame(os_X), pd.DataFrame(os_y))



