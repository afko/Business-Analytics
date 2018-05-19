# -*- coding: utf-8 -*-
"""
Created on Tue May  8 02:04:58 2018

Assignment6

"""

import pandas as pd
import numpy as np
import re
import string


# Data Preprocessing
data = pd.read_csv(r'C:\\Users\김성제\Google Drive\공부\4학년 1학기\Business Analytics\Data set\Week9\\gender-classifier-DFE-791531.csv', encoding = 'latin1')
bin_data = data[data['gender'].isin(['female', 'male'])]

text = bin_data['text']

url_pattern = '(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9]\.[^\s]{2,})'
clean = text.apply(lambda x: re.sub('#\w+', '', x))
clean = clean.apply(lambda x: re.sub('@\w+', '', x))
clean = clean.apply(lambda x: re.sub(url_pattern, '', x))

string.punctuation
string.ascii_letters 

clean = clean.apply(lambda x: ''.join([y for y in x if not y in string.punctuation]))
clean = clean.apply(lambda x: ''.join([y for y in x if y in string.ascii_letters+' ']))
clean = clean.apply(lambda x: x.lower())



from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(min_df = 8) 
cv.fit(clean)
vocab = cv.vocabulary_ 

X = cv.transform(clean) 
type(X)
X = X.toarray()
y = (bin_data['gender'] == 'female') * 1
     
##__##
# Preprocessing for making DFS
x_df = pd.DataFrame(X, columns = cv.get_feature_names())
y_reindex = pd.Series(y)
y_reindex = y_reindex.reset_index()
y_reindex = y_reindex.drop(columns=['index'])
x_df['y'] = y_reindex
y_reindex = x_df['y']

class_name = x_df['y'].unique()
for c in class_name:
    print(c)

# Practise for making DFS
for x_col in x_df.columns:
    if x_col == 'y':
        break
    else:
        a = x_df['and'][x_df['and']!=0] # 윗칸 분모
        a = a.count()
        
        b = x_df['and'][x_df['and']!=0][x_df['y']==0] # 윗칸 분자 
        b = b.count()
        
        c = x_df['and'][x_df['and']==0][x_df['y']==0] # 아랫칸 왼쪽 분자
        c = c.count()
        
        d = x_df['and'][x_df['y']==0] # 아랫칸 왼쪽 분모
        d = d.count()
        
        e = x_df['and'][x_df['and']!=0][x_df['y']!=0] # 아랫칸 오른쪽 분자
        e = e.count()
        
        f = x_df['and'][x_df['y']!=0] # 아랫칸 오른쪽 분모
        f = f.count()

dfs_score = (b/a)/((c/d)+(e/f)+1)

##_______##
# DFS 
def dfs(X, y):
    X_re = pd.DataFrame(X, columns = cv.get_feature_names())
    X_re['y'] = y
    class_name = X_re['y'].unique()
    
    dfs_scores = []
    for x_col in X_re.columns:
        if x_col == 'y':
            break
        else:
            final_dfs = 0
            for cn in class_name:
                a = X_re[x_col][X_re[x_col]!=0] 
                a = a.count()
                
                b = X_re[x_col][X_re[x_col]!=0][X_re['y']==cn] 
                b = b.count()
                
                c = X_re[x_col][X_re[x_col]==0][X_re['y']==cn] 
                c = c.count()
                
                d = X_re[x_col][X_re[x_col]==0]
                d = d.count()
                
                e = X_re[x_col][X_re[x_col]!=0][X_re['y']!=cn]
                e = e.count()
                
                f = X_re[x_col][X_re['y']!=cn]
                f = f.count()
            
                dfs_score = (b/a)/((c/d)+(e/f)+1)
                final_dfs += dfs_score
            dfs_scores.append(final_dfs)
    dfs_scores = np.asarray(dfs_scores)
    zeroarr = np.zeros(len(X[0]))
    dfs_res = (dfs_scores,zeroarr)
    return dfs_res   

##__##

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif


from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
y = y.values
y_reindex = y_reindex.values

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits = 5 , shuffle = True, random_state = 10)

ks = [100, 500, 1000, 2000]


# chi2
mnb_chi2 = []
for a in ks:    
    fs_chi2 = SelectKBest(score_func = chi2, k = a)
    fs_chi2.fit(X, y)
    Xred_chi2 = fs_chi2.transform(X)
    
    mnb_scores = []
    
    for train_set, valid_set in skf.split(Xred_chi2, y):
        mnb.fit(Xred_chi2[train_set], y[train_set])
        mnb_score = mnb.score(Xred_chi2[valid_set], y[valid_set])
        mnb_scores.append(mnb_score)    
    mnb_chi2.append(sum(mnb_scores)/len(mnb_scores))    

# mutual_info_classif
mnb_mic = []
for a in ks:
    fs_mic = SelectKBest(score_func = mutual_info_classif, k = a)
    fs_mic.fit(X,y)
    Xred_mic = fs_mic.transform(X)
    
   
    mnb_scores_mic = []
    for train_set, valid_set in skf.split(Xred_mic, y):
        mnb.fit(Xred_mic[train_set], y[train_set])
        mnb_score = mnb.score(Xred_mic[valid_set], y[valid_set])
        mnb_scores_mic.append(mnb_score)   
    mnb_mic.append(sum(mnb_scores_mic)/len(mnb_scores_mic))

# DFS
mnb_dfs = []
for a in ks:
    
    fs_dfs = SelectKBest(score_func = dfs, k = a)
    fs_dfs.fit(X, y_reindex)
    Xred_dfs = fs_dfs.transform(X)

    mnb_scores_dfs = []
    for train_set, valid_set in skf.split(Xred_dfs, y_reindex):
        mnb.fit(Xred_dfs[train_set], y_reindex[train_set])
        mnb_score = mnb.score(Xred_dfs[valid_set], y_reindex[valid_set])
        mnb_scores_dfs.append(mnb_score)  
    mnb_dfs.append(sum(mnb_scores_dfs)/len(mnb_scores_dfs))
    
    
    