# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 13:54:10 2018

Assignment5

"""

import pandas as pd
import time

# HOUSE
house_data = pd.read_csv(r"C:\Users\김성제\Google Drive\공부\4학년 1학기\Business Analytics\Data set\Week2\\kc_house_data.csv")
house_X = house_data[['bedrooms', 'sqft_lot','bathrooms', 'sqft_living','waterfront', 'view', 'condition', 'grade']]
house_y = house_data['price'] 

# ELECTION
vote = pd.read_csv(r"C:\Users\김성제\Google Drive\공부\4학년 1학기\Business Analytics\Data set\Week4\\US_County_Level_Presidential_Results_12-16.csv", index_col=0)
county = pd.read_csv(r"C:\Users\김성제\Google Drive\공부\4학년 1학기\Business Analytics\Data set\Week4\\county_facts.csv")
merge = pd.merge(vote, county, left_on = 'FIPS', right_on = 'fips',  how ='inner')
merge['target'] = (merge['votes_dem_2016']>merge['votes_gop_2016'])*1
merge['target'].value_counts()
ak = merge[merge['state_abbr']=='AK']
ak_mean = ak.mean().to_frame().T
election_data = pd.concat((merge[merge['state_abbr']!='AK'], ak_mean)) 
vars = [x for x in election_data.columns if "RHI" in x]
[vars.append(x) for x in election_data.columns if 'PST' in x]
election_X = election_data[vars]
election_y = election_data['target']


trees = [5, 10, 20, 50, 100]


# HOUSE SALES RANDOM FOREST
from sklearn.model_selection import KFold
kf = KFold(n_splits = 5, shuffle = True, random_state = 10)

house_X = house_X.values
house_y = house_y.values

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

result_house = pd.DataFrame(index = ["MSE", "Processing Time"])



for tree in trees:
    start_time = time.time()
    mses = []

    rfr = RandomForestRegressor(n_estimators = tree,  min_samples_leaf = 10)
    
    for train_set, valid_set in kf.split(house_X, house_y):
        rfr.fit(house_X[train_set], house_y[train_set])
        house_y_pred = rfr.predict(house_X[valid_set])
        
        mses.append(mean_squared_error(house_y[valid_set], house_y_pred))
        
    avg_mse_score = sum(mses) / len(mses)
    end_time = time.time()
    process_time = end_time - start_time
    
    result_house[tree] = [avg_mse_score, process_time]
    
    
# ELECTION RANDOM FOREST
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits =5 , shuffle = True, random_state=10)

election_X = election_X.values
election_y = election_y.values

from sklearn.metrics  import recall_score, precision_score, f1_score, accuracy_score 
from sklearn.ensemble import RandomForestClassifier


result = pd.DataFrame(index = ["acc", "recall", "precision", "f1", "Processing Time"])

for tree in trees:
    start_time = time.time()
    accs = []
    recalls = []
    precs = []
    f1s = []
    
    rfc = RandomForestClassifier(n_estimators = tree, min_samples_leaf = 10)
    
    for train_set, valid_set in skf.split(election_X, election_y):
        rfc.fit(election_X[train_set], election_y[train_set])
        election_y_pred = rfc.predict(election_X[valid_set])
        
        accs.append(accuracy_score(election_y[valid_set], election_y_pred))
        recalls.append(recall_score(election_y[valid_set], election_y_pred))
        precs.append(precision_score(election_y[valid_set], election_y_pred))
        f1s.append(f1_score(election_y[valid_set], election_y_pred))
    
    avg_acc_score = sum(accs)/ len(accs)    
    avg_recall_score = sum(recalls)/ len(recalls)
    avg_precision_score = sum(precs)/ len(precs)  
    avg_f_score = sum(f1s)/ len(f1s)  
    
    end_time = time.time()
    process_time = end_time - start_time
    result[tree] = [avg_acc_score, avg_recall_score, avg_precision_score, avg_f_score, process_time]


