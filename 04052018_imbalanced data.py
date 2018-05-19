# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 10:30:38 2018

BA_04052018
imbalanced data

@author: Justin
"""

####
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold

data = load_iris()
X = data.data
y = data.target

kfold = StratifiedKFold(shuffle = True, random_state = 10) # random_state를 넣어서 그 정도를 다르게 할 수 있다. Seed를 고정

for train_set, valid_set in kfold.split(X,y):
    print(train_set)
    
####


import numpy as np
np.random.seed(0)

n_sample_1 = 1000
n_sample_2 = 100

#np.r_은 array들을 적절하게 concatenation처럼 잇는다.
X = np.r_[np.random.randn(n_sample_1 , 2)*1.5 , np.random.randn(n_sample_2, 2)*0.5+[2,2]] #2,2는 for minority class
y = [0]*n_sample_1 + [1]*n_sample_2 # 0이 1000개, 1이 100개가 있는 1100*1 행렬이 나옴

import matplotlib.pyplot as plt
plt.scatter(X[:,0], X[:,1], c = y)
plt.show()

from sklearn.svm import SVC

svc = SVC(kernel = 'linear')
svc.fit(X,y)
svc.score(X,y)
y_pred = svc.predict(X)

from sklearn.metrics import recall_score, precision_score

recall_score(y, y_pred)
precision_score(y, y_pred)

# sampling method는 high accuracy를 보장하지는 않는다.


# Over Sampling
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler()

X_sampled, y_sampled = ros.fit_sample(X,y) # minority에서 반복적으로 뽑혔기 때문에 전부 같은 class가 나온다. (overfitting되서)

svc.fit(X_sampled , y_sampled)
svc.score(X_sampled, y_sampled)

y_pred = svc.predict(X_sampled)

recall_score(y_sampled, y_pred)
precision_score(y_sampled, y_pred)


# Under Sampling
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(return_indices = True) # 저 parameter를 True로 해야한다.

X_resampled, y_resampled, inds = rus.fit_sample(X,y) # 200개로 줄어들어서 majority의 density가 현저하게 줄어들었다.


# SMOTE
from imblearn.over_sampling import SMOTE

smote = SMOTE(k_neighbors = 10)

X_sampled, y_sampled = smote.fit_sample(X,y) # new data point를 합성해서 만들었기 때문에 오리지널 데이터 셋과는 살짝 다른 분포를 보여준다.

X_sampled1, y_sampled1 = ros.fit_sample(X,y)
X_sampled2, y_sampled2 = smote.fit_sample(X,y)


# ROS와 SMOTE data unique성 비교
import pandas as pd

X_sampled1 = pd.DataFrame(X_sampled1)
len(X_sampled1.drop_duplicates()) # unique가 많지않다

X_sampled2 = pd.DataFrame(X_sampled2)
len(X_sampled2.drop_duplicates()) # new data가 있기 때문에 unique가 상대적으로 많긴 하다.


# Tomek Link
from imblearn.under_sampling import TomekLinks

tl = TomekLinks(return_indices = True)
X_resampled,y_resampled,inds=tl.fit_sample(X,y)


# One-sided selection
# remove every data point, 근데 그 전에 k-nn을 적용해야한다.

from imblearn.under_sampling import OneSidedSelection

oss = OneSidedSelection(n_neighbors=1, n_seeds_S=1)
X_resampled, y_resampled = oss.fit_sample(X,y)



#Cost-sensitive Learning

svc = SVC(kernel = 'linear', class_weight = {1:10})
svc.fit(X,y)

y_pred = svc.predict(X)
recall_score(y,y_pred)

