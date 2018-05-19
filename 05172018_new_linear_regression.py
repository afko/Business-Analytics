# -*- coding: utf-8 -*-
"""
Created on Thu May 17 09:48:04 2018

BA_05172018
Bias and Variance Linear Regression

@author: Justin
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.datasets import load_diabetes
import numpy as np

data = load_diabetes()
X = data.data
y = data.target


lr = LinearRegression() # 적용시키고 싶으면 일단 이걸 만들고, fit해야한다
lr.fit(X,y)
lr.coef_ # it contained 10 values 왜냐하면 10개의 설명 변수가 있기 때문이다.

ridge = Ridge(alpha = 10) # 그냥 알파값 1로함
ridge.fit(X,y)
ridge.coef_

np.abs(lr.coef_).sum()
np.abs(ridge.coef_).sum() # 알파값을 높여 페널티를 주었더니 값이 좀 떨어지게 된다.

lasso = Lasso(alpha = 0.1)
lasso.fit(X,y)
lasso.coef_ # 알파값을 1에서 낮추니 feature가 더 많이 선택되게 되었다.

# 적절한 알파값을 모르니까 grid search algorithm을 써야한다?

en = ElasticNet(alpha = 1) # alpha = 람다를 의미 l1_ratio는 베타를 의미
en.fit(X,y)
en.coef_ 


from sklearn.model_selection import train_test_split

Xtr, Xval, ytr, yval = train_test_split(X, y, test_size = 0.2)

alphas = np.logspace(-4, 4, 9)

r2 = []
for alpha in alphas:
    ridge.alpha = alpha
    ridge.fit(Xtr, ytr)
    r2.append(ridge.score(Xval, yval))    
r2 # not stable thing
# 이 방법을 통해 어떤 숫자가 다른 숫자보다 나은지를 알 수 있다.

r2 = []
for alpha in alphas:
    lasso.alpha = alpha
    lasso.fit(Xtr, ytr)
    r2.append(lasso.score(Xval, yval))   
r2

lr.fit(Xtr, ytr)
lr.score(Xval, yval) # r2 on a validation set lasso와 비교해서 이것과 저것중 가장 나은것이 무엇인지 비교할 수 있다.

### new exam
alphas = np.logspace(-4, 4, 20)
coefs = []
for alpha in alphas:
    lasso.alpha = alpha
    lasso.fit(X,y)
    coefs.append(lasso.coef_)

coefs = np.array(coefs)
coefs = coefs[:, ::-1]
alphas = alphas[::-1]

import matplotlib.pyplot as plt

for i in range(10):
    plt.plot(np.log(alphas), coefs[:,i])

import pandas as pd
import datetime
store = pd.read_csv(r'C:\\Users\김성제\Google Drive\공부\4학년 1학기\Business Analytics\Data set\Week10\store.csv\\store.csv')
train = pd.read_csv(r'C:\\Users\김성제\Google Drive\공부\4학년 1학기\Business Analytics\Data set\Week10\train.csv\\train.csv', dtype={'StateHoliday' : 'str'})   

X = train[train['Open']==1]

dummy = pd.get_dummies(train['StateHoliday'], prefix = 'StateHoliday') 

cat_var = ['StateHoliday', 'DayOfWeek']

for var in cat_var:
    dummy = pd.get_dummies(train[var], prefix = var, drop_first = True)
    X = pd.concat((X, dummy), axis = 1)
    
X = X.drop (cat_var, axis = 1)

store['InvCompetitionDistance'] = 1/ store['CompetitionDistance']
store['InvCompetitionDistance'] = store['InvCompetitionDistance'].fillna(0)
store['CompetitionOpenSinceMonth'] = store['CompetitionOpenSinceMonth'].fillna(12)
store['CompetitionOpenSinceYear'] = store['CompetitionOpenSinceYear'].fillna(2016)    


store['CompetitionOpen'] = store[["CompetitionOpenSinceYear", "CompetitionOpenSinceMonth"]].apply(lambda x: datetime.datetime(int (x.values[0]),int(x.values[1]), 1), axis=1)  


cat_var = ['StoreType', 'Assortment']
for var in cat_var:
    dummy = pd.get_dummies(store[var], prefix = var,drop_first = True)
    store = pd.concat((store,dummy),axis = 1)
store = store.drop(cat_var, axis = 1)

X = X.merge(store, on = 'Store')    
X.loc[X['Date']<X['CompetitionOpen'], 'InvCompetitionDistance']=0

X = X.sort_values(['Store', 'Date'])
sel_store = X[X['Store']==14]


X['CompetitionOpenSince'] = X['Date'] - X['CompetitionOpen']

X.dtypes

X['CompetitionOpenSince'].head()
X['CompetitionOpenSince']= X['CompetitionOpenSince'].astype('timedelta64[D]')

X.loc[X['CompetitionOpenSince']<0, 'CompetitionOpenSince']=0

