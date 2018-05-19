# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 10:07:09 2018

BA_03222018
k-fold cross validation

@author: Justin
"""

import pandas as pd
data = pd.read_csv(r"C:\Users\김성제\Google Drive\공부\4학년 1학기\Business Analytics\Data set\Week2\\kc_house_data.csv")


# 오늘은 pandas로 날짜를 다뤄보자
# date가 현재는 string이다.

t = pd.to_datetime(data['date'])
t.dtype # pandas로 convert 하기전과 후에 어떤 dtype인지 확인하기 위해서 쓰임

data['datetime'] = t

data['datetime'].dt.year
data['datetime'].dt.month
data['datetime'].dt.day
data['datetime'].dt.dayofweek # 0는 월요일 1은 화요일 ... 뭐 그런식

data['sold_year'] = data['datetime'].dt.year # 몇 년도에 집이 팔렸는지 뭐 그런
data['diff_built_year'] = data['sold_year'] - data['yr_built']

data['diff_built_year'].describe()

sel_data = data[data['diff_built_year'] < 0] # 이런식으로 좀 문제인 애들이 실제로 존재하기 마련이다.

data['bedrooms'].value_counts().sort_index() # 이상치를 찾기 위해서 사용할 수 있다.
data['bathrooms'].value_counts().sort_index() # 이상하면 제외하자... (0.25는 뭐야...)


pd.date_range(start = '2013-01-01', end = '2013-01-31', freq = 'D') # datetime의 간격을 확인하고자
pd.date_range(start = '2013-01-01', end = '2013-12-31', freq = 'M')

s1 = pd.date_range(start = '2013-01-01', end = '2013-01-31', freq = 'D')
s2 = pd.date_range(start = '2014-03-01', end = '2014-03-31', freq = 'D')

diff = s2 - s1
diff.values

diff.astype("timedelta64[M]") # astype 으로 ns 를 year로 바꿀 수 있다.


data['diff_renov_year'] = data['sold_year'] - data['yr_renovated']
data['diff_renov_year'].describe()

data['diff_renov_year'].hist(bins = 30)



from sklearn.model_selection import train_test_split
# from sklearn.cross_validation # old version

trains_set, test_set = train_test_split(data, test_size = 0.2) # 무작위로 train set과 test set이 정해짐

from sklearn.datasets import load_iris

iris = load_iris()
irisX = iris.data
irisY = iris.target

trainX, testx, trainY, testY = train_test_split(irisX, irisY, test_size = 0.2) # default로 shuffle은 true다.

from sklearn.model_selection import KFold

kfold = KFold(n_splits = 3, shuffle = True)

# example
X = list(range(8,14))
for tr_set, val_set in kfold.split(X): # index 로 나온다 어느 범위건
    print(tr_set, val_set)

    
import numpy as np    
for tr_set, val_set in kfold.split(irisY):
    print(np.bincount(irisY[tr_set]) / len(tr_set), np.bincount(irisY[val_set]) / len(val_set))    
    
np.bincount([1,2,1,0,3]) # 각 수의 빈도를 나타내준다. 0은 1개 1은 2개 ... 이런식으로 한다.


from sklearn.model_selection import StratifiedKFold

skfold = StratifiedKFold(n_splits = 3, shuffle = True)
for tr_set, val_set in skfold.split(irisX, irisY ):
    print(np.bincount(irisY[tr_set]) / len(tr_set), np.bincount(irisY[val_set]) / len(val_set))    

# 어떻게 이 split 된 값들을 사용할 것인가?

from sklearn.linear_model import LogisticRegression # Classification problem에 사용되는 방법 중 하나를 사용해보자
clf = LogisticRegression() # logisticregrssion object를 만들자.

acc = [] # validation의 accuracy를 저장하기 위함

for tr_set, val_set in skfold.split(irisX, irisY):
    trX = irisX[tr_set]
    trY = irisY[tr_set] # Target 설정
    clf.fit(trX, trY)
    acc.append(clf.score(irisX[val_set], irisY[val_set]))

acc
np.mean(acc)


from sklearn.naive_bayes import GaussianNB # 또 다른 방법으로 해보자
gnb = GaussianNB()

acc_nb = []

for tr_set, val_set in skfold.split(irisX, irisY):
    trX = irisX[tr_set]
    trY = irisY[tr_set] # Target 설정
    gnb.fit(trX, trY)
    acc_nb.append(gnb.score(irisX[val_set], irisY[val_set]))

acc_nb
np.mean(acc_nb)


from sklearn.metrics import mean_squared_error # 다시 공부하자... f 를 위해
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

y = data['price']
X = data[['bedrooms', 'bathrooms', 'waterfront', 'sqft_living', 'diff_built_year']]

reg.fit(X,y)

y_pred = reg.predict(X)

mean_squared_error(y, y_pred)




