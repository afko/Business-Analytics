# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 10:09:20 2018

BA_04122018
Ensemble Learning

@author: Justin
"""

import numpy as np # numerical python

T = 10 # the # of bootstrap samples
n = 100
boot_size = 100 # bootstrap은 중복을 허용하기 때문에 샘플의 수와 같아도 상관없다. (중복이 없다면 샘플보다는 낮아야한다.)

ind = []
for i in range(T):
    ind.append(np.random.choice(range(n), size = boot_size, replace = True))

print(len(np.unique(ind[1]))) # some data sample은 duplicated 됐다는 것을 알 수 있다.


from sklearn.datasets import load_iris

data = load_iris()
X = data.data
y = data.target

from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 1)

bgc = BaggingClassifier(base_estimator = knn, n_estimators = 3, oob_score = True) # 특정 object를 parameter로 넣는다. oob란 out-of-bag 이다.
bgc.fit(X,y)
bgc.base_estimator_ # base_estimator에 대한 info를 볼 수 있음
bgc.estimators_ 
bgc.estimators_samples_ # Selected Sample에 대한 info를 볼 수 있음

samples = bgc.estimators_samples_
selX = X[samples[0]]

bgc.oob_score_
bgc.score(X,y) # 1-neightbors를 사용했기 때문에 1에 거의 가깝다.

from sklearn.model_selection import train_test_split

trnX, valX, trnY, valY = train_test_split(X, y, test_size = 0.2, stratify = y)

bgc.fit(trnX, trnY)
bgc.score(valX, valY)


# drawing

bgc.fit(X[:, :2], y)

trnModels = bgc.estimators_
samples = bgc.estimators_samples_

xmin, xmax = X[:, 0].min(), X[:, 0].max()
ymin, ymax = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.linspace(xmin-0.5, xmax+0.5, 100), np.linspace(ymin-0.5, ymax+0.5, 100))

zz = np.c_[xx.ravel(), yy.ravel()] # columnize(?)


zz_pred = trnModels[0].predict(zz)

import matplotlib.pyplot as plt

i = 2
zz_pred = trnModels[i].predict(zz)
plt.contourf(xx, yy, zz_pred.reshape(xx.shape), alpha = 0.5)
plt.scatter(X[samples[0],0], X[samples[0],1], c=y[samples[0]])


for i in range(3):
    zz_pred = trnModels[i].predict(zz)
    plt.contourf(xx, yy, zz_pred.reshape(xx.shape), alpha = 0.5)

plt.scatter(X[:, 0], X[:, 1], c = y)

# n_estimators = 10
bgc = BaggingClassifier(base_estimator = knn, n_estimators = 10, oob_score = True) 
for i in range(10):
    zz_pred = trnModels[i].predict(zz)
    plt.contourf(xx, yy, zz_pred.reshape(xx.shape), alpha = 0.5)

plt.scatter(X[:, 0], X[:, 1], c = y)


# sin function graph를 위한 random set 만들기

x = np.random.uniform(-4,4,100)
y = np.sin(x) + np.random.normal(size = 100, scale = 0.4)
x = x.reshape((-1, 1))

xx = np.linspace(-4, 4, 100)
yy = np.sin(xx)

plt.scatter(x,y)
plt.plot(xx, yy, 'r')


from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

t = DecisionTreeRegressor(max_depth = 3)

bgr = BaggingRegressor(base_estimator = t, n_estimators = 3)

bgr.fit(x.reshape((-1, 1)), y) # 1-d를 2-d로 바꿔줘야함

trnModels = bgr.estimators_
y_pred = trnModels[0].predict(xx.reshape((-1,1)))
y_pred = trnModels[1].predict(xx.reshape((-1,1)))
plt.plot(xx, y_pred)

for i in range(3):
    y_pred = trnModels[i].predict(xx.reshape((-1, 1)))
    plt.plot(xx, y_pred, 'grey')
plt.plot(xx, yy, 'r')
y_pred = bgr.predict(xx.reshape((-1,1)))
plt.plot(xx, y_pred, 'b')
plt.plot(xx, yy, 'r')    


bgr = BaggingRegressor(base_estimator = t, n_estimators = 100)
bgr.fit(x.reshape((-1, 1)), y)

for i in range(100):
    y_pred = trnModels[i].predict(xx.reshape((-1, 1)))
    plt.plot(xx, y_pred, 'grey', alpha = 0.3)
plt.plot(xx, yy, 'r')
y_pred = bgr.predict(xx.reshape((-1,1)))
plt.plot(xx, y_pred, 'b')
plt.plot(xx, yy, 'r')    


from sklearn.ensemble import RandomForestClassifier

data = load_iris()
X = data.data
y = data.target
rf = RandomForestClassifier(max_depth = 3, min_samples_leaf = 10, oob_score = True )
rf.fit(X,y)

rf.oob_score_
rf.estimators_

rf.feature_importances_
# rf는 performance의 향상에 초점이 맞춰져 있다.

plt.barh(range(4), rf.feature_importances_)
plt.yticks(range(4), data.feature_names) 

