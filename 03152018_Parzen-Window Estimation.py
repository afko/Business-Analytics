# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 10:37:30 2018

BA_03152018
Parzen-Window Estimation

@author: Justin
"""


import pandas as pd
data = pd.read_csv(r"C:\Users\김성제\Google Drive\공부\4학년 1학기\Business Analytics\Data set\Week2\\kc_house_data.csv")
data.dtypes
data.describe() # Basic Statistic 정보를 준다.

data['waterfront'].value_counts() # Categorized 되어서 개수를 세어보기 위함. 그 후 dummy variable로 바꾸어야함
data['condition'].value_counts() # 5개의 level이 있지만, 입맛에 맞게 그 level을 줄일 수도 있는 것이다.
len(data)

data['price'].plot.box() # IQR 공부하기

data['waterfront'].value_counts().plot.bar()

data['price'].plot.hist(bins = 20) # bins는 histogram에 나타낼 동일 간격의 개수를 의미
data['price'].plot.hist(bins = 5)

#!!! from pandas.plotting import sccater_matrix # 버전 낮아서 못하니까 업데이트 합시다.

corr = data[['price', 'bedrooms', 'sqft_lot']].corr()



from sklearn.linear_model import LinearRegression
reg = LinearRegression()

X = data[['bedrooms', 'sqft_lot','bathrooms', 'sqft_living','waterfront', 'view', 'condition', 'grade']] # input matrix
y = data['price'] #output target

reg.fit(X,y)
reg.score(X,y) # R-squared 높지 않지만, 그렇다고 낮지도 않다.


y_pred = reg.predict(X)
resid = y-y_pred

resid.plot.kde()


import numpy as np

y_new = np.log(y)

reg.fit(X,y_new)

data['log_price'] = np.log(data['price'])

#for Assignment
reg.coef_ #Interpret the effects of new variables
reg.intercept_



data['yr_built'].value_counts(sort = True)
len(data['yr_built'])


pd.__version__


np.__version__



