# -*- coding: utf-8 -*-
"""
Created on Thu May 10 09:27:38 2018

BA_05102018
Time Series

@author: Justin
"""

import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np

store = pd.read_csv(r'C:\\Users\김성제\Google Drive\공부\4학년 1학기\Business Analytics\Data set\Week10\store.csv\\store.csv')
train = pd.read_csv(r'C:\\Users\김성제\Google Drive\공부\4학년 1학기\Business Analytics\Data set\Week10\train.csv\\train.csv', dtype={'StateHoliday':str})

len(train['Store'].unique())

ts = train[train['Store']==1] # More than 2 years record
ts.dtypes # date가 object로 되어있는것을 알 수 있다 => date type으로 바꿔주어야한다.
ts['Date'] = pd.to_datetime(ts['Date']) # object를 datetime으로
ts.dtypes # datatime64로 바뀐것을 알 수 있다.

ts = ts.sort_values('Date')

plt.plot(ts['Sales']) # X axis가 숫자이기 때문에 date로 바꿔주고싶다.
ts = ts.set_index('Date') # index를 바꿔주고 date가 나오게 하자.
plt.plot(ts['Sales'])

# Average Sales  
avg_sales = train.groupby(by = ['Date'])['Sales'].min() # based on key variable(Date), Sales라는 numeric data를 확인하고싶다.
avg_sales = train.groupby(by = ['Date'])['Sales'].max() 
avg_sales = train.groupby(by = ['Date'])['Sales'].mean() 

avg_sales.index = pd.to_datetime(avg_sales.index)
plt.plot(avg_sales) # avg_sales를 이용하면 trend를 알기어려워

# window를 이용하기
ma = avg_sales.rolling(window = 30).mean()
plt.plot(ma)

train['Customers'].describe() # useful function for summary statistics
train['Customers'].plot.hist(20)

store['CompetitionDistance'].describe()

store['StoreType'].value_counts() # 4 different sort type
store['Assortment'].value_counts() # 3 different assortment level
train['StateHoliday'].value_counts() # 0이 두개가 나오는데 왜 그런것일까? A. pandas는 data를 자동으로 읽음. 처음엔 0이 나와서 numeric으로 읽다가 갑자기 a,b,c 같은게 나오면 다시 string으로 가고, 따라서 다시 0을 읽으면 string으로 읽음. 그래서 ! dtype을 설정해줘야함


data = pd.merge(train, store, on ='Store', how = 'inner') # left, right, key_value
store['CompetitionOpenSinceYear'].describe() # 특정 시점부터 가치 있는 데이터가 있다.

#?? 근데 이거 안 씀
store['CompetitionOpenSince'] = store[['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth']].apply(lambda x: datetime.datetime(int(x.values[0]), int(x.values[1]), 1) if x['CompetitionOpenSinceYear'] != np.nan else np.nan, axis = 1)

store1 = store[store['CompetitionOpenSinceMonth'].isnull()==False]
store1['CompetitionOpenSince'] = store1[['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth']].apply(lambda x: datetime.datetime(int(x.values[0]), int(x.values[1]),1), axis = 1)

store1[:5][['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth']].apply(lambda x:print(x), axis = 1)

data1 = pd.merge(train, store1, on = 'Store')
data1['Date'] = pd.to_datetime(data1['Date'])

data1['Compare'] = data1['Date'] >= data1['CompetitionOpenSince']

data1 = data1.sort_values(['Store', 'Date'])

store['Promo2SinceWeek'].describe() # week number 32라면 1년중 32번째 주를 의미


