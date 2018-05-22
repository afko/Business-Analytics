# -*- coding: utf-8 -*-
"""
Created on Mon May 21 19:18:53 2018

Assignment7

"""
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import datetime
import numpy as np
import matplotlib.pyplot as plt


## Data
store = pd.read_csv(r'C:\\Users\김성제\Google Drive\공부\4학년 1학기\Business Analytics\Data set\Week10\store.csv\\store.csv')
train = pd.read_csv(r'C:\\Users\김성제\Google Drive\공부\4학년 1학기\Business Analytics\Data set\Week10\train.csv\\train.csv', dtype={'StateHoliday' : 'str'})   

store['PromoInterval'].unique()

## Data Preprocessing
X = train[train['Open']==1] 
X['Date'] = pd.to_datetime(X['Date'])

cat_var = ['StateHoliday', 'DayOfWeek']
for var in cat_var:
    dummy = pd.get_dummies(train[var], prefix = var, drop_first = True)
    X = pd.concat((X, dummy), axis = 1) 
X = X.drop (cat_var, axis = 1)

store['InvCompetitionDistance'] = 1 / store['CompetitionDistance']
store['InvCompetitionDistance'] = store['InvCompetitionDistance'].fillna(0) 
store['CompetitionOpenSinceMonth'] = store['CompetitionOpenSinceMonth'].fillna(12) 
store['CompetitionOpenSinceYear'] = store['CompetitionOpenSinceYear'].fillna(2016)
store['CompetitionOpen'] = store[["CompetitionOpenSinceYear", "CompetitionOpenSinceMonth"]].apply(lambda x: datetime.datetime(int (x.values[0]),int(x.values[1]), 1), axis=1)

# Promo2
store['Promo2SinceWeek'] = store['Promo2SinceWeek'].fillna(52)
store['Promo2SinceYear'] = store['Promo2SinceYear'].fillna(2016)
store['Promo2SinceMonth'] = round((store['Promo2SinceWeek']/4.35)+0.5)
store['Promo2Date'] = store[["Promo2SinceYear", "Promo2SinceMonth"]].apply(lambda x: datetime.datetime(int (x.values[0]),int(x.values[1]), 1), axis=1) 


cat_var = ['StoreType', 'Assortment']
for var in cat_var:
    dummy = pd.get_dummies(store[var], prefix = var, drop_first = True)
    store = pd.concat((store,dummy),axis = 1)
store = store.drop(cat_var, axis = 1)

X = X.merge(store, on = 'Store')
X.loc[X['Date']<X['CompetitionOpen'], ['InvCompetitionDistance']]=0

X = X.sort_values(['Store', 'Date'])

X['CompetitionOpenSince'] = X['Date'] - X['CompetitionOpen']
X['CompetitionOpenSince']= X['CompetitionOpenSince'].astype('timedelta64[D]')
X.loc[X['CompetitionOpenSince']<0, 'CompetitionOpenSince']=0

X['Promo2Since'] = X['Date'] - X['Promo2Date']
X['Promo2Since'] = X['Promo2Since'].astype('timedelta64[D]')
X.loc[X['Promo2Since']<0, 'Promo2Since'] = 0 

# Dummy for History sales(the month of year) and PromoInterval
X['Date_Month'] = pd.DatetimeIndex(X['Date']).month
cat_var = ['Date_Month', 'PromoInterval']
for var in cat_var:
    dummy = pd.get_dummies(X[var], prefix = var)
    X = pd.concat((X, dummy), axis = 1)

# Historical sales - the average last year sales
ma = pd.DataFrame(columns = ["Date", "InvAvgSales", "AvgSales"])
datetime.timedelta()
avg_sales = X.groupby(by = ['Date'])['Sales'].mean()
ma["AvgSales"] = avg_sales.rolling(window = 30).mean()
ma["AvgSales"] = ma["AvgSales"].fillna(0)
ma["InvAvgSales"] = (1 / avg_sales.rolling(window = 30).mean())
ma["InvAvgSales"] = ma["InvAvgSales"].fillna(0)
ma["Date"] = ma.index + datetime.timedelta(days = 365)

X = pd.merge(X, ma, on = "Date", how = "left")
X['AvgSales'] = X['AvgSales'].fillna(0)
X['InvAvgSales'] = X['InvAvgSales'].fillna(0)

## Drop useless column for analysis
X = X.drop(['Date_Month', 'PromoInterval', 'Store','CompetitionOpen', 'CompetitionDistance', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth', 'InvAvgSales', "Promo2SinceWeek", "Promo2SinceYear", "Promo2SinceMonth", "Promo2Date"], axis = 1)

# Train set
X_train = X[X['Date']>= datetime.datetime(2013, 1, 1)][X['Date']< datetime.datetime(2015,1,1)]
X_train = X_train.drop('Date', axis = 1)
y_train = X_train['Sales']
X_train = X_train.drop('Sales', axis = 1)
predictors = X_train.columns
y_train = y_train.values
X_train = X_train.values

# Valid set
X_valid = X[X['Date']>= datetime.datetime(2015, 1, 1)]
X_valid = X_valid.drop('Date', axis = 1)
y_valid = X_valid['Sales']
X_valid = X_valid.drop('Sales', axis = 1)
y_valid = y_valid.values
X_valid = X_valid.values

## Analysis

# Linear Regression
lr = LinearRegression()
lr.fit(X_train,y_train)
lr_coef = pd.Series(lr.coef_, predictors).sort_values()
lr_coef.plot(kind = 'bar', title = 'LR Coefficients')
lr.score(X_valid, y_valid)
lr.coef_
lr.intercept_
vif_lr = 1/ (1-lr.score(X_valid,y_valid))
np.abs(lr.coef_).sum()

alphas = np.logspace(-4, 4, 9)

# Ridge Regression
ridge = Ridge(alpha = 1) 
ridge.fit(X_train, y_train)
ridge_coef = pd.Series(ridge.coef_, predictors).sort_values()
ridge_coef.plot(kind = 'bar', title = 'Ridge Coefficients')
r2_ridge = []
ridge_coefs = []
ridge_coefs_sum = []
for alpha in alphas:
    ridge.alpha = alpha
    ridge.fit(X_train, y_train)
    r2_ridge.append(ridge.score(X_valid, y_valid))
    ridge_coefs.append(pd.Series(ridge.coef_, predictors).sort_values())
    ridge_coefs_sum.append(np.abs(ridge.coef_).sum())
    
ridge_coefs[7].plot(kind = 'bar', title = 'Ridge Coefficients')    
r2_ridge
ridge.coef_
np.abs(ridge.coef_).sum()

# Lasso Regression
lasso = Lasso(alpha = 1)
lasso.fit(X_train, y_train)

las_coef = pd.Series(lasso.coef_, predictors).sort_values()

r2_lasso = []
lasso_coefs = []
lasso_coefs_sum = []
for alpha in alphas:
    lasso.alpha = alpha
    lasso.fit(X_train, y_train)
    r2_lasso.append(lasso.score(X_valid, y_valid))
    lasso_coefs.append(pd.Series(lasso.coef_, predictors).sort_values())
    lasso_coefs_sum.append(np.abs(lasso.coef_).sum())
lasso_coefs[7].plot(kind = 'bar', title = 'Lasso Coefficients')
r2_lasso

