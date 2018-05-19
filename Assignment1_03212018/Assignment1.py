# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 14:41:40 2018

Assignment1

"""

# Data Loading
import pandas as pd
data = pd.read_csv(r"C:\Users\김성제\Google Drive\공부\4학년 1학기\Business Analytics\Data set\Week2\\kc_house_data.csv")


# for scatter matrix
from pandas.plotting import scatter_matrix as sm
sm(data[['price','grade']])

# Histogram of 'yr_built' and 'yr_renovated' (not in report)
data['yr_built'].plot.hist(bins=40)
data['yr_renovated'].plot.hist(bins=40)

# KDE of each column
data['yr_built'].plot.kde()
data['yr_renovated'].plot.kde()

# Simple statistic information of yr_built
data['yr_built'].describe()

# Convert to new data set ---- yr_built
yr_built_new = []
for year in data['yr_built']:
    if year < 1951:
        yr_built_new.append(1)
    elif year >= 1951 and year < 1975:
        yr_built_new.append(2)
    elif year >= 1975 and year < 1997:
        yr_built_new.append(3)
    else:
        yr_built_new.append(4)

yr_built_new = pd.Series(yr_built_new)        
data['new_yr_built'] = yr_built_new
data['new_yr_built'].plot.kde()   

# correlation between price and new_yr_built
corr = data[['price', 'new_yr_built']].corr()
print(corr)

# Convert to new data set - yr_renovated
yr_renovated_new = []
for year in data['yr_renovated']:
    if year != 0:
        yr_renovated_new.append(1)
    else:
        yr_renovated_new.append(0)

yr_renovated_new = pd.Series(yr_renovated_new)
data['new_yr_renovated'] = yr_renovated_new
data['new_yr_renovated'].plot.kde()

# correlation between price and new_yr_renovated
data[['price','new_yr_renovated']].corr()


# for linear regression

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg_built = LinearRegression()
reg_result = LinearRegression()

X = data[['new_yr_renovated','new_yr_built','bedrooms', 'sqft_lot', 'bathrooms', 'sqft_living','waterfront', 'view', 'condition', 'grade']] # input matrix
# X = data[['sqft_lot', 'bedrooms', 'sqft_above', 'bathrooms', 'sqft_living','waterfront', 'view', 'condition', 'grade','yr_built','yr_renovated']] 
y = data['price'] #output target

reg.fit(X,y)
reg.score(X,y)

reg_built.fit(X,y)
reg_built.score(X,y) 

reg_result.fit(X,y)
reg_result.score(X,y)

vif = 1/(1-reg.score(X,y))
print(vif)

y_pred = reg.predict(X)
resid = y-y_pred

resid.plot.kde()

import numpy as np

# convert to log_price
y_new = np.log(y)
reg_result.fit(X,y_new)

data['log_price'] = np.log(data['price'])
data['log_price'].plot.kde()
data['price'].plot.kde()


reg.coef_ 
reg.intercept_

reg_result.coef_
reg_result.intercept_







        





########## just practice ########## 
yr_built_new = []
for year in data['yr_built']:
    if year < 1920:
        yr_built_new.append(1)
    elif year >= 1920 and year <1940:
        yr_built_new.append(2)
    elif year >= 1940 and year < 1960:
        yr_built_new.append(3)
    elif year >= 1960 and year < 1980:
        yr_built_new.append(4)
    elif year >= 1980 and year < 2000:
        yr_built_new.append(5)
    else:
        yr_built_new.append(6)

yr_built_new = pd.Series(yr_built_new)        
data['yr_built_categorized'] = yr_built_new
                
data['yr_built_categorized'].plot.kde()   

data[['price', 'yr_renovated']].corr()
print(corr)




from statsmodels.formula.api import ols

data_2 = pd.DataFrame({'price': data['price'],'bedrooms': data['bedrooms'], 'sqft_lot': data['sqft_lot'], 'bathrooms': data['bathrooms'],
                       'sqft_living': data['sqft_living'], 'waterfront': data['waterfront'], 'view': data['view'],
                        'condition': data['condition'], 'grade': data['grade']})
model = ols("price ~ bedrooms + sqft_lot + bathrooms + sqft_living + waterfront + view + condition + grade ", data_2).fit()
print (model.summary())

data_3 = pd.DataFrame({'price': data['price'],'bathrooms': data['bathrooms'],
                       'sqft_living': data['sqft_living'], 'waterfront': data['waterfront'], 'view': data['view'],
                        'condition': data['condition'], 'grade': data['grade']})





new_year = []
for yr in range(len(data)):
    if data['yr_renovated'][yr] != 0:
        ny = 2015 - data['yr_renovated'][yr]
        new_year.append(ny)
    elif data['yr_renovated'][yr] == 0:
        ny1 = 2015 - data['yr_built'][yr]
        new_year.append(ny1)

new_year = pd.Series(new_year)

data['new_year'] = new_year
data['new_year'].describe()
data['new_year'].plot.kde()

yr_built_new = []
for year in data['new_year']:
    if year < 16:
        yr_built_new.append(4)
    elif year >= 16 and year < 38:
        yr_built_new.append(3)
    elif year >= 38 and year < 61:
        yr_built_new.append(2)
    else:
        yr_built_new.append(1)

yr_built_new = pd.Series(yr_built_new)        
data['yr_built_categorized'] = yr_built_new
                
data['yr_built_categorized'].plot.kde()   


corr = data[['price', 'new_year']].corr()


for column in data.columns:
    data['sqft_living'].plot.kde() 


import matplotlib.pyplot as plt
scat = plt.scatter(data['price'], data['bedrooms'])
plt.xlabel('price', fontsize = 16)
plt.ylabel('bedrooms', fontsize = 16)