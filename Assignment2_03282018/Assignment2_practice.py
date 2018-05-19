# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 20:33:04 2018

@author: Justin
"""


import pandas as pd
data = pd.read_csv(r"C:\Users\김성제\Google Drive\공부\4학년 1학기\Business Analytics\Data set\Week2\\kc_house_data.csv")

zip_vc = data['zipcode'].value_counts()
long_vc = data['long'].value_counts()
lat_vc = data['lat'].value_counts()



data['zipcode'][data['lat']==-122.232]
data['zipcode'].plot.kde()
data['long'].plot.kde()
data['lat'].plot.kde()

data['yr_built'][data['zipcode']==98004].mean()

