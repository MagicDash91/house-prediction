%%writefile house.py
import streamlit as st
import pandas as pd
import numpy as np


st.header('House Price Prediction Using Random Forest Regressor')

df = pd.read_csv('HousingPrices-Amsterdam-August-2021.csv')
df2 = df.drop(columns=['Unnamed: 0', 'Address','Zip'])
df2['Price'].fillna(df2['Price'].median(), inplace = True)
import scipy.stats as stats
z = np.abs(stats.zscore(df2))
data_clean = df2[(z<3).all(axis = 1)] #print all of rows that have z<3 (z score below 3)

X = data_clean.drop('Price', axis=1)
y = data_clean['Price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(random_state=0)
regr.fit(X_train, y_train)



area = st.number_input('House Area in m2')
room = st.number_input('Rooms in the hhouse')
lon = st.number_input('Longitude')
lat = st.number_input('Latitude')


Xnew3 = [[area,room,lon,lat]]

y_pred_prob4 = regr.predict(Xnew3)
hasil = y_pred_prob4.astype(int)
hasil2 = (str(hasil).lstrip('[').rstrip(']'))
st.write("Price Prediction")
st.info(hasil2)
