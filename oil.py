# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 23:25:15 2023

@author: ashwi
"""

import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
from prophet.plot import plot_plotly , plot_components_plotly

###################333### Title and content ############################
st.title('Time Series Forecasting')
st.write('Oil Price Predictions')

##################### Sidebar Header Content #########################
st.sidebar.write('To forecast Oil Prices for next 6 Months from an orginaldata')

data = st.file_uploader('DCOILWTICO',type='csv')

if data is not None:
     appdata = pd.read_csv(data)  #read the data from
     appdata = appdata.resample('D').mean()
     appdata["price"] = appdata["DCOILWTICO"].ffill()
     appdata.rename(columns={"DATE":"ds","DCOILWTICO":"y"},inplace=True)
     appdata['ds'] = pd.to_datetime(appdata['ds'],errors='coerce') 
     st.write(data) #display the data  
     max_date = appdata['ds'].max() #compute latest date in the data 

st.write("SELECT FORECAST PERIOD") #text displayed

periods_input = st.number_input('How many days forecast do you want?',min_value = 1, max_value = 365)

# The minimum number of days a user can select is one, while the maximum is  #365 (yearly forecast) 

if data is not None:
     obj = Prophet() #Instantiate Prophet object
     obj.fit(appdata)  #fit the data 

# Visualize the forecasted data

st.write("VISUALIZE FORECASTED DATA")  
st.write("The following plot shows future predicted values. 'yhat' is the predicted value; upper and lower limits are 80% confidence intervals by default")

if data is not None:
     future = obj.make_future_dataframe(periods=periods_input)
     fcst = obj.predict(future)  #make prediction for the extended data
     forecast = fcst[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
     forecast_filtered =  forecast[forecast['ds'] > max_date]    
     st.write(forecast_filtered)  #Display some forecasted records
     st.write('''The next visual shows the actual (black dots) and predicted (blue line) values over time.''')
     figure1 = obj.plot(fcst) #plot the actual and predicted values
     st.write(figure1)  #display the plot
     st.write('''The next visual shows the actual (black dots) and predicted (blue line) values over time with slidebar chart''')
     figure2 = plot_plotly(obj,fcst)
     st.write(figure2)
     st.write("The following plots show a high level trend of predicted values, day of week trends and yearly trends (if dataset contains multiple years’ data).Blue shaded area represents upper and lower confidence intervals.")
     figure3 = obj.plot_components(fcst) 
     st.write(figure3) 























