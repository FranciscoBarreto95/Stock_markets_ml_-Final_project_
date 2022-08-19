#In order for consistency we will use the directly retrieved from Yahoo 
!pip install matplotlib
!pip install seaborn
#libraries needed for seeing data and surface data exploration
import pandas as pd 
import numpy as np 
#import matplotlib.pyplot as plt
import seaborn as sns 
import yfinance as yf
from datetime import datetime
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import warnings
### Streamlit imports needed
import hydralit_components as hc
import time
import streamlit as st
import streamlit.components.v1 as components
import mpld3
from PIL import Image
title_container = st.container()
col1, col2 = st.columns([2, 18])
image = Image.open('Ironhack_logo_2.png')
with title_container:
    with col1:
        st.image(image, width=70)
    with col2:
        st.markdown('<h1 style="color: #32c3ff;">Stocks Tendency & Price Prediction</h1>',
                        unsafe_allow_html=True)

#st.title('Stocks tendency prediction')
#st.markdown("""Provides the tendency prediction of the Security  
#                Predicts the close price of the stock for the current day""")

with st.expander("How to use:  Use when the market is open.  "):
    st.write("""Provides the tendency prediction of the Security  
    Predicts the close price of the stock for D & D+1""")


#Input Ticker element
title_2 = st.text_input("Please Insert the ticker from Yahoo Finance of your security:")
st.write('The current ticker is', title_2)
st.title('Prophet - Stock Yearly Seasonality effect')

#### Algo part 
Ticker_input_2 = title_2.upper()
title = yf.Ticker(Ticker_input_2)
title_hist = title.history(period="max").reset_index()
df = title_hist

#lets plot all the open price 
fig1= px.line(title_hist,x="Date", y= ["Open", "Close"] , title="Stock Open and Close price")
st.plotly_chart(fig1, use_container_width=True)

#defining the new df 
new_df = df[['Date', 'Close']]

# Multiplicative effect - Prophet
new_names = {"Date": "ds","Close": "y"}
new_df.rename(columns=new_names, inplace=True)

# Seasonality effect - Prophet

model_2 = Prophet(yearly_seasonality = True)

model_2.fit(new_df)
future_2 = model_2.make_future_dataframe(periods = 365)
forecast_2 = model_2.predict(future_2)

#Forecast for today
st.markdown("""**Price Forecast for today**""")
Today = datetime.today().strftime('%Y-%m-%d')
st.text(forecast_2[forecast_2['ds'] == Today]['yhat'].item())

#Forecast for next day
st.markdown("""**Price Forecast for tomorrow**""")
next_day = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')
st.text(forecast_2[forecast_2['ds'] == next_day]['yhat'].item())

# Show Graph 
fig_2 =plot_plotly(model_2, forecast_2)
st.plotly_chart(fig_2, use_container_width=True)
from prophet.diagnostics import cross_validation
df_cv_2 = cross_validation(model_2, initial='730 days', period='180 days', horizon = '365 days')


from prophet.diagnostics import performance_metrics
df_p_2 = performance_metrics(df_cv_2)
table_2 =df_p_2[["horizon","rmse","mae"]].loc[:5]
st.markdown("**Perfomance Metrics - Seasonality Model**")
st.code(table_2)
## To avoid values to cross to pages 
for key in st.session_state.keys():
    del st.session_state[key]