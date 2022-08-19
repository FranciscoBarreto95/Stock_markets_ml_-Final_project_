#In order for consistency we will use the directly retrieved from Yahoo 
 from gettext import install


!pip install matplotlib
!pip install seaborn
#libraries needed for seeing data and surface data exploration
from modulefinder import IMPORT_NAME
import pandas as pd 
import numpy as np 
import matplotlib
#import matplotlib.pyplot as plt
import seaborn as sns 
import yfinance as yf
from datetime import datetime
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
        st.markdown('<h1 style="color: #32c3ff;">Stocks tendency prediction</h1>',
                        unsafe_allow_html=True)


#st.title('Stocks tendency prediction')
#st.markdown("""Provides the tendency prediction of the Security  
#                Predicts the close price of the stock for the current day""")

with st.expander("How to use:  Use when the market is open.  "):
    st.write("""Provides the tendency prediction of the Security  
    Predicts the close price of the stock for the current day""")


##### Algo part 


title_1 = st.text_input("Please Insert the ticker from Yahoo Finance of your security:")
st.write('The current ticker is', title_1)
Ticker_input_1 = title_1.upper()
title = yf.Ticker(Ticker_input_1)
title_hist = title.history(period="max").reset_index()
df = title_hist

#lets plot all the open price 
fig1= px.line(title_hist,x="Date", y= ["Open", "Close"] , title="Security Open and Close price")
st.plotly_chart(fig1, use_container_width=True)

from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df["Date"]=pd.to_datetime(df.Date,format="%Y-%m-%d")
df.index=df['Date']

#fig2=px.line(df["Close"],title='Close Price history')
#st.plotly_chart(fig2, use_container_width=True)

from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
data=df.sort_index(ascending=True,axis=0)
new_dataset=pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])
for i in range(0,len(data)):
    new_dataset["Date"][i]=data['Date'][i]
    new_dataset["Close"][i]=data["Close"][i]
new_dataset.index=new_dataset.Date
new_dataset.drop("Date",axis=1,inplace=True)
#SPLITING THE DATASET 
final_dataset=new_dataset.values
train_data=final_dataset[0:987,:]
valid_data=final_dataset[987:,:]
#SCALLER
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(final_dataset)
x_train_data,y_train_data=[],[]
for i in range(60,len(train_data)):
    x_train_data.append(scaled_data[i-60:i,0])
    y_train_data.append(scaled_data[i,0])
#SPLITING THE DATA TRAINTEST SPLIT
x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)

x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))

#LSTM MODEL APPLICATION 
lstm_model=Sequential()
lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

lstm_model.compile(loss='mean_squared_error',optimizer='adam')
lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose=2)

inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
inputs_data=inputs_data.reshape(-1,1)
inputs_data=scaler.transform(inputs_data)

X_test=[]
for i in range(60,inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
Y_pred_train = lstm_model.predict(x_train_data)
Y_pred_train=scaler.inverse_transform(Y_pred_train)

closing_price=lstm_model.predict(X_test)
closing_price=scaler.inverse_transform(closing_price)

train_data=new_dataset[:987]
valid_data=new_dataset[987:]
valid_data['Predictions']=closing_price

from sklearn.metrics import mean_squared_error
trainScore = np.sqrt(mean_squared_error(y_train_data, Y_pred_train))
st.text('Train Score: %.2f RMSE' % (trainScore))
test = np.sqrt(mean_squared_error(valid_data["Close"], valid_data["Predictions"]))
st.text('Test Score: %.2f RMSE' % (test))
# Result table tweeks
valid_data["diff"]=valid_data["Close"]-valid_data["Predictions"]
valid_data.reset_index(inplace=True)

# a dedicated single loader 
with hc.HyLoader('Now doing loading',hc.Loaders.pulse_bars,):
    time.sleep(10)

# final Graph to display 
fig3 = go.Figure([
    go.Scatter(
        name='Close Price',
        x=valid_data['Date'],
        y=valid_data['Close'],
        mode='lines',
        marker=dict(color="#ff0000"),
        line=dict(width=1),
        showlegend=True
    ),
    
    go.Scatter(
        name='Predictions',
        x=valid_data['Date'],
        y=valid_data['Predictions'],
        marker=dict(color="#4169e1"),
        line=dict(width=1),
        mode='lines',
        fillcolor='rgba(0, 255, 0, 0.3)',
        fill='tonexty',
        showlegend=True
    ),
    go.Scatter(
        name='Diff Value',
        x=valid_data['Date'],
        y=valid_data['diff'],
        mode='lines',
        marker=dict(color="#006400"),
        line=dict(width=1),
        showlegend=True
    ),
])
fig3.update_layout(
    xaxis_title='Date',
    yaxis_title='Price',
    title='Prediction vs Close Price',
    hovermode="x"
)
st.plotly_chart(fig3, use_container_width=True)
# Table with last results 
st.text('Data Overview - latest predictions')
st.table(valid_data.tail())
## To avoid values to cross to pages 
for key in st.session_state.keys():
    del st.session_state[key]