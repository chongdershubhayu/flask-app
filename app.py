from flask import Flask, request, render_template, redirect
from onnxruntime import InferenceSession
import yfinance as yfin
from pandas_datareader import data as pdr
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib.figure import Figure
import base64
from io import BytesIO
import logging
import requests
import json
import os


INFERENCE_ENDPOINT = os.environ.get("MODEL_URL")
#INFERENCE_ENDPOINT = "https://stock-predict-model-stock-predict.apps.rosa-9m6tt.m01r.p1.openshiftapps.com/v2/models/stock-predict-model/infer"

app = Flask(__name__)

@app.route('/')
def form():
    stock = ["IBM", "AAPL", "MSFT"]
    past_duration = ["6mo", "1y"]
    #future_duration = ["30","40"]
    return render_template('form.html', past_duration=past_duration, stock=stock)

@app.route('/data', methods = ['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        form_data = request.form
        dates = request.form
        app.logger.info(request.form)
        df = yfin.download(tickers=form_data['ticker'], period=form_data['past_duration'])
        if df.empty:
            app.logger.error(f"No data returned for {form_data['ticker']} with period {form_data['past_duration']}")
        else:
            print(df.head())  # Log the first few rows of the stock data
        dataset = df['Close'].fillna(method='ffill')
        print(dataset.head())  # Check the dataset after filling missing values
        dataset = dataset.values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(dataset)
        dataset = scaler.transform(dataset)
        # generate the input and output sequences
        n_lookback = 60  # length of input sequences (lookback period)
        #n_forecast = int(form_data['future_duration'])  # length of output sequences (forecast period)
        n_forecast = 30
        X = []
        Y = []
        for i in range(n_lookback, len(dataset) - n_forecast + 1):
            X.append(dataset[i - n_lookback: i])
            Y.append(dataset[i: i + n_forecast])
        X = np.array(X)
        Y = np.array(Y)
        # generate the forecasts
        X_ = dataset[- n_lookback:]  # last available input sequence
        X_ = X_.reshape(1, n_lookback, 1)
        X = X_.astype(np.float32)
        X = X.tolist()
        json_data = {
                "inputs": [
                {
                "name": "lstm_input",
                "datatype": "FP32",
                "shape": [1,60,1],
                "data": X
                }
            ]
        }
        response = requests.post(INFERENCE_ENDPOINT, json=json_data, verify=False)
        result = response.json()
        print(result)
        result_data = result['outputs'][0]['data']
        Y_ = np.array(result_data).reshape(-1, 1)
        Y_ = scaler.inverse_transform(Y_)
        # organize the results in a data frame
        df_past = df[['Close']].reset_index()
        df_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
        print("Check if the 'Actual' column is populated correctly")
        print(df_past.head())  # Check if the 'Actual' column is populated correctly
        #df_past['Date'] = pd.to_datetime(df_past['Date'])
        df_past['Date'] = pd.to_datetime(df_past['Date'], errors='coerce')
        print("line no 87")
        print(df_past['Date'])
        
        # Step 2: Handle missing or invalid dates, if any.
        if df_past['Date'].isna().any():
            app.logger.warning("Some entries in 'Date' column are invalid and have been converted to NaT.")
            df_past = df_past.dropna(subset=['Date'])
        
        # Step 3: Make sure that the last date in df_past is valid
        last_date = df_past['Date'].iloc[-1]
        if pd.isna(last_date):
            return "Error: The last date in df_past is invalid!"
        df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
        #df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
        df_future['Date'] = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_forecast)
        if len(Y_.flatten()) != n_forecast:
            return f"Error: The length of forecast data ({len(Y_.flatten())}) does not match n_forecast ({n_forecast})"
        df_future['Forecast'] = Y_.flatten()
        df_future['Actual'] = np.nan
        #result = pd.concat([df_past, df_future])
        result = pd.concat([df_past, df_future], ignore_index=False)
        print("Print result data line no 117")
        print(result)
        #result = result.set_index('Date')
        # Step 9: Sort the concatenated results by Date (ensure correct chronological order)
        result = result.sort_values(by='Date').reset_index(drop=True)
        print("Print result data line no 122")
        print(result)
        # Final check for NaN values
        print("NaN values in 'Actual' column:", result['Actual'].isna().sum())
        print("NaN values in 'Forecast' column:", result['Forecast'].isna().sum())
        print("Check if both actual and forecast data are present")
        print(result.head())  # Check if both actual and forecast data are present
        # Generate the figure **without using pyplot**.
        fig = Figure()
        ax = fig.subplots()
        fig.suptitle(form_data['ticker'])
        print("Check 'Actual' data")
        print(result['Actual'].head())  # Check 'Actual' data
        print("Check 'Forecast' data")
        print(result['Forecast'].head())  # Check 'Forecast' data
        #ax.plot(results)
        # Plot both Actual and Forecast columns
        #ax.plot(result.index, result['Actual'], label='Actual', color='blue')
        ax.plot(result.index, result['Forecast'], label='Forecast', color='red')
        ax.legend()  # Add a legend to distinguish Actual vs Forecast
        # Save it to a temporary buffer.
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0) # Reset the pointer to the beginning of the buffer
        # Embed the result in the html output.
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return f"<img src='data:image/png;base64,{data}'/>"
        #return render_template('data.html',form_data = form_data)

@app.route('/health', methods=['GET'])
def health():
    """Return service health"""
    return 'ok'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='9000')
    
