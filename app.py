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
        dataset = df['Close'].fillna(method='ffill')
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
        df_past['Date'] = pd.to_datetime(df_past['Date'])
        df_past['Forecast'] = np.nan
        if not df_past.empty:
            df_past.at[df_past.index[-1], 'Forecast'] = df_past['Actual'].iloc[-1]
       else:
            app.logger.error("DataFrame df_past is empty!")
            return "Error: No data to predict."
        #df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]
        df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
        df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
        df_future['Forecast'] = Y_.flatten()
        df_future['Actual'] = np.nan
        results = pd.concat([df_past, df_future])
        results = results.set_index('Date')
        # Generate the figure **without using pyplot**.
        fig = Figure()
        ax = fig.subplots()
        fig.suptitle(form_data['ticker'])
        ax.plot(results)
        # Save it to a temporary buffer.
        buf = BytesIO()
        fig.savefig(buf, format="png")
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
    
