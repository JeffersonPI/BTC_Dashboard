import pandas as pd
import numpy as np
import requests
from datetime import datetime


class BTCModel:
    def __init__(self, model, features, df_base):
        self.model = model
        self.features = features
        self.df_base = df_base.copy()

    
    # GET LIVE BTC PRICE
    
    def get_live_btc(self):
        try:
            url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
            r = requests.get(url, timeout=5)
            return float(r.json()["price"])
        except:
            return None

    # PREDICTION
    
    def predict(self, df, price):
        df = df.copy()

        df["lag_1"] = df["Actual"].shift(1)
        df["lag_2"] = df["Actual"].shift(2)
        df["lag_3"] = df["Actual"].shift(3)

        df["rolling_mean_7"] = df["Actual"].rolling(7).mean()
        df["rolling_std_7"] = df["Actual"].rolling(7).std()

        df["price_change"] = df["Actual"].pct_change()
        df["volatility"] = df["price_change"].rolling(7).std()

        df["momentum_7"] = df["Actual"] - df["Actual"].shift(7)

        df["rolling_max_7"] = df["Actual"].rolling(7).max()
        df["rolling_min_7"] = df["Actual"].rolling(7).min()

        df_feat = df.dropna()

        if not df_feat.empty:
            X = df_feat[self.features].iloc[-1:]
            ret = self.model.predict(X)[0]
            pred_price = price * (1 + ret)
            pred_price = np.clip(pred_price, price * 0.97, price * 1.03)
            return pred_price

        return price

    
    # SIGNAL
    
    def generate_signal(self, pred, price):
        ret = (pred - price) / price

        if ret > 0.002:
            return "BUY"
        elif ret < -0.002:
            return "SELL"
        else:
            return "HOLD"

    
    # MAIN FUNCTION 

    def get_live_data(self):
        price = self.get_live_btc()

        if price is None:
            return None, None

        df_live = self.df_base.copy()

        new_row = pd.DataFrame({
            "time": [datetime.now()],
            "Actual": [price]
        })

        df_live = pd.concat([df_live, new_row], ignore_index=True)
        df_live = df_live.tail(50)

        pred = self.predict(df_live, price)
        sig = self.generate_signal(pred, price)

        df_live["price"] = df_live["Actual"]
        df_live["predicted"] = pred
        df_live["signal"] = sig

        latest = df_live.iloc[-1]

        return df_live, latest