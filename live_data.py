import pandas as pd
import numpy as np
import requests



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
        df = df.sort_values("time").reset_index(drop=True)
        
        if len(df) < 10:
            return price

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
        
        for col in self.features:
            if col not in df.columns:
                df[col] = np.nan

        df_feat = df.dropna(subset=self.features)

        if not df_feat.empty:
            X = df_feat[self.features].iloc[-1:]
            
            if X.isnull().values.any():
                return price
            
            ret = self.model.predict(X)[0]
            ret = np.clip(ret, -0.05, 0.05)
            return price * (1 + ret)

        return price

    
    # SIGNAL
    
    def generate_signal(self, pred, price):
        ret = (pred - price) / price

        if ret > 0.001:
            return "BUY"
        elif ret < -0.001:
            return "SELL"
        else:
            return "HOLD"

    
    # MAIN FUNCTION 

    def get_live_data(self):
        price = self.get_live_btc()

        if price is None:
            return None, None

        df_live = self.df_base.copy()
        df_live["time"] = pd.to_datetime(df_live["time"]) 
        df_live = df_live.sort_values("time").reset_index(drop=True)
        df_live["Actual"] = df_live["Actual"].astype(float)

        new_row = pd.DataFrame({
            "time": [pd.Timestamp.now()],
            "Actual": [price]
        })

        df_live = pd.concat([df_live, new_row], ignore_index=True)

        df_live["time"] = pd.to_datetime(df_live["time"]).dt.floor("min")
        df_live = df_live.drop_duplicates(subset="time", keep="last") 
        df_live = df_live.tail(100)

        pred = self.predict(df_live, price)
        sig = self.generate_signal(pred, price)

        df_live["price"] = df_live["Actual"]
        
        df_live["predicted"] = np.nan
        df_live.loc[df_live.index[-1], "predicted"] = pred
        
        df_live["signal"] = None
        df_live.loc[df_live.index[-1], "signal"] = sig

        latest = df_live.iloc[-1]

        return df_live, latest