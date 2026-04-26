![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![ML](https://img.shields.io/badge/Machine%20Learning-RandomForest-green)

# 🚀 BTC Trading Dashboard (Real-Time ML)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![ML](https://img.shields.io/badge/Machine%20Learning-RandomForest-green)

A **real-time Bitcoin trading dashboard** built with **Streamlit** and powered by **Machine Learning**.  
This system integrates **historical data** and **live market data** to generate **price predictions** and **trading signals (BUY / SELL / HOLD)**.

---

## Features

- **Live Bitcoin Price (API)**
- **Machine Learning Prediction (Random Forest)**
- **Interactive Chart (Plotly)**
- **Trading Signals (BUY / SELL / HOLD)**
- **Trading Simulation (Portfolio Performance)**
- **Auto Refresh System**
- **Custom UI (Dark Trading Theme)**

---

## Machine Learning Model

The system uses:

- **Random Forest Regressor**
- Feature Engineering:
  - Lag features (`lag_1`, `lag_2`, `lag_3`)
  - Rolling mean & standard deviation
  - Price change & volatility
  - Momentum indicators

Prediction output is converted into price and used to generate trading signals.

---

## Tech Stack

- **Python**
- **Streamlit**
- **Pandas / NumPy**
- **Plotly**
- **Scikit-learn**
- **Binance API**

---

## Project Structure
.
├── app.py # Main Streamlit dashboard
├── live_data.py # ML + live data logic (class-based)
├── btc_predictions.csv # Historical dataset
├── model_rf_price.pkl # Trained ML model
├── features.pkl # Feature list
├── styles.css # Custom styling
├── components.html # Custom UI components
├── requirements.txt # Dependencies


---

## How to Run

### 1. Clone repository

git clone https://github.com/your-username/btc-dashboard.git
cd btc-dashboard

### 2. Install dependencies
pip install -r requirements.txt

### 3. Run app
streamlit run app.py

---

## Deployment

This project can be deployed using:

- **Streamlit Community Cloud 

---

## System Workflow

1. Fetch live BTC price from API  
2. Combine with historical data  
3. Generate features  
4. Predict price using ML model  
5. Generate trading signal  
6. Display on dashboard  

---

## Research Context

This project was developed as part of a research study:

> **“Bitcoin Price Prediction Using Machine Learning Based on Historical and On-Chain Data”**

The system demonstrates how machine learning can assist in financial decision-making and trading simulations.

---

## Disclaimer

This project is for **educational and research purposes only**.  
It does **not provide financial advice**.

---

## Author

**Jefferson Iskandar**

---

## Future Improvements

- Candlestick chart (OHLC)  
- Technical indicators (RSI, MACD)  
- Model optimization  
- Database integration  
- Real trading API integration  

---
