import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
from live_data import BTCModel
import joblib
import warnings
warnings.filterwarnings("ignore")



st.set_page_config(page_title="BTC Prediction Dashboard", layout="wide")

st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)

st_autorefresh(interval=30000, key="refresh")


# LOAD DATA (CSV)
df_plot = pd.read_csv("btc_predictions.csv")
df_plot["Date"] = pd.to_datetime(df_plot["Date"])
df_plot = df_plot.set_index("Date")

model = joblib.load("model_rf_price.pkl")
features = joblib.load("features.pkl")

df_live_base = df_plot.reset_index()[["Date", "Actual"]].tail(100)
df_live_base.columns = ["time", "Actual"]

if "df_live" not in st.session_state or st.session_state.df_live is None:
    st.session_state.df_live = df_live_base.copy()

btc_model = BTCModel(model, features, df_live_base)

df_live_temp, latest_live = btc_model.get_live_data()

if df_live_temp is not None:
    st.session_state.df_live = pd.concat(
        [st.session_state.df_live, df_live_temp.tail(1)],
        ignore_index=True
    ).tail(50)

    df_live = st.session_state.df_live
else:
    df_live = None
    
if df_live is None:
    st.warning("⚠️ Live data unavailable (API issue)")
    

    
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Chart Detail", "🧠 Insight"])

# SAFETY CHECK
if df_plot.empty:
    st.error("Data kosong! Cek CSV.")
    st.stop()

# LOAD CSS
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# LOAD HTML HEADER
def load_html():
    html_path = os.path.join(os.path.dirname(__file__), "components.html")
    with open(html_path) as f:
        components.html(f.read(), height=120)

load_html()


    
with tab1:

    # Date Range Picker
    st.subheader("📅 Filter Date")

    start_date = st.date_input("Start Date", df_plot.index.min())
    end_date = st.date_input("End Date", df_plot.index.max())
    
    zoom = st.sidebar.slider("Zoom Data", 50, 300, 120)
    
    df_zoom = df_plot.tail(zoom).copy()
    df_filtered = df_zoom.loc[start_date:end_date]
    df_filtered = df_filtered.copy()
    
    if df_filtered.empty:
        st.warning("Data tidak tersedia pada rentang tanggal ini")
        st.stop()
    
    # SIDEBAR
    st.sidebar.header("⚙️ Settings")
    
    initial_balance = st.sidebar.number_input(
    "💰 Initial Capital ($)",
    min_value=100,
    max_value=100000,
    value=1000,
    step=100
    )

    models = st.sidebar.multiselect(
        "Select Model",
        ["Actual", "LR", "RF", "RF_OnChain"],
        default=["Actual", "RF_OnChain"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.write("📊 Real-time BTC Dashboard")
    st.sidebar.write("🔄 Auto update every 3s")


    # SIGNAL
    df_filtered["Return_Pred"] = df_filtered["RF_OnChain"].pct_change().fillna(0)

    def signal(x):
        if x > 0.002:
            return "BUY"
        elif x < -0.002:
            return "SELL"
        else:
            return "HOLD"

    df_filtered["Signal"] = df_filtered["Return_Pred"].apply(signal)


    # TOP METRICS
    col1, col2, col3, col4, col5 = st.columns(5)

    if latest_live is not None:
        col1.metric("🔥 Live BTC", f"${latest_live['price']:,.0f}")

        # DELTA
        delta = latest_live["predicted"] - latest_live["price"]

        col2.metric(
            "🤖 Live Prediction",
            f"${latest_live['predicted']:,.0f}",
            delta=f"{delta:,.0f} USD"
        )
        
        signal_live = latest_live["signal"]
        
        col3.metric("📢 Live Signal", signal_live)

        signal_text = latest_live["signal"]

    else:
        col1.metric("💰 BTC Price", f"${df_filtered['Actual'].iloc[-1]:,.0f}")
        col2.metric("🤖 Prediction", f"${df_filtered['RF_OnChain'].iloc[-1]:,.0f}")
        col3.metric("📢 Signal", df_filtered["Signal"].iloc[-1])

        signal_text = df_filtered["Signal"].iloc[-1]


    # 🎨 SIGNAL WARNA (SATU SAJA)
    signal_color = {
        "BUY": "green",
        "SELL": "red",
        "HOLD": "gray"
    }

    st.markdown(
        f"<h3 style='color:{signal_color.get(signal_text, 'white')}'>Signal: {signal_text}</h3>",
        unsafe_allow_html=True
    )

    col4.metric("📊 Data Points", len(df_filtered))
    
    strength = df_filtered["Return_Pred"].iloc[-1] * 100
    col5.metric("⚡ Strength", f"{strength:.2f}%")
        
    # Chart
    st.subheader("📈 Interactive Chart")

    fig = go.Figure()

    for col in models:
        fig.add_trace(go.Scatter(
            x=df_filtered.index,
            y=df_filtered[col],
            mode='lines',
            name=col,
            line=dict(width=3 if col == "Actual" else 2)
        ))

    # BUY signal
    buy = df_filtered[df_filtered["Signal"] == "BUY"]
    fig.add_trace(go.Scatter(
        x=buy.index,
        y=buy["Actual"],
        mode='markers',
        marker=dict(color='green', size=10, symbol='triangle-up'),
        name="BUY",
        hovertemplate='BUY<br>Date: %{x}<br>Price: %{y:,.0f}'
    ))
    # SELL signal
    sell = df_filtered[df_filtered["Signal"] == "SELL"]
    fig.add_trace(go.Scatter(
        x=sell.index,
        y=sell["Actual"],
        mode='markers',
        marker=dict(color='red', size=10, symbol='triangle-down'),
        name="SELL",
        hovertemplate='SELL<br>Date: %{x}<br>Price: %{y:,.0f}'  
    ))

    fig.update_layout(
        template="plotly_dark",
        hovermode="closest",
        height=500,
        xaxis=dict(rangeslider=dict(visible=False)),
        title="BTC Price Prediction & Trading Signals",
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    st.plotly_chart(fig, use_container_width=True)
    
    ## LIVE PANEL
    if latest_live is not None:
        st.markdown("## ⚡ Live Market Data")

        colL1, colL2, colL3 = st.columns(3)

        colL1.metric("Live Price", f"${latest_live['price']:,.0f}")
        delta_live = latest_live["predicted"] - latest_live["price"]

        colL2.metric(
            "Predicted Price",
            f"${latest_live['predicted']:,.0f}",
            delta=f"{delta_live:,.0f}"
        )
        
        colL3.markdown(
            f"<h3 style='color:{signal_color.get(latest_live['signal'],'white')}'>Signal: {latest_live['signal']}</h3>",
            unsafe_allow_html=True
        )
        
        st.caption(f"Last update: {latest_live['time']}")
        
    ## LIVE TREND CHART
    if df_live is not None and "signal" in df_live.columns:
        st.markdown("## 📡 Live BTC Trend")

        fig_live = go.Figure()

        fig_live.add_trace(go.Scatter(
            x=df_live["time"],
            y=df_live["price"],
            name="Price",
            mode='lines+markers'
        ))

        # SIGNAL MARKER
        buy = df_live[df_live["signal"] == "BUY"]
        sell = df_live[df_live["signal"] == "SELL"]

        fig_live.add_trace(go.Scatter(
            x=buy["time"],
            y=buy["price"],
            mode='markers',
            marker=dict(color='green', size=10, symbol='triangle-up'),
            name="BUY"
        ))

        fig_live.add_trace(go.Scatter(
            x=sell["time"],
            y=sell["price"],
            mode='markers',
            marker=dict(color='red', size=10, symbol='triangle-down'),
            name="SELL"
        ))
        
        fig_live.add_trace(go.Scatter(
            x=df_live["time"],
            y=df_live["predicted"],
            name="Prediction",
            mode='lines'
        ))

        fig_live.update_layout(
            template="plotly_dark",
            title="📡 Live BTC Price Movement",
            xaxis_title="Time",
            yaxis_title="Price ($)"
            )
        
        if latest_live is not None:
            fig_live.add_hline(
                y=latest_live["price"],
                line_dash="dot",
                line_color="yellow",
                annotation_text="Current Price"
                )
        
        fig_live.update_traces(
        hovertemplate="Time: %{x}<br>Price: $%{y:,.0f}"
        )  
         
        st.markdown("## ⚡ Real-Time BTC Chart")
        st.plotly_chart(fig_live, use_container_width=True)
        
        with st.expander("📄 Live Data (Debug)"):
            st.dataframe(df_live.tail(10))


    # PROFIT SIMULATION
    st.markdown("## 💰 Trading Simulation")

    balance = initial_balance
    position = 0
    portfolio = []

    for i in range(len(df_filtered)):
        s = df_filtered["Signal"].iloc[i]
        price = df_filtered["Actual"].iloc[i]

        if s == "BUY" and position == 0:
            position = balance / price
            balance = 0

        elif s == "SELL" and position > 0:
            balance = position * price
            position = 0

        total = balance + (position * price)
        portfolio.append(total)

    df_filtered["Portfolio"] = portfolio

    final = portfolio[-1]
    profit = (final - initial_balance) / initial_balance * 100

    colA, colB = st.columns(2)
    colA.metric("💵 Final Balance", f"${final:,.2f}")
    colB.metric(
        "📊 Profit (%)",
        f"{profit:.2f}%",
        delta=f"{profit:.2f}%"
    )

    # PORTFOLIO CHART
    fig_port = go.Figure()

    fig_port.add_trace(go.Scatter(
    x=df_filtered.index,
    y=df_filtered["Portfolio"],
    name="Portfolio"
    ))
    
    fig_port.update_traces(
    hovertemplate="Date: %{x}<br>Balance: $%{y:,.2f}"
    )

    fig_port.update_layout(
    template="plotly_dark",
    title="📊 Portfolio Performance",
    xaxis_title="Date",
    yaxis_title="Balance ($)"
    )

    fig_port.add_hline(
    y=initial_balance,
    line_dash="dash",
    line_color="white",
    annotation_text="Initial Capital",
    annotation_position="top left"
    )
    
    st.markdown('<div class="section-card"><div class="section-title">💰 Portfolio Performance</div></div>', unsafe_allow_html=True)
    st.plotly_chart(fig_port, use_container_width=True)
    
    # TABLE
    st.markdown("## 📄 Recent Signals")
    st.dataframe(
        df_filtered[["Actual", "RF_OnChain", "Signal"]]
        .tail(10)
        .style.format({
            "Actual": "{:,.0f}",
            "RF_OnChain": "{:,.0f}"
        })
    )

    st.caption("⚡ Powered by Machine Learning + Real-Time Data")
    
with tab2:
    st.subheader("📈 Full Chart")

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot["Actual"],
        name="Actual"
    ))

    fig2.add_trace(go.Scatter(
        x=df_plot.index,
        y=df_plot["RF_OnChain"],
        name="Prediction"
    ))

    fig2.update_layout(template="plotly_dark")

    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("🧠 Insight")

    st.write("""
    1. Model mampu mengikuti tren harga Bitcoin dengan baik, namun masih memiliki keterbatasan dalam menangkap volatilitas ekstrem.

    2. Data on-chain tidak memberikan peningkatan signifikan dalam jangka pendek, namun tetap memberikan insight terhadap aktivitas jaringan.

    3. Sistem ini mengintegrasikan data historis dan real-time untuk menghasilkan prediksi dan sinyal trading secara dinamis.

    4. Dashboard ini dapat digunakan sebagai alat bantu analisis dan simulasi strategi trading.
    """)