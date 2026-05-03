import streamlit as st
import pandas as pd
import numpy as np
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
df_full = pd.read_csv("btc_full.csv")
df_pred = pd.read_csv("btc_predictions.csv")


df_full["Date"] = pd.to_datetime(df_full["Date"])
df_pred["Date"] = pd.to_datetime(df_pred["Date"])

df_full = df_full.set_index("Date")
df_pred = df_pred.set_index("Date")[["RF_OnChain"]]

df_plot = df_full.join(df_pred, how="left")
df_plot = df_plot.sort_index()


if "RF_OnChain" not in df_plot.columns:
    df_plot["RF_OnChain"] = np.nan

df_plot.index = pd.to_datetime(df_plot.index)

model = joblib.load("model_rf_price.pkl")
features = joblib.load("features.pkl")

df_live_base = df_plot.reset_index()[["Date", "Actual"]].tail(200)
df_live_base = df_live_base.rename(columns={"Date": "time"})
df_live_base["time"] = pd.to_datetime(df_live_base["time"])

if "df_live" not in st.session_state or st.session_state.df_live is None:
    st.session_state.df_live = df_live_base.copy()

btc_model = BTCModel(model, features, df_live_base)

df_live_temp, latest_live = btc_model.get_live_data()

if df_live_temp is not None:
    st.session_state.df_live = pd.concat(
        [st.session_state.df_live, df_live_temp.tail(1)],
        ignore_index=True
    ).tail(200)
    
    st.session_state.df_live["time"] = pd.to_datetime(st.session_state.df_live["time"]).dt.floor("min")
    st.session_state.df_live = st.session_state.df_live.drop_duplicates(subset="time")
    
df_live = st.session_state.df_live if "df_live" in st.session_state else None

# historis
df_hist = df_plot.reset_index()[["Date", "Actual"]]
df_hist = df_hist.rename(columns={"Date": "time"})
df_hist["time"] = pd.to_datetime(df_hist["time"])

## cek apakah ada live data
    
if df_live is not None and not df_live.empty:
        df_all = pd.concat([
            df_hist,
            df_live[["time", "Actual"]]
        ], ignore_index=True)   
else:
    df_all = df_hist.copy()

df_all["time"] = pd.to_datetime(df_all["time"], errors="coerce")
df_all = df_all.dropna(subset=["time"])
df_all = df_all.drop_duplicates(subset="time")
df_all = df_all.sort_values("time")

df_rf = df_plot.reset_index()[["Date", "RF_OnChain"]]
df_rf = df_rf.rename(columns={"Date": "time"})

df_all["time"] = pd.to_datetime(df_all["time"]).dt.floor("D")
df_rf["time"] = pd.to_datetime(df_rf["time"]).dt.floor("D")

df_all = df_all.merge(df_rf, on="time", how="left")

df_all["RF_OnChain"] = df_all["RF_OnChain"].where(
    (df_all["time"] >= "2024-01-01") &
    (df_all["time"] <= "2024-12-31")
)

    
if df_live_temp is None:
    st.warning("⚠️ Live data unavailable (API issue)")
    
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Chart Detail", "🧠 Insight"])

# SAFETY CHECK
if df_plot.empty:
    st.warning("No data available for selected date range")
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

    # ambil range data asli
    min_date = df_plot.index.min()
    max_date = df_all["time"].max()

    start_date = st.date_input(
        "Start Date",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )
    
    end_date = st.date_input(
        "End Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )
    
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    df_all = df_all.set_index("time")

    df_filtered = df_all.copy()

    # handle RF_OnChain (karena live gak punya)
    if "RF_OnChain" not in df_filtered.columns:
        df_filtered["RF_OnChain"] = np.nan
    
    #FILTER DATE
    df_filtered = df_filtered[
    (df_filtered.index >= start_date) &
    (df_filtered.index <= end_date)
    ]

    zoom = st.sidebar.slider("Zoom Data", 50, len(df_filtered), len(df_filtered))
    
    
    # SORT + ZOOM
    df_filtered = df_filtered.sort_index()

    if len(df_filtered) > zoom:
        df_filtered = df_filtered.tail(zoom)
    
    df_filtered = df_filtered.dropna(subset=["Actual"])
    
    if df_filtered.empty:
        st.warning("Data tidak tersedia pada rentang tanggal ini")
        st.stop()
    
    if len(df_filtered) < 5:
        st.warning("Data terlalu sedikit untuk analisis")
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
        ["Actual", "RF_OnChain"],
        default=["Actual", "RF_OnChain"]
    )
    models = [m for m in models if m in df_filtered.columns]
    
    st.sidebar.markdown("---")
    st.sidebar.write("📊 Real-time BTC Dashboard")
    st.sidebar.write("🔄 Auto update every 30s")


    # SIGNAL
    df_filtered["Return_Pred"] = (
        df_filtered["RF_OnChain"]
        .pct_change()
        .rolling(5)
        .mean()
    )

    df_filtered["Return_Pred"] = df_filtered["Return_Pred"].fillna(0)


    def signal(x):
                
        if x > 0.001:
            return "BUY"
        elif x < -0.001:
            return "SELL"
        else:
            return "HOLD"

    df_filtered["Signal"] = "HOLD"  # default

    mask = df_filtered["RF_OnChain"].notna()

    df_filtered.loc[mask, "Signal"] = df_filtered.loc[mask, "Return_Pred"].apply(signal)


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

    fig.add_trace(go.Scatter(
        x=df_filtered.index,
        y=df_filtered["Actual"],
        name="Actual",
        line=dict(width=3)
    ))

    # RF_OnChain (HANYA YANG ADA)
    if "RF_OnChain" in models:
        rf = df_filtered[df_filtered["RF_OnChain"].notna()]

        fig.add_trace(go.Scatter(
            x=rf.index,
            y=rf["RF_OnChain"],
            name="RF_OnChain",
            line=dict(width=2)
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

     # HIGHLIGHT PERIODE MODEL (2024)
    train_start = pd.to_datetime("2024-01-01")
    train_end = pd.to_datetime("2024-12-31")

    fig.add_vrect(
        x0=train_start,
        x1=train_end,
        fillcolor="cyan",
        opacity=0.08,
        layer="below",
        line_width=0,
        annotation_text="Model Training Period",
        annotation_position="top left"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("⚠️ RF_OnChain hanya tersedia pada periode training (2024)")
    st.caption("⚠️ RF_OnChain hanya muncul saat model menghasilkan prediksi (tidak selalu dari awal tahun)")

    
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
        
        df_live = df_live.sort_values("time")

        fig_live = go.Figure()
        
        # HISTORICAL
        fig_live.add_trace(go.Scatter(
        x=df_hist["time"],
        y=df_hist["Actual"],
        name="Historical",
        mode='lines',
        line=dict(width=1)
        ))

        # LIVE
        fig_live.add_trace(go.Scatter(
            x=df_live["time"],
            y=df_live["Actual"],
            name="Live",
            mode='lines',
            line=dict(width=2)
        ))
        
        df_live_plot = df_live.copy()
        
        df_live_plot["signal_plot"] = df_live_plot["signal"]

        df_live_plot["signal_plot"] = df_live_plot["signal_plot"].where(
        df_live_plot["signal_plot"] != df_live_plot["signal_plot"].shift()
        )

        df_live_plot["signal_plot"] = df_live_plot["signal_plot"].fillna("")
        
        # SIGNAL MARKER
        buy = df_live_plot[df_live_plot["signal_plot"] == "BUY"]
        sell = df_live_plot[df_live_plot["signal_plot"] == "SELL"]

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
            mode='markers+lines',
            line=dict(color="yellow", dash="dot"),
            connectgaps=True
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
            st.dataframe(df_live_plot.tail(10))


    # 💰 PROFIT SIMULATION
    st.markdown("## 💰 Trading Simulation")

    st.caption("📌 Trading simulation hanya menggunakan data training (2024)")
    st.caption("📌 Menggunakan rule-based strategy untuk efisiensi dan stabilitas sistem")

    # 🔥 PREP DATA
    df_sim = df_plot.reset_index().copy()

    df_sim = df_sim.drop_duplicates(subset="Date")
    df_sim = df_sim.sort_values("Date")
    df_sim = df_sim.rename(columns={"Date": "time"})

    df_sim["time"] = pd.to_datetime(df_sim["time"])
    df_sim = df_sim[df_sim["time"] <= "2024-12-31"]

    # 🔥 LIMIT DATA (biar ringan)
    df_sim = df_sim.tail(365)

    # 🔥 FEATURE (RINGAN)
    df_sim["ret"] = df_sim["Actual"].pct_change().rolling(3).mean()

    # 🔥 INIT
    balance = initial_balance
    position = 0
    portfolio = []
    signals_sim = []

    threshold = 0.001

    # 🔥 LOOP (TANPA ML)
    for i in range(len(df_sim)):
        price = df_sim["Actual"].iloc[i]

        if i > 0:
            r = df_sim["ret"].iloc[i]

            if r > threshold:
                signal = "BUY"
            elif r < -threshold:
                signal = "SELL"
            else:
                signal = "HOLD"
        else:
            signal = "HOLD"

        # 🔥 EXECUTION
        if signal == "BUY" and position == 0:
            position = balance / price
            balance = 0

        elif signal == "SELL" and position > 0:
            balance = position * price
            position = 0

        total = balance + (position * price)
        portfolio.append(total)
        signals_sim.append(signal)

    # 🔥 SAVE RESULT
    df_sim["Portfolio"] = portfolio
    df_sim["Signal"] = signals_sim

    # 🔥 MAP KE CHART (BIAR CONSISTENT)
    df_sim_map = df_sim[["time", "Signal"]].copy()
    df_sim_map["time"] = pd.to_datetime(df_sim_map["time"])

    df_filtered = df_filtered.reset_index()
    df_filtered["time"] = pd.to_datetime(df_filtered["time"])

    df_filtered = df_filtered.merge(
        df_sim_map,
        on="time",
        how="left",
        suffixes=("", "_sim")
    )

    df_filtered["Signal"] = df_filtered["Signal_sim"].fillna(df_filtered["Signal"])
    df_filtered = df_filtered.set_index("time")

    # 🔥 RESULT
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
    x=df_sim["time"],
    y=df_sim["Portfolio"],
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
    st.markdown("## 📄 Recent Signals (Training Data 2024)")

    st.caption("📌 Menampilkan sinyal trading dari hasil simulasi pada periode training (2024)")

    df_recent = df_sim.copy()

    df_recent["time"] = pd.to_datetime(df_recent["time"])

    df_recent = df_recent[
        (df_recent["time"] >= "2024-01-01") &
        (df_recent["time"] <= "2024-12-31")
    ]

    df_recent = df_recent.tail(400)
    
    def color_signal(val):
        if val == "BUY":
            return "color: green"
        elif val == "SELL":
            return "color: red"
        return "color: gray"

    st.dataframe(
        df_recent[["time", "Actual", "Signal"]]
        .sort_values("time", ascending=False)
        .style
        .format({"Actual": "{:,.0f}"})
        .applymap(color_signal, subset=["Signal"])
    )
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
    
    st.write("""
    1. Model mampu mengikuti tren harga Bitcoin dengan baik, namun masih memiliki keterbatasan dalam menangkap volatilitas ekstrem.

    2. Data on-chain tidak memberikan peningkatan signifikan dalam jangka pendek, namun tetap memberikan insight terhadap aktivitas jaringan.

    3. Sistem ini mengintegrasikan data historis dan real-time untuk menghasilkan prediksi dan sinyal trading secara dinamis.

    4. Dashboard ini dapat digunakan sebagai alat bantu analisis dan simulasi strategi trading.
    """)