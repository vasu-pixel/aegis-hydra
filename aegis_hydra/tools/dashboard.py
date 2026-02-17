import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="AEGIS HYDRA CONTROL CENTER", layout="wide", initial_sidebar_state="collapsed")

# Custom Dark Theme CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #1a1c24; padding: 15px; border-radius: 10px; border-left: 5px solid #00ff00; }
    .metric-row { display: flex; justify-content: space-between; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- UTILITIES ---
@st.cache_data(ttl=1)
def load_signal_data(symbol):
    filename = f"hft_signals_{symbol}.csv"
    if not os.path.exists(filename):
        return pd.DataFrame()
    try:
        # Read last 500 rows for performance
        df = pd.read_csv(filename)
        if df.empty: return df
        df['timestamp'] = pd.to_datetime(df['recv_time'], unit='s')
        return df.tail(500)
    except:
        return pd.DataFrame()

def get_portfolio_data():
    symbols = ["BTCUSD", "ETHUSD", "USDTUSD"]
    data = {}
    for s in symbols:
        df = load_signal_data(s)
        if not df.empty:
            data[s] = df
    return data

# --- HEADER ---
st.title("🛡️ AEGIS HYDRA | TRI-FORCE CONTROL CENTER")
st.markdown("---")

# --- LIVE METRICS ---
portfolio = get_portfolio_data()

if not portfolio:
    st.warning("Waiting for HFT logs... (Run launcher.py to start engines)")
    time.sleep(2)
    st.rerun()

cols = st.columns(len(portfolio))
total_pnl = 0.0

for i, (symbol, df) in enumerate(portfolio.items()):
    last_row = df.iloc[-1]
    pnl = last_row.get('pnl', 0.0)
    total_pnl += pnl
    
    with cols[i]:
        st.metric(label=f"{symbol} PRICE", 
                  value=f"${last_row['mid']:.2f}", 
                  delta=f"{last_row.get('mag', 0.0):.4f} Mag")
        st.caption(f"Latency: {last_row['net_lat']:.2f}ms | Energy: {last_row.get('energy', 0.0):.4f}")

# --- MAIN GRAPH: 3-ASSET NORMALIZED PRICE ---
st.subheader("📊 Multi-Asset Alpha Shield (Normalized Price %)")

fig = go.Figure()

for symbol, df in portfolio.items():
    # Normalize price to % change from start of buffer
    first_price = df['mid'].iloc[0]
    norm_price = (df['mid'] / first_price - 1) * 100
    
    fig.add_trace(go.Scatter(x=df['timestamp'], y=norm_price, name=symbol, mode='lines'))

fig.update_layout(
    template="plotly_dark",
    height=500,
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis_title="Time",
    yaxis_title="Price Change (%)",
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

# --- PHYSICS & TRADES ---
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("🌀 Ising Physics: Magnetization vs Energy")
    # Show BTC Magnetization (The Leader signal)
    if "BTCUSD" in portfolio:
        btc_df = portfolio["BTCUSD"]
        fig_phys = make_subplots(specs=[[{"secondary_y": True}]])
        fig_phys.add_trace(go.Scatter(x=btc_df['timestamp'], y=btc_df['mag'], name="Magnetization", line=dict(color='#00ff00')), secondary_y=False)
        fig_phys.add_trace(go.Scatter(x=btc_df['timestamp'], y=btc_df['energy'], name="Energy", line=dict(color='#ff00ff', dash='dot')), secondary_y=True)
        fig_phys.update_layout(template="plotly_dark", height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_phys, use_container_width=True)

with c2:
    st.subheader("📜 Recent Trades")
    if os.path.exists("paper_trades.csv"):
        trades_df = pd.read_csv("paper_trades.csv").tail(10)
        st.dataframe(trades_df, use_container_width=True)
    else:
        st.info("No trades logged yet.")

# --- AUTO REFRESH ---
time.sleep(2)
st.rerun()
