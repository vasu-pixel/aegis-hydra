import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
import math
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="AEGIS HYDRA CONTROL CENTER", layout="wide", initial_sidebar_state="collapsed")

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(DATA_DIR, "..", "..")

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    [data-testid="stMetricValue"] { font-size: 24px; }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data(ttl=2)
def load_signals(symbol):
    path = os.path.join(PROJECT_ROOT, f"hft_signals_{symbol}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, header=None, names=['timestamp', 'signal', 'price', 'pnl'])
        df['time'] = pd.to_datetime(df['timestamp'], unit='s')
        return df.tail(1000)
    except:
        return pd.DataFrame()

@st.cache_data(ttl=2)
def load_latency(symbol):
    path = os.path.join(PROJECT_ROOT, f"hft_latency_{symbol}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, header=None, names=['timestamp', 'latency_ms'])
        df['time'] = pd.to_datetime(df['timestamp'], unit='s')
        return df.tail(500)
    except:
        return pd.DataFrame()

@st.cache_data(ttl=2)
def load_trades():
    path = os.path.join(PROJECT_ROOT, "paper_trades.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path).tail(30)
    except:
        return pd.DataFrame()

# --- HEADER ---
st.title("🛡️ AEGIS HYDRA | TRI-FORCE CONTROL CENTER")
st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")
st.markdown("---")

# --- LOAD ALL ASSETS ---
SYMBOLS = ["BTCUSD", "ETHUSD", "USDTUSD"]
COLORS = {"BTCUSD": "#F7931A", "ETHUSD": "#627EEA", "USDTUSD": "#26A17B"}
ICONS = {"BTCUSD": "🟡", "ETHUSD": "🔵", "USDTUSD": "🟢"}

data = {}
for s in SYMBOLS:
    df = load_signals(s)
    if not df.empty:
        data[s] = df

if not data:
    st.warning("⏳ Waiting for signal data... Make sure `launcher.py` is running.")
    time.sleep(3)
    st.rerun()

# =====================================================
# SECTION 1: TOP-LINE PORTFOLIO METRICS
# =====================================================
total_pnl = 0.0
total_buys = 0
total_sells = 0
total_wins = 0
total_losses = 0

asset_stats = {}
for symbol, df in data.items():
    buys = df[df['signal'].str.contains('BUY', na=False)]
    sells = df[df['signal'].str.contains('SELL', na=False)]
    total_trades = len(buys) + len(sells)
    
    pnl_series = df['pnl']
    last_pnl = pnl_series.iloc[-1]
    total_pnl += last_pnl
    
    # Win/Loss: count transitions where PnL increased vs decreased
    pnl_diff = pnl_series.diff().dropna()
    wins = (pnl_diff > 0).sum()
    losses = (pnl_diff < 0).sum()
    total_wins += wins
    total_losses += losses
    total_buys += len(buys)
    total_sells += len(sells)
    
    # Max Drawdown
    cummax = pnl_series.cummax()
    drawdown = pnl_series - cummax
    max_dd = drawdown.min()
    
    # Sharpe Ratio (annualized from tick returns)
    returns = pnl_diff
    sharpe = 0.0
    if len(returns) > 1 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * math.sqrt(252 * 24 * 60)
    
    asset_stats[symbol] = {
        'price': df['price'].iloc[-1],
        'pnl': last_pnl,
        'signal': df['signal'].iloc[-1],
        'buys': len(buys),
        'sells': len(sells),
        'trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0,
        'max_dd': max_dd,
        'sharpe': sharpe,
    }

# Portfolio Summary Row
port_win_rate = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0

p1, p2, p3, p4, p5 = st.columns(5)
p1.metric("💰 Portfolio PnL", f"{total_pnl:+.4f}%")
p2.metric("📊 Total Trades", f"{total_buys + total_sells}")
p3.metric("🏆 Win Rate", f"{port_win_rate:.1f}%")
p4.metric("📈 Wins / 📉 Losses", f"{total_wins} / {total_losses}")
p5.metric("⚡ Active Engines", f"{len(data)}/{len(SYMBOLS)}")

st.markdown("---")

# Per-Asset Metrics
cols = st.columns(len(data))
for i, (symbol, stats) in enumerate(asset_stats.items()):
    with cols[i]:
        st.markdown(f"### {ICONS.get(symbol, '⚪')} {symbol}")
        st.metric("Price", f"${stats['price']:,.2f}")
        
        m1, m2 = st.columns(2)
        m1.metric("PnL", f"{stats['pnl']:+.4f}%")
        m2.metric("Win Rate", f"{stats['win_rate']:.1f}%")
        
        m3, m4 = st.columns(2)
        m3.metric("Max DD", f"{stats['max_dd']:.4f}%")
        m4.metric("Sharpe", f"{stats['sharpe']:.2f}")
        
        st.caption(f"Signal: {stats['signal']} | Trades: {stats['trades']} (B:{stats['buys']} S:{stats['sells']})")

st.markdown("---")

# =====================================================
# SECTION 2: LIVE PRICE OVERLAY (NORMALIZED)
# =====================================================
st.subheader("📊 Live Price Overlay (Normalized %)")

fig_price = go.Figure()
for symbol, df in data.items():
    first_price = df['price'].iloc[0]
    if first_price > 0:
        norm = (df['price'] / first_price - 1) * 100
        fig_price.add_trace(go.Scatter(
            x=df['time'], y=norm, name=symbol,
            line=dict(color=COLORS.get(symbol, "#fff"), width=2), mode='lines'
        ))

fig_price.update_layout(
    template="plotly_dark", height=400,
    margin=dict(l=20, r=20, t=10, b=20),
    yaxis_title="Change (%)", hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02)
)
st.plotly_chart(fig_price, use_container_width=True)

# =====================================================
# SECTION 3: PnL + LATENCY
# =====================================================
left, right = st.columns([3, 2])

with left:
    st.subheader("📈 PnL Curves")
    fig_pnl = go.Figure()
    for symbol, df in data.items():
        fig_pnl.add_trace(go.Scatter(
            x=df['time'], y=df['pnl'], name=symbol,
            line=dict(color=COLORS.get(symbol, "#fff"), width=2), mode='lines'
        ))
    fig_pnl.update_layout(
        template="plotly_dark", height=300,
        margin=dict(l=20, r=20, t=10, b=20),
        yaxis_title="PnL (%)", hovermode="x unified"
    )
    st.plotly_chart(fig_pnl, use_container_width=True)

with right:
    st.subheader("⏱️ Latency Monitor")
    has_latency = False
    fig_lat = go.Figure()
    for symbol in data.keys():
        lat_df = load_latency(symbol)
        if not lat_df.empty:
            has_latency = True
            fig_lat.add_trace(go.Scatter(
                x=lat_df['time'], y=lat_df['latency_ms'], name=symbol,
                line=dict(color=COLORS.get(symbol, "#fff"), width=1), mode='lines'
            ))
    
    if has_latency:
        fig_lat.update_layout(
            template="plotly_dark", height=300,
            margin=dict(l=20, r=20, t=10, b=20),
            yaxis_title="Latency (ms)", hovermode="x unified"
        )
        st.plotly_chart(fig_lat, use_container_width=True)
    else:
        st.info("No latency data yet.")

# =====================================================
# SECTION 4: TRADE LOG
# =====================================================
st.subheader("📜 Trade History")
trades = load_trades()
if not trades.empty:
    st.dataframe(trades, use_container_width=True, height=300)
else:
    st.info("No trades logged yet.")

# --- AUTO REFRESH ---
time.sleep(3)
st.rerun()
