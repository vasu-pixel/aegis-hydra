
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time

# Config
st.set_page_config(layout="wide", page_title="Aegis Hydra: HFT Monitor")
ANALYSIS_FILE = "hft_analysis.csv"
SIGNALS_FILE = "hft_signals.csv"
WINDOW_SIZE = 200

st.title("⚡ Aegis Hydra: HFT Lead-Lag Monitor")

# Auto-refresh logic
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()

def load_data():
    if not os.path.exists(ANALYSIS_FILE):
        return pd.DataFrame(), pd.DataFrame()
    
    # Read Analysis Data (last N lines to separate read)
    # Using python engine for stability with bad lines
    try:
        # Read tail using system command for speed on large files? 
        # For now, just pandas read_csv
        df = pd.read_csv(ANALYSIS_FILE, names=['time', 'price', 'z_score', 'n', 'mlofi'])
        if len(df) > WINDOW_SIZE:
             df = df.iloc[-WINDOW_SIZE:]
    except:
        df = pd.DataFrame()

    try:
        if os.path.exists(SIGNALS_FILE):
            sigs = pd.read_csv(SIGNALS_FILE, names=['time', 'msg', 'price', 'pnl'])
        else:
            sigs = pd.DataFrame()
    except:
        sigs = pd.DataFrame()
        
    return df, sigs

# Placeholder for charts
chart_placeholder = st.empty()
metrics_placeholder = st.empty()

while True:
    df, sigs = load_data()
    
    if not df.empty:
        # Metrics
        last_row = df.iloc[-1]
        with metrics_placeholder.container():
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Spot Price", f"${last_row['price']:.2f}")
            c2.metric("Z-Score (Lead-Lag)", f"{last_row['z_score']:.2f}", 
                     delta_color="off" if abs(last_row['z_score']) < 3 else "inverse")
            c3.metric("Criticality (n)", f"{last_row['n']:.2f}",
                     delta_color="inverse" if last_row['n'] > 0.8 else "normal")
            c4.metric("MLOFI", f"{last_row['mlofi']:.4f}")

        # Charts
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05,
                            subplot_titles=("Spot Price & Signals", "Z-Score (Futures - Spot)", "Hawkes Criticality"))

        # 1. Price
        fig.add_trace(go.Scatter(x=df['time'], y=df['price'], mode='lines', name='Price', line=dict(color='cyan')), row=1, col=1)
        
        # Signals Overlay
        if not sigs.empty:
            recent_sigs = sigs[sigs['time'] >= df['time'].min()]
            for _, row in recent_sigs.iterrows():
                symbol = "circle"
                color = "yellow"
                size = 8
                if "BUY" in row['msg']: 
                    symbol = "triangle-up"
                    color = "green"
                    size = 12
                elif "SELL" in row['msg']:
                    symbol = "triangle-down"
                    color = "red"
                    size = 12
                
                fig.add_trace(go.Scatter(x=[row['time']], y=[row['price']], mode='markers',
                                         marker=dict(symbol=symbol, color=color, size=size),
                                         showlegend=False, hoverinfo='text', text=row['msg']), row=1, col=1)

        # 2. Z-Score
        fig.add_trace(go.Scatter(x=df['time'], y=df['z_score'], mode='lines', name='Z-Score', line=dict(color='magenta')), row=2, col=1)
        fig.add_hline(y=3.0, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=-3.0, line_dash="dash", line_color="red", row=2, col=1)

        # 3. Criticality
        fig.add_trace(go.Scatter(x=df['time'], y=df['n'], mode='lines', name='Criticality', line=dict(color='orange')), row=3, col=1)
        fig.add_hline(y=0.8, line_dash="dot", line_color="red", row=3, col=1)

        fig.update_layout(height=800, template="plotly_dark", showlegend=False)
        chart_placeholder.plotly_chart(fig, use_container_width=True, key="live_hft_chart")
    
    time.sleep(1)
