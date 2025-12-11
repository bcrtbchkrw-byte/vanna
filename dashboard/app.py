import glob
import os
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from utils import fetch_ai_decisions, fetch_positions, fetch_trades, get_system_status

# --- Configuration ---
st.set_page_config(
    page_title="Vanna | AI Trading Terminal",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styles ---
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .metric-card {
        background-color: #262730;
        border: 1px solid #464b5f;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .status-online { color: #00fa9a; font-weight: bold; }
    .status-offline { color: #ff6347; font-weight: bold; }
    .status-unknown { color: #ffa500; font-weight: bold; }
    
    /* Tables */
    [data-testid="stDataFrame"] {
        border: 1px solid #464b5f;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- Functions ---
def get_latest_log_content(lines=100):
    """Get the tail of the latest log file."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        return "Log directory not found."
    
    # Find all vanna_*.log files
    log_files = glob.glob(os.path.join(log_dir, "vanna_*.log"))
    if not log_files:
        return "No log files found."
    
    # Sort by modification time
    latest_log = max(log_files, key=os.path.getmtime)
    
    try:
        with open(latest_log, "r") as f:
            # Read all lines and take the last N
            # For very large files this is inefficient, but okay for typical daily logs
            content = f.readlines()
            return "".join(content[-lines:])
    except Exception as e:
        return f"Error reading log: {e}"

# --- Sidebar ---
with st.sidebar:
    st.title("🤖 Vanna Trader")
    st.caption(f"v1.0.0 | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.markdown("---")
    
    st.subheader("System Status")
    status = get_system_status()
    
    # Trader Status
    s_trader = status.get('trader', 'UNKNOWN')
    color_t = "status-online" if s_trader == 'ONLINE' else "status-offline"
    st.markdown(f"**Trader Core**: <span class='{color_t}'>{s_trader}</span>", unsafe_allow_html=True)
    
    # IBKR Status
    s_ibkr = status.get('ib-gateway', 'UNKNOWN')
    color_i = "status-online" if s_ibkr == 'ONLINE' else "status-offline"
    st.markdown(f"**IBKR Gateway**: <span class='{color_i}'>{s_ibkr}</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

# --- Main Content ---
st.title("Dashboard")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Live Operations", "AI Pipeline", "Logs"])

# --- Tab 1: Overview ---
with tab1:
    st.header("Performance Overview")
    
    # Fetch Data
    trades = fetch_trades(limit=500)
    positions = fetch_positions()
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_pnl = trades['pnl'].sum() if not trades.empty else 0.0
    win_rate = 0.0
    if not trades.empty:
        closed_trades = trades[trades['status'] == 'CLOSED']
        if not closed_trades.empty:
            wins = len(closed_trades[closed_trades['pnl'] > 0])
            win_rate = (wins / len(closed_trades)) * 100
            
    open_pos_count = len(positions)
    
    col1.metric("Total P&L", f"${total_pnl:,.2f}", delta=None)
    col2.metric("Win Rate", f"{win_rate:.1f}%")
    col3.metric("Open Positions", str(open_pos_count))
    col4.metric("Active Strategy", "Iron Condor / Spreads") # Placeholder or derive from config
    
    st.markdown("---")
    
    # Chart
    if not trades.empty and 'entry_time' in trades.columns:
        # Convert entry_time to datetime
        trades['entry_time'] = pd.to_datetime(trades['entry_time'])
        trades_sorted = trades.sort_values('entry_time')
        trades_sorted['cumulative_pnl'] = trades_sorted['pnl'].fillna(0).cumsum()
        
        fig = px.area(trades_sorted, x='entry_time', y='cumulative_pnl', 
                      title="Cumulative P&L",
                      labels={'entry_time': 'Date', 'cumulative_pnl': 'P&L ($)'},
                      template="plotly_dark")
        fig.update_traces(line_color='#00fa9a')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trade history available for charting.")

# --- Tab 2: Live Operations ---
with tab2:
    st.header("Live Operations")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Active Positions")
        positions = fetch_positions()
        if not positions.empty:
            st.dataframe(positions, use_container_width=True, hide_index=True)
        else:
            st.info("No open positions.")
            
    with col2:
        st.subheader("Recent Trades")
        trades_recent = fetch_trades(limit=20)
        if not trades_recent.empty:
            st.dataframe(
                trades_recent[['symbol', 'strategy', 'pnl', 'status']], 
                use_container_width=True, 
                hide_index=True
            )
        else:
            st.info("No recent trades.")

# --- Tab 3: AI Pipeline ---
with tab3:
    st.header("AI Decision Pipeline")
    
    decisions = fetch_ai_decisions(limit=100)
    
    if not decisions.empty:
        # Layout metrics for AI
        total_decisions = len(decisions)
        approvals = len(decisions[decisions['decision'] == 'APPROVE'])
        rejections = len(decisions[decisions['decision'] == 'REJECT'])
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Analyzed", total_decisions)
        m2.metric("Approvals", approvals)
        m3.metric("Rejections", rejections)
        
        st.markdown("### Audit Decision Log")
        
        # Color code decisions
        def color_decision(val):
            color = '#00fa9a' if val == 'APPROVE' else '#ff6347'
            return f'color: {color}'

        st.dataframe(
            decisions.style.map(color_decision, subset=['decision']),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No AI decisions logged yet.")

# --- Tab 4: Logs ---
with tab4:
    st.header("System Logs")
    
    # Auto-refresh checkbox
    auto_refresh = st.checkbox("Auto-refresh logs (every 5s)")
    
    log_content = get_latest_log_content()
    st.code(log_content, language="text")
    
    if auto_refresh:
        time.sleep(5)
        st.rerun()
