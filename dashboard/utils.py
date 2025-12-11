import os
import sqlite3

import pandas as pd

DB_PATH = "data/vanna.db"

def get_connection():
    """Get a synchronous SQLite connection (read-only mode preferred for dashboard)."""
    if not os.path.exists(DB_PATH):
        return None
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        return conn
    except sqlite3.Error:
        # Fallback to standard connection if URI mode fails
        return sqlite3.connect(DB_PATH)

def fetch_trades(limit=50):
    """Fetch recent trades."""
    conn = get_connection()
    if not conn:
        return pd.DataFrame()
    
    query = f"""
        SELECT id, symbol, strategy, entry_time, exit_time, 
               entry_price, exit_price, quantity, pnl, status, notes
        FROM trades 
        ORDER BY entry_time DESC 
        LIMIT {limit}
    """
    try:
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        print(f"Error fetching trades: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def fetch_positions():
    """Fetch open positions."""
    conn = get_connection()
    if not conn:
        return pd.DataFrame()
    
    query = """
        SELECT id, symbol, strategy, legs, entry_time, 
               entry_price, quantity, target_profit, stop_loss, status
        FROM positions 
        WHERE status = 'OPEN' 
        ORDER BY entry_time DESC
    """
    try:
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        print(f"Error fetching positions: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def fetch_ai_decisions(limit=100):
    """Fetch AI decision log."""
    conn = get_connection()
    if not conn:
        return pd.DataFrame()
    
    query = f"""
        SELECT id, symbol, phase, decision, reasoning, confidence, cost, created_at
        FROM ai_decisions 
        ORDER BY created_at DESC 
        LIMIT {limit}
    """
    try:
        df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        print(f"Error fetching AI decisions: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def get_system_status():
    """Check status of Docker containers."""
    status = {
        'trader': 'UNKNOWN',
        'ib-gateway': 'UNKNOWN'
    }
    
    try:
        # Check trader container (self)
        # Inside docker, hostname is often the container ID, but we can't easily check 'docker ps' from within
        # unless we mount docker socket.
        # Assuming we can't access docker socket easily, we just say ONLINE if this code runs.
        status['trader'] = 'ONLINE' 
        
        # Check IB Gateway connectivity via ping or socket check
        # We can try to connect to the IB port
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        # Assuming localhost since we run in host network or adjacent container?
        # In docker-compose, 'ib-gateway' is the host
        # But 'config.py' says IBKR_HOST defaults to 'ib-gateway'
        host = os.getenv('IBKR_HOST', 'ib-gateway')
        port = int(os.getenv('IBKR_PORT', '4002'))
        result = sock.connect_ex((host, port))
        if result == 0:
            status['ib-gateway'] = 'ONLINE'
        else:
            status['ib-gateway'] = 'OFFLINE'
        sock.close()
        
    except Exception as e:
        print(f"Status check error: {e}")
        
    return status
