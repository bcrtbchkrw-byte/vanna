import os
import sys

import pandas as pd

# Add parent dir to path to allow importing dashboard.utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dashboard.utils import fetch_ai_decisions, fetch_positions, fetch_trades, get_system_status


def test_fetch_trades():
    print("Testing fetch_trades...", end=" ")
    df = fetch_trades()
    assert isinstance(df, pd.DataFrame), "Result is not a DataFrame"
    print("✅ PASSED")

def test_fetch_positions():
    print("Testing fetch_positions...", end=" ")
    df = fetch_positions()
    assert isinstance(df, pd.DataFrame), "Result is not a DataFrame"
    print("✅ PASSED")

def test_fetch_ai_decisions():
    print("Testing fetch_ai_decisions...", end=" ")
    df = fetch_ai_decisions()
    assert isinstance(df, pd.DataFrame), "Result is not a DataFrame"
    print("✅ PASSED")

def test_system_status():
    print("Testing system_status...", end=" ")
    status = get_system_status()
    assert isinstance(status, dict), "Result is not a dict"
    assert 'trader' in status
    assert 'ib-gateway' in status
    print(f"✅ PASSED (Status: {status})")

if __name__ == "__main__":
    test_fetch_trades()
    test_fetch_positions()
    test_fetch_ai_decisions()
    test_system_status()
    print("Ref dashboard utils test passed")
