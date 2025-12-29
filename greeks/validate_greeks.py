#!/usr/bin/env python3
"""
validate_greeks.py

VALIDAČNÍ SKRIPT - Spusť s ThetaData Terminal!

POUŽITÍ:
    # Pokud Terminal běží lokálně:
    python validate_greeks.py
    
    # Pokud Terminal běží na jiném PC (např. Windows):
    python validate_greeks.py 192.168.1.100
"""

import requests
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from io import StringIO
from datetime import datetime, timedelta
import sys

# Import our modules
from rates_provider import RatesProvider
from greeks_engine import GreeksEngine

# ThetaData host - default localhost, or from command line
if len(sys.argv) > 1:
    THETADATA_HOST = sys.argv[1]
else:
    THETADATA_HOST = "localhost"

THETADATA_PORT = 25503
THETADATA_URL = f"http://{THETADATA_HOST}:{THETADATA_PORT}/v3"


def check_thetadata():
    """Check if ThetaData Terminal is running."""
    try:
        # Use same endpoint as jedem9.py
        r = requests.get(f"{THETADATA_URL}/option/list/expirations", 
                        params={'symbol': 'SPY'}, timeout=5)
        return r.status_code == 200
    except:
        return False


def get_thetadata_greeks(symbol: str, num_strikes: int = 5):
    """Get current Greeks from ThetaData - query individual strikes like jedem9.py."""
    print(f"\nFetching Greeks for {symbol} from ThetaData...")
    
    # Get expirations
    r = requests.get(f"{THETADATA_URL}/option/list/expirations",
                    params={'symbol': symbol}, timeout=30)
    if r.status_code != 200:
        print(f"  Error getting expirations: {r.status_code}")
        return None
    
    exps = pd.read_csv(StringIO(r.text))
    if 'expiration' not in exps.columns or len(exps) == 0:
        print(f"  No expirations found")
        return None
    
    # Find 20-40 DTE expiration
    today = datetime.now()
    exp_dates = pd.to_datetime(exps['expiration'])
    
    target_exp = None
    for exp in sorted(exp_dates):
        dte = (exp - today).days
        if 20 <= dte <= 40:
            target_exp = exp
            break
    
    if target_exp is None:
        for exp in sorted(exp_dates):
            if (exp - today).days > 7:
                target_exp = exp
                break
    
    if target_exp is None:
        print("  No suitable expiration")
        return None
    
    dte = (target_exp - today).days
    exp_str = target_exp.strftime('%Y%m%d')
    print(f"  Expiration: {target_exp.date()} (DTE: {dte})")
    
    # Get strikes
    r = requests.get(f"{THETADATA_URL}/option/list/strikes",
                    params={'symbol': symbol, 'expiration': exp_str}, timeout=30)
    if r.status_code != 200:
        print(f"  Error getting strikes: {r.status_code}")
        return None
    
    strikes_df = pd.read_csv(StringIO(r.text))
    if 'strike' not in strikes_df.columns or len(strikes_df) == 0:
        print(f"  No strikes found")
        return None
    
    strikes = sorted([float(s) for s in strikes_df['strike'].unique()])
    
    # Get one strike to find underlying price
    test_strike = strikes[len(strikes)//2]  # middle strike
    yesterday = (today - timedelta(days=1)).strftime('%Y%m%d')
    
    # Try yesterday first (market might be closed today)
    trade_date = None
    underlying = None
    for date_try in [yesterday, (today - timedelta(days=2)).strftime('%Y%m%d'), 
                     (today - timedelta(days=3)).strftime('%Y%m%d')]:
        params = {
            'symbol': symbol,
            'expiration': exp_str,
            'strike': f"{test_strike:.3f}",
            'right': 'call',
            'date': date_try,
            'interval': '1h'
        }
        r = requests.get(f"{THETADATA_URL}/option/history/greeks/first_order",
                        params=params, timeout=30)
        if r.status_code == 200:
            test_df = pd.read_csv(StringIO(r.text))
            if len(test_df) > 0 and 'underlying_price' in test_df.columns:
                underlying = test_df['underlying_price'].iloc[-1]
                trade_date = date_try
                break
    
    if underlying is None:
        print(f"  Could not get underlying price")
        return None
    
    print(f"  Underlying: ${underlying:.2f} (from {trade_date})")
    
    # Find ATM strikes
    atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - underlying))
    start_idx = max(0, atm_idx - num_strikes)
    end_idx = min(len(strikes), atm_idx + num_strikes + 1)
    selected_strikes = strikes[start_idx:end_idx]
    
    print(f"  Fetching {len(selected_strikes)} strikes around ATM...")
    
    # Fetch Greeks for each strike - ALL ORDERS
    all_data = []
    for strike in selected_strikes:
        for right in ['call', 'put']:
            row_data = {'strike': strike, 'right': right.upper(), 'dte': dte}
            
            # First order Greeks
            params = {
                'symbol': symbol,
                'expiration': exp_str,
                'strike': f"{strike:.3f}",
                'right': right,
                'date': trade_date,
                'interval': '1h'
            }
            
            try:
                r = requests.get(f"{THETADATA_URL}/option/history/greeks/first_order",
                               params=params, timeout=30)
                if r.status_code == 200:
                    df = pd.read_csv(StringIO(r.text))
                    if len(df) > 0:
                        last_row = df.iloc[-1]
                        row_data.update({
                            'underlying_price': last_row['underlying_price'] if 'underlying_price' in last_row.index else np.nan,
                            'implied_vol': last_row['implied_vol'] if 'implied_vol' in last_row.index else np.nan,
                            'delta': last_row['delta'] if 'delta' in last_row.index else np.nan,
                            'theta': last_row['theta'] if 'theta' in last_row.index else np.nan,
                            'vega': last_row['vega'] if 'vega' in last_row.index else np.nan,
                            'rho': last_row['rho'] if 'rho' in last_row.index else np.nan,
                        })
            except Exception as e:
                print(f"    Error fetching 1st order for {strike} {right}: {e}")
            
            # Second order Greeks
            try:
                r = requests.get(f"{THETADATA_URL}/option/history/greeks/second_order",
                               params=params, timeout=30)
                if r.status_code == 200:
                    df = pd.read_csv(StringIO(r.text))
                    if len(df) > 0:
                        last_row = df.iloc[-1]
                        row_data.update({
                            'gamma': last_row['gamma'] if 'gamma' in last_row.index else np.nan,
                            'vanna': last_row['vanna'] if 'vanna' in last_row.index else np.nan,
                            'charm': last_row['charm'] if 'charm' in last_row.index else np.nan,
                            'vomma': last_row['vomma'] if 'vomma' in last_row.index else np.nan,
                            'veta': last_row['veta'] if 'veta' in last_row.index else np.nan,
                        })
            except Exception as e:
                print(f"    Error fetching 2nd order for {strike} {right}: {e}")
            
            # Third order Greeks
            try:
                r = requests.get(f"{THETADATA_URL}/option/history/greeks/third_order",
                               params=params, timeout=30)
                if r.status_code == 200:
                    df = pd.read_csv(StringIO(r.text))
                    if len(df) > 0:
                        last_row = df.iloc[-1]
                        row_data.update({
                            'speed': last_row['speed'] if 'speed' in last_row.index else np.nan,
                            'zomma': last_row['zomma'] if 'zomma' in last_row.index else np.nan,
                            'color': last_row['color'] if 'color' in last_row.index else np.nan,
                            'ultima': last_row['ultima'] if 'ultima' in last_row.index else np.nan,
                        })
            except Exception as e:
                print(f"    Error fetching 3rd order for {strike} {right}: {e}")
            
            if 'underlying_price' in row_data:
                all_data.append(row_data)
    
    if not all_data:
        print(f"  No data fetched")
        return None
    
    result_df = pd.DataFrame(all_data)
    print(f"  Got {len(result_df)} quotes with all Greeks orders")
    
    return result_df


def validate_symbol(symbol: str, provider: RatesProvider, engine: GreeksEngine):
    """Validate Greeks for a symbol - all orders."""
    print(f"\n{'='*60}")
    print(f"VALIDATING: {symbol}")
    print("="*60)
    
    # Get ThetaData Greeks
    td_df = get_thetadata_greeks(symbol)
    if td_df is None or len(td_df) == 0:
        print("  No data from ThetaData")
        return None
    
    # Get our rates
    underlying = td_df['underlying_price'].iloc[0]
    rates = provider.get_rates(symbol, stock_price=underlying)
    r, q = rates['r'], rates['q']
    
    print(f"\n  Our rates: r={r:.4f} ({r*100:.2f}%), q={q:.4f} ({q*100:.2f}%)")
    print(f"  ThetaData uses: r=SOFR, q=0 (ignores dividends)")
    
    # Debug: print columns
    print(f"\n  DEBUG - DataFrame columns: {list(td_df.columns)}")
    print(f"  DEBUG - First row sample:")
    print(f"    {td_df.iloc[0].to_dict()}")
    
    # IV column name
    iv_col = 'implied_vol' if 'implied_vol' in td_df.columns else 'iv'
    print(f"  DEBUG - Using IV column: {iv_col}")
    
    results = []
    skipped_reasons = {'no_iv': 0, 'invalid_iv': 0, 'no_delta': 0, 'invalid_delta': 0, 'error': 0}
    
    for _, row in td_df.iterrows():
        try:
            S = float(row['underlying_price'])
            K = float(row['strike'])
            T = float(row['dte']) / 365.0
            
            # Check IV
            if iv_col not in row or pd.isna(row[iv_col]):
                skipped_reasons['no_iv'] += 1
                continue
            iv = float(row[iv_col])
            
            # Skip rows with invalid IV
            if iv <= 0 or iv > 5 or np.isnan(iv):
                skipped_reasons['invalid_iv'] += 1
                continue
            
            right_val = str(row['right']).upper()
            is_call = right_val in ['CALL', 'C']
            
            # ThetaData values
            td_delta = row.get('delta', np.nan)
            if pd.isna(td_delta):
                skipped_reasons['no_delta'] += 1
                continue
            td_delta = float(td_delta)
            
            td_gamma = float(row.get('gamma', np.nan)) if not pd.isna(row.get('gamma')) else np.nan
            td_theta = float(row.get('theta', np.nan)) if not pd.isna(row.get('theta')) else np.nan
            td_vega = float(row.get('vega', np.nan)) if not pd.isna(row.get('vega')) else np.nan
            td_vanna = float(row.get('vanna', np.nan)) if not pd.isna(row.get('vanna')) else np.nan
            td_charm = float(row.get('charm', np.nan)) if not pd.isna(row.get('charm')) else np.nan
            td_vomma = float(row.get('vomma', np.nan)) if not pd.isna(row.get('vomma')) else np.nan
            td_veta = float(row.get('veta', np.nan)) if not pd.isna(row.get('veta')) else np.nan
            td_speed = float(row.get('speed', np.nan)) if not pd.isna(row.get('speed')) else np.nan
            td_zomma = float(row.get('zomma', np.nan)) if not pd.isna(row.get('zomma')) else np.nan
            td_color = float(row.get('color', np.nan)) if not pd.isna(row.get('color')) else np.nan
            td_ultima = float(row.get('ultima', np.nan)) if not pd.isna(row.get('ultima')) else np.nan
            
            # Skip rows with clearly invalid data
            if np.isnan(td_delta) or abs(td_delta) > 1.5 or td_delta == 0:
                skipped_reasons['invalid_delta'] += 1
                continue
            
            # Our calculations (with q=0 to match ThetaData)
            td_r = r  # They also use SOFR
            td_q = 0.0  # They ignore dividends
            
            # First order
            calc_delta = engine.delta(S, K, T, td_r, td_q, iv, is_call)
            calc_gamma = engine.gamma(S, K, T, td_r, td_q, iv)
            calc_theta = engine.theta(S, K, T, td_r, td_q, iv, is_call)
            calc_vega = engine.vega(S, K, T, td_r, td_q, iv)
            
            # Second order
            calc_vanna = engine.vanna(S, K, T, td_r, td_q, iv)
            calc_charm = engine.charm(S, K, T, td_r, td_q, iv, is_call)
            calc_vomma = engine.vomma(S, K, T, td_r, td_q, iv)
            calc_veta = engine.veta(S, K, T, td_r, td_q, iv)
            
            # Third order
            calc_speed = engine.speed(S, K, T, td_r, td_q, iv)
            calc_zomma = engine.zomma(S, K, T, td_r, td_q, iv)
            calc_color = engine.color(S, K, T, td_r, td_q, iv)
            calc_ultima = engine.ultima(S, K, T, td_r, td_q, iv)
            
            # Calculate errors (percentage)
            def pct_err(calc, td):
                if np.isnan(td) or np.isnan(calc) or td == 0:
                    return np.nan
                return abs(calc - td) / max(abs(td), 1e-10) * 100
            
            results.append({
                'strike': K, 
                'right': 'C' if is_call else 'P',
                'iv': iv,
                # First order
                'td_delta': td_delta, 'calc_delta': calc_delta, 'delta_err': pct_err(calc_delta, td_delta),
                'td_gamma': td_gamma, 'calc_gamma': calc_gamma, 'gamma_err': pct_err(calc_gamma, td_gamma),
                'td_theta': td_theta, 'calc_theta': calc_theta, 'theta_err': pct_err(calc_theta, td_theta),
                # Second order  
                'td_vanna': td_vanna, 'calc_vanna': calc_vanna, 'vanna_err': pct_err(calc_vanna, td_vanna),
                'td_charm': td_charm, 'calc_charm': calc_charm, 'charm_err': pct_err(calc_charm, td_charm),
                'td_vomma': td_vomma, 'calc_vomma': calc_vomma, 'vomma_err': pct_err(calc_vomma, td_vomma),
                # Third order
                'td_speed': td_speed, 'calc_speed': calc_speed, 'speed_err': pct_err(calc_speed, td_speed),
                'td_zomma': td_zomma, 'calc_zomma': calc_zomma, 'zomma_err': pct_err(calc_zomma, td_zomma),
                'td_color': td_color, 'calc_color': calc_color, 'color_err': pct_err(calc_color, td_color),
                'td_ultima': td_ultima, 'calc_ultima': calc_ultima, 'ultima_err': pct_err(calc_ultima, td_ultima),
            })
        except Exception as e:
            skipped_reasons['error'] += 1
            print(f"    ERROR processing row: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not results:
        print(f"  No valid results")
        print(f"  DEBUG - Skip reasons: {skipped_reasons}")
        return None
    
    results_df = pd.DataFrame(results)
    
    # Print First Order Greeks comparison
    print(f"\n  FIRST ORDER GREEKS (q=0 to match ThetaData):")
    print(f"  {'Strike':>8} {'Type':>4} {'IV':>6} | {'TD Δ':>8} {'Calc Δ':>8} {'Err%':>6} | {'TD θ':>8} {'Calc θ':>8} {'Err%':>6}")
    print(f"  {'-'*8} {'-'*4} {'-'*6} | {'-'*8} {'-'*8} {'-'*6} | {'-'*8} {'-'*8} {'-'*6}")
    
    for _, r in results_df.head(10).iterrows():
        print(f"  {r['strike']:>8.1f} {r['right']:>4} {r['iv']:>6.2%} | "
              f"{r['td_delta']:>8.4f} {r['calc_delta']:>8.4f} {r['delta_err']:>5.2f}% | "
              f"{r['td_theta']:>8.4f} {r['calc_theta']:>8.4f} {r['theta_err']:>5.2f}%")
    
    # Print Second Order Greeks comparison (if available)
    if not results_df['td_vanna'].isna().all():
        print(f"\n  SECOND ORDER GREEKS:")
        print(f"  {'Strike':>8} {'Type':>4} | {'TD Vanna':>10} {'Calc':>10} {'Err%':>7} | {'TD Charm':>10} {'Calc':>10} {'Err%':>7}")
        print(f"  {'-'*8} {'-'*4} | {'-'*10} {'-'*10} {'-'*7} | {'-'*10} {'-'*10} {'-'*7}")
        
        for _, r in results_df.head(10).iterrows():
            vanna_td = r['td_vanna'] if not np.isnan(r['td_vanna']) else 0
            charm_td = r['td_charm'] if not np.isnan(r['td_charm']) else 0
            print(f"  {r['strike']:>8.1f} {r['right']:>4} | "
                  f"{vanna_td:>10.4f} {r['calc_vanna']:>10.4f} {r['vanna_err']:>6.2f}% | "
                  f"{charm_td:>10.6f} {r['calc_charm']:>10.6f} {r['charm_err']:>6.2f}%")
    
    # Print Third Order Greeks comparison (if available)
    if not results_df['td_speed'].isna().all():
        print(f"\n  THIRD ORDER GREEKS:")
        print(f"  {'Strike':>8} {'Type':>4} | {'TD Speed':>12} {'Calc':>12} {'Err%':>7} | {'TD Zomma':>12} {'Calc':>12} {'Err%':>7}")
        print(f"  {'-'*8} {'-'*4} | {'-'*12} {'-'*12} {'-'*7} | {'-'*12} {'-'*12} {'-'*7}")
        
        for _, r in results_df.head(10).iterrows():
            speed_td = r['td_speed'] if not np.isnan(r['td_speed']) else 0
            zomma_td = r['td_zomma'] if not np.isnan(r['td_zomma']) else 0
            print(f"  {r['strike']:>8.1f} {r['right']:>4} | "
                  f"{speed_td:>12.8f} {r['calc_speed']:>12.8f} {r['speed_err']:>6.2f}% | "
                  f"{zomma_td:>12.8f} {r['calc_zomma']:>12.8f} {r['zomma_err']:>6.2f}%")
    
    # Calculate averages
    avg_delta = results_df['delta_err'].mean()
    avg_theta = results_df['theta_err'].mean()
    avg_gamma = results_df['gamma_err'].dropna().mean()
    avg_vanna = results_df['vanna_err'].dropna().mean()
    avg_charm = results_df['charm_err'].dropna().mean()
    avg_speed = results_df['speed_err'].dropna().mean()
    avg_zomma = results_df['zomma_err'].dropna().mean()
    avg_color = results_df['color_err'].dropna().mean()
    
    print(f"\n  AVERAGE ERRORS:")
    print(f"    1st Order: Delta={avg_delta:.2f}%, Theta={avg_theta:.2f}%, Gamma={avg_gamma:.2f}%")
    if not np.isnan(avg_vanna):
        print(f"    2nd Order: Vanna={avg_vanna:.2f}%, Charm={avg_charm:.2f}%")
    if not np.isnan(avg_speed):
        print(f"    3rd Order: Speed={avg_speed:.2f}%, Zomma={avg_zomma:.2f}%, Color={avg_color:.2f}%")
    
    # Overall assessment
    max_err = max(avg_delta, avg_theta)
    if max_err < 2:
        print("  ✅ PERFECT MATCH - formulas are correct!")
    elif max_err < 5:
        print("  ✅ GOOD MATCH - small differences likely from timing")
    else:
        print("  ⚠️ DIFFERENCES - check formula or data")
    
    return {
        'symbol': symbol, 'r': r, 'q': q,
        'delta_err': avg_delta, 'theta_err': avg_theta, 'gamma_err': avg_gamma,
        'vanna_err': avg_vanna, 'charm_err': avg_charm,
        'speed_err': avg_speed, 'zomma_err': avg_zomma, 'color_err': avg_color
    }


def main():
    print("="*60)
    print("GREEKS VALIDATION")
    print("="*60)
    print(f"Time: {datetime.now()}")
    print(f"ThetaData: {THETADATA_HOST}:{THETADATA_PORT}")
    
    # Check ThetaData
    print("\nChecking ThetaData Terminal...")
    if not check_thetadata():
        print("❌ ThetaData Terminal not running!")
        print("   Start Terminal on localhost:25503")
        sys.exit(1)
    print("✅ Connected")
    
    # Initialize
    print("\nInitializing rates provider...")
    provider = RatesProvider(cache_dir="rates_cache")
    engine = GreeksEngine(provider)
    
    # Fetch SOFR
    print("\nFetching SOFR from NY Fed...")
    try:
        provider.fetch_sofr_history(100)
    except Exception as e:
        print(f"❌ Failed: {e}")
        sys.exit(1)
    
    # Test symbols - same as jedem9.py
    symbols = [
        'SPY', 'QQQ', 'AAPL', 'MSFT', 'IWM',
        'AMZN', 'TSLA', 'NVDA', 'COIN', 'AMD',
        'JPM', 'SMCI', 'GLD', 'TLT'
    ]
    
    # Fetch dividends for each
    print("\nFetching dividend history...")
    for symbol in symbols:
        try:
            provider.fetch_dividend_history(symbol)
            provider.fetch_eod_prices(symbol)
        except:
            print(f"  Warning: Could not fetch data for {symbol}")
    
    # Validate each
    all_results = []
    for symbol in symbols:
        result = validate_symbol(symbol, provider, engine)
        if result:
            all_results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if all_results:
        summary = pd.DataFrame(all_results)
        print(f"\n{summary.to_string(index=False)}")
        
        print(f"\n" + "-"*40)
        print("NOTE: ThetaData uses q=0 (no dividends)")
        print("Your calculations include dividend yield.")
        print("Expected difference: ~1-5% for dividend-paying stocks")
        
        print(f"\nYour SOFR: {provider.get_sofr():.4f} ({provider.get_sofr()*100:.2f}%)")
        
        for r in all_results:
            print(f"{r['symbol']} dividend yield: {r['q']:.4f} ({r['q']*100:.2f}%)")
    
    print("\n✅ Validation complete!")


if __name__ == "__main__":
    main()