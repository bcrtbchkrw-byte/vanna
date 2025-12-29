#!/usr/bin/env python3
"""
prefetch_rates.py

Stáhne VŠECHNA data pro r a q - spusť JEDNOU dokud máš ThetaData subscription!

Data se uloží do rates_cache/ a zůstanou navždy.
"""

from rates_provider import RatesProvider

# Stejné symboly jako v jedem9.py
SYMBOLS = [
    'SPY', 'QQQ', 'AAPL', 'MSFT', 'IWM',
    'AMZN', 'TSLA', 'NVDA', 'COIN', 'AMD',
    'JPM', 'SMCI', 'GLD', 'TLT'
]

if __name__ == "__main__":
    print("="*60)
    print("PREFETCH ALL RATES DATA")
    print("="*60)
    print(f"Symbols: {SYMBOLS}")
    print("="*60)
    
    provider = RatesProvider(cache_dir="rates_cache")
    
    # Stáhne:
    # - SOFR z NY Fed (2500 dní = ~7 let)
    # - Dividendy z ThetaData pro každý symbol (15 let)
    # - EOD ceny z ThetaData pro každý symbol (15 let)
    provider.prefetch_all(SYMBOLS, years=15)
    
    provider.print_status()
    
    print("\n" + "="*60)
    print("✅ Data uložena v rates_cache/")
    print("   Tyto soubory si ZÁLOHUJ - zůstanou navždy!")
    print("="*60)