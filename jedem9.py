import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import time
import requests
from io import StringIO
import gc
import json
import os
import logging

# Nastavení loggingu
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('options_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OptionsDataDownloader:
    def __init__(self, output_dir="options_ml_full", terminal_host="localhost", terminal_port=25503):
        """
        jedem9_fixed.py - OPRAVENÁ VERZE
        
        OPRAVY:
        1. Retry logika pro API volání
        2. Logování chyb místo tichého ignorování
        3. Robustnější early exit (10 dní místo 5)
        4. Lepší odhad listing date podle typu expirace
        5. Opravený moneyness filtr
        """
        self.base_url = f"http://{terminal_host}:{terminal_port}/v3"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.progress_file = self.output_dir / "download_progress.json"
        self.progress = self.load_progress()
        
        self.underlying_cache = {}
        
        self.symbols = [
            'SPY', 'QQQ', 'AAPL', 'MSFT', 'IWM', 
            'AMZN', 'TSLA', 'NVDA', 'COIN', 'AMD',
            'JPM', 'SMCI', 'GLD', 'TLT'
        ]
        
        # Konfigurace retry
        self.max_retries = 3
        self.retry_delay_base = 2  # exponential backoff base
    
    def api_call_with_retry(self, url, params, timeout=45):
        """
        Provede API volání s retry logikou a exponential backoff.
        
        Returns:
            requests.Response nebo None při selhání
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, params=params, timeout=timeout)
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:  # Rate limit
                    wait_time = self.retry_delay_base ** (attempt + 2)
                    logger.warning(f"Rate limit hit, čekám {wait_time}s...")
                    time.sleep(wait_time)
                elif response.status_code >= 500:  # Server error
                    wait_time = self.retry_delay_base ** attempt
                    logger.warning(f"Server error {response.status_code}, retry {attempt+1}/{self.max_retries}")
                    time.sleep(wait_time)
                else:
                    # Klientská chyba (4xx kromě 429) - neretry
                    logger.debug(f"API vrátilo {response.status_code} pro {url}")
                    return None
                    
            except requests.exceptions.Timeout as e:
                last_error = e
                wait_time = self.retry_delay_base ** attempt
                logger.warning(f"Timeout, retry {attempt+1}/{self.max_retries}, čekám {wait_time}s")
                time.sleep(wait_time)
                
            except requests.exceptions.ConnectionError as e:
                last_error = e
                wait_time = self.retry_delay_base ** (attempt + 1)
                logger.warning(f"Connection error, retry {attempt+1}/{self.max_retries}, čekám {wait_time}s")
                time.sleep(wait_time)
                
            except requests.exceptions.RequestException as e:
                last_error = e
                logger.error(f"Request error: {e}")
                break
        
        if last_error:
            logger.error(f"API volání selhalo po {self.max_retries} pokusech: {last_error}")
        return None

    def load_progress(self):
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Chyba při načítání progress souboru: {e}")
                return {}
        return {}
    
    def save_progress(self):
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
        except IOError as e:
            logger.error(f"Chyba při ukládání progress: {e}")
            
    def mark_chunk_complete(self, symbol, year):
        key = f"{symbol}_{year}"
        self.progress[key] = {'completed': True, 'timestamp': datetime.now().isoformat()}
        self.save_progress()

    def is_chunk_complete(self, symbol, year):
        key = f"{symbol}_{year}"
        return self.progress.get(key, {}).get('completed', False)

    def connect(self):
        logger.info(f"Zkouším připojení k ThetaData Terminal V3...")
        try:
            test_url = f"{self.base_url}/option/list/expirations"
            response = requests.get(test_url, params={'symbol': 'SPY'}, timeout=10)
            if response.status_code == 200:
                logger.info("Připojení úspěšné")
                return True
            else:
                logger.error(f"Připojení selhalo: HTTP {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Připojení selhalo: {e}")
            return False

    def estimate_listing_date(self, expiration):
        """
        Odhadne listing date opce podle typu expirace.
        
        - Weekly opce: ~8 týdnů před expirací
        - Monthly opce: ~9 měsíců před expirací  
        - LEAPS: ~2.5 roku před expirací
        """
        # Zjistit jestli je to 3. pátek v měsíci (monthly)
        exp_date = expiration.date() if hasattr(expiration, 'date') else expiration
        
        # Najít 3. pátek v měsíci expirace
        first_day = exp_date.replace(day=1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday() + 7) % 7)
        third_friday = first_friday + timedelta(weeks=2)
        
        days_to_exp = (expiration - datetime.now()).days if isinstance(expiration, datetime) else 365
        
        if exp_date == third_friday:
            # Monthly opce
            if days_to_exp > 365:
                # LEAPS - listují se ~2.5 roku předem
                return expiration - timedelta(days=900)
            else:
                # Standardní monthly - ~9 měsíců předem
                return expiration - timedelta(days=270)
        else:
            # Weekly opce - ~8 týdnů předem
            return expiration - timedelta(days=56)

    def get_expirations(self, symbol):
        """Získá všechny expirace pro symbol"""
        url = f"{self.base_url}/option/list/expirations"
        response = self.api_call_with_retry(url, {'symbol': symbol}, timeout=30)
        
        if response is None:
            logger.warning(f"Nepodařilo se získat expirace pro {symbol}")
            return []
        
        try:
            df = pd.read_csv(StringIO(response.text))
            if 'expiration' in df.columns and len(df) > 0:
                return sorted(pd.to_datetime(df['expiration']).tolist())
            return []
        except Exception as e:
            logger.error(f"Chyba při parsování expirací pro {symbol}: {e}")
            return []
    
    def get_strikes_for_expiration(self, symbol, expiration):
        """Získá strikes pro danou expiraci"""
        url = f"{self.base_url}/option/list/strikes"
        params = {
            'symbol': symbol,
            'expiration': expiration.strftime('%Y%m%d')
        }
        response = self.api_call_with_retry(url, params, timeout=30)
        
        if response is None:
            logger.debug(f"Žádné strikes pro {symbol} exp {expiration.date()}")
            return []
        
        try:
            df = pd.read_csv(StringIO(response.text))
            if 'strike' in df.columns and len(df) > 0:
                return sorted([float(s) for s in df['strike'].unique()])
            return []
        except Exception as e:
            logger.error(f"Chyba při parsování strikes: {e}")
            return []

    def get_underlying_with_strike(self, symbol, expiration, strike, date):
        cache_key = f"{symbol}_{date.strftime('%Y%m%d')}"
        if cache_key in self.underlying_cache: 
            return self.underlying_cache[cache_key]
        
        url = f"{self.base_url}/option/history/greeks/first_order"
        params = {
            'symbol': symbol, 
            'expiration': expiration.strftime('%Y%m%d'),
            'strike': f"{strike:.3f}", 
            'right': 'call', 
            'date': date.strftime('%Y%m%d'), 
            'interval': '1m'
        }
        
        response = self.api_call_with_retry(url, params, timeout=20)
        
        if response is None:
            return None
            
        try:
            df = pd.read_csv(StringIO(response.text))
            if 'underlying_price' in df.columns:
                res = df[['timestamp', 'underlying_price']].copy()
                self.underlying_cache[cache_key] = res
                return res
        except Exception as e:
            logger.debug(f"Chyba při získávání underlying: {e}")
        
        return None

    def download_bulk_ohlc_day(self, symbol, expiration, date):
        url = f"{self.base_url}/option/history/ohlc"
        params = {
            'symbol': symbol, 
            'expiration': expiration.strftime('%Y%m%d'),
            'strike': '*', 
            'right': 'both', 
            'date': date.strftime('%Y%m%d'), 
            'interval': '1m'
        }
        
        response = self.api_call_with_retry(url, params, timeout=45)
        
        if response is None:
            return None
            
        try:
            df = pd.read_csv(StringIO(response.text))
            if len(df) > 0:
                df['symbol'] = symbol
                df['expiration'] = expiration
                df['date'] = date
                return df
        except Exception as e:
            logger.debug(f"Chyba při parsování OHLC: {e}")
        
        return None

    def download_bulk_oi_day(self, symbol, expiration, date):
        url = f"{self.base_url}/option/history/open_interest"
        params = {
            'symbol': symbol, 
            'expiration': expiration.strftime('%Y%m%d'),
            'strike': '*', 
            'right': 'both', 
            'date': date.strftime('%Y%m%d')
        }
        
        response = self.api_call_with_retry(url, params, timeout=45)
        
        if response is None:
            return None
            
        try:
            df = pd.read_csv(StringIO(response.text))
            if 'open_interest' in df.columns:
                if 'timestamp' in df.columns:
                    return df.groupby(['strike', 'right'])['open_interest'].last().reset_index()
                return df[['strike', 'right', 'open_interest']].copy()
        except Exception as e:
            logger.debug(f"Chyba při parsování OI: {e}")
        
        return None

    def process_dataframe(self, df, underlying_df, oi_df, moneyness_range, apply_moneyness_filter=True):
        """Garantuje konzistentní schéma"""
        merged = df.merge(underlying_df[['timestamp', 'underlying_price']], on='timestamp', how='left')
        
        if oi_df is not None and len(oi_df) > 0:
            merged = merged.merge(oi_df[['strike', 'right', 'open_interest']], 
                                 on=['strike', 'right'], how='left')
        else:
            merged['open_interest'] = np.nan
        
        if apply_moneyness_filter and 'underlying_price' in merged.columns and 'strike' in merged.columns:
            avg_price = merged['underlying_price'].mean()
            if pd.notna(avg_price):
                min_s, max_s = avg_price * moneyness_range[0], avg_price * moneyness_range[1]
                merged = merged[(merged['strike'] >= min_s) & (merged['strike'] <= max_s)]
        
        # Numerické sloupce
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'strike', 'open_interest', 'underlying_price']
        for col in numeric_cols:
            if col in merged.columns:
                merged[col] = merged[col].astype('float64')
        
        # OPRAVA: Normalizovat všechny datetime sloupce na stejnou přesnost (us)
        # To zabrání PyArrow schema mismatch errorům
        datetime_cols = ['expiration', 'date']
        for col in datetime_cols:
            if col in merged.columns:
                # Převést na datetime pokud ještě není, pak na microseconds
                merged[col] = pd.to_datetime(merged[col]).astype('datetime64[us]')
                
        return merged

    def download_year_chunk(self, symbol, year, moneyness_range=(0.7, 1.3), 
                           max_dte=365, apply_moneyness_filter=True):
        """
        Stáhne data pro jeden symbol a rok.
        
        Args:
            symbol: Ticker symbol
            year: Rok ke stažení
            moneyness_range: Rozsah moneyness (min, max)
            max_dte: Maximální DTE pro stahování
            apply_moneyness_filter: Zda aplikovat moneyness filtr
        """
        if self.is_chunk_complete(symbol, year):
            logger.info(f"✅ {symbol} {year} již hotovo.")
            return True

        final_file = self.output_dir / f"{symbol}_{year}.parquet"
        if final_file.exists():
            logger.warning(f"⚠️ Nalezen neúplný soubor pro {year}, začínám znovu.")
            final_file.unlink()

        year_start = datetime(year, 1, 1)
        year_end = datetime(year, 12, 31)
        
        logger.info(f"\n🚀 START: {symbol} {year}")
        
        # Získat VŠECHNY expirace
        logger.info("  Stahuji seznam expirací...")
        all_expirations = self.get_expirations(symbol)
        
        # Filtrovat expirace podle listing date a max_dte
        expirations = []
        for exp in all_expirations:
            # Použít lepší odhad listing date
            earliest_listing = self.estimate_listing_date(exp)
            
            # Bereme ji, pokud se začala obchodovat před/během tohoto roku
            # A expiruje během/po tomto roce
            if earliest_listing <= year_end and exp >= year_start:
                expirations.append(exp)
        
        if len(expirations) == 0:
            logger.info(f"  Žádné expirace s DTE <= {max_dte} pro tento rok.")
            self.mark_chunk_complete(symbol, year)
            return True
            
        logger.info(f"  Nalezeno {len(expirations)} expirací (max DTE: {max_dte}).")

        writer = None
        rows_written = 0
        days_with_data = 0
        expirations_with_data = 0
        
        try:
            for i, expiration in enumerate(expirations, 1):
                exp_strikes = self.get_strikes_for_expiration(symbol, expiration)
                if len(exp_strikes) == 0: 
                    logger.info(f"  Exp [{i}/{len(expirations)}] {expiration.date()} ... ⊘ Bez strikes")
                    continue
                
                median_strike = np.median(exp_strikes)
                
                # Určit trading range pro tuto expiraci
                earliest_trading = self.estimate_listing_date(expiration)
                
                # Stahujeme jen data Z TOHOTO ROKU
                current_date = max(year_start, earliest_trading)
                loop_end = min(year_end, expiration)
                
                # Kontrola: Pokud celý trading range je mimo tento rok, skip
                if current_date > year_end or loop_end < year_start:
                    logger.debug(f"  Exp {expiration.date()} mimo rok")
                    continue
                
                logger.info(f"  Exp [{i}/{len(expirations)}] {expiration.date()} ...")
                
                chunk_buffer = []
                days_checked = 0
                days_had_data = 0
                first_data_found = False
                consecutive_empty_days = 0
                
                while current_date <= loop_end:
                    if current_date.weekday() >= 5:
                        current_date += timedelta(days=1)
                        continue
                    
                    # Kontrola DTE pro TENTO den
                    dte = (expiration - current_date).days
                    if dte > max_dte:
                        current_date += timedelta(days=1)
                        continue
                    
                    days_checked += 1
                    
                    # OHLC
                    ohlc = self.download_bulk_ohlc_day(symbol, expiration, current_date)
                    
                    if ohlc is not None and len(ohlc) > 0:
                        first_data_found = True
                        consecutive_empty_days = 0  # Reset counter
                        
                        # Underlying
                        und = self.get_underlying_with_strike(symbol, expiration, median_strike, current_date)
                        
                        if und is not None and len(und) > 0:
                            # OI (může být None)
                            oi = self.download_bulk_oi_day(symbol, expiration, current_date)
                            
                            processed_df = self.process_dataframe(
                                ohlc, und, oi, moneyness_range, 
                                apply_moneyness_filter=apply_moneyness_filter
                            )
                            
                            if not processed_df.empty:
                                chunk_buffer.append(processed_df)
                                days_had_data += 1
                    else:
                        consecutive_empty_days += 1
                        
                        # Robustnější early exit - 10 dní místo 5
                        # + kontrola že jsme už někdy měli data
                        if first_data_found and consecutive_empty_days >= 10:
                            logger.debug(f"    Early exit po {consecutive_empty_days} prázdných dnech")
                            break
                    
                    current_date += timedelta(days=1)
                
                # Uložit expiraci
                if chunk_buffer:
                    full_chunk_df = pd.concat(chunk_buffer, ignore_index=True)
                    table = pa.Table.from_pandas(full_chunk_df)
                    
                    if writer is None:
                        writer = pq.ParquetWriter(final_file, table.schema, compression='snappy')
                    
                    writer.write_table(table)
                    rows_written += len(full_chunk_df)
                    days_with_data += days_had_data
                    expirations_with_data += 1
                    
                    logger.info(f"    ✓ {len(full_chunk_df):,} řádků ({days_had_data}/{days_checked} dní)")
                    
                    del full_chunk_df, table, chunk_buffer
                    gc.collect()
                else:
                    logger.info(f"    ✗ Bez dat ({days_checked} dní zkontrolováno)")

                # Čištění cache každých 50 expirací
                if i % 50 == 0 and len(self.underlying_cache) > 20:
                    self.underlying_cache.clear()
                    gc.collect()

            if writer:
                writer.close()
                logger.info(f"\n✅ HOTOVO {year}: {rows_written:,} řádků")
                logger.info(f"   {expirations_with_data}/{len(expirations)} expirací s daty")
                logger.info(f"   {days_with_data} obchodních dní")
            else:
                logger.info(f"\nℹ️ Rok {year} prázdný.")
                
            self.mark_chunk_complete(symbol, year)
            return True

        except Exception as e:
            logger.error(f"\n❌ CHYBA {symbol} {year}: {e}")
            import traceback
            traceback.print_exc()
            if writer: 
                writer.close()
            return False

    def download_all(self, years_back=10, moneyness_range=(0.7, 1.3), max_dte=365, 
                    apply_moneyness_filter=True):
        """
        Stáhne data pro všechny symboly a roky.
        
        Args:
            years_back: Kolik let zpět stahovat
            moneyness_range: Rozsah moneyness filtru
            max_dte: Maximální DTE (dny do expirace) pro stahování
                    365 = jen expirace do 1 roku
                    730 = až 2 roky (LEAPS)
            apply_moneyness_filter: Zda aplikovat moneyness filtr
        """
        if not self.connect(): 
            logger.error("Nelze se připojit k ThetaData Terminal")
            return
        
        end_year = datetime.now().year
        start_year = end_year - years_back
        years = range(start_year, end_year + 1)
        
        total_symbols = len(self.symbols)
        total_years = len(list(years))
        
        logger.info(f"Začínám stahování: {total_symbols} symbolů × {total_years} let")
        
        for sym_idx, symbol in enumerate(self.symbols, 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Symbol [{sym_idx}/{total_symbols}]: {symbol}")
            logger.info(f"{'='*50}")
            
            for year in range(start_year, end_year + 1):
                self.download_year_chunk(
                    symbol, year, moneyness_range, max_dte, apply_moneyness_filter
                )
                gc.collect()
        
        logger.info("\n" + "="*50)
        logger.info("STAHOVÁNÍ DOKONČENO")
        logger.info("="*50)


if __name__ == "__main__":
    dl = OptionsDataDownloader(output_dir="options_full_history_fixed")
    
    # Test s jedním symbolem
    dl.symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'IWM', 'AMZN', 'TSLA', 'NVDA', 'COIN', 'AMD','JPM', 'SMCI', 'GLD', 'TLT'] 
    
    # Stáhnout 15 let s max DTE 365 dní
    dl.download_all(years_back=15, max_dte=365, apply_moneyness_filter=True)