#!/usr/bin/env python3
"""
earnings_features.py

Earnings Features pro ML/LSTM/PPO

DŮLEŽITÉ: Earnings jsou kritické pro options trading!
- IV stoupá před earnings (Implied Volatility)
- IV crush po earnings (IV padá)
- Historický pohyb určuje expected move

Features které přidáváme:
1. days_to_earnings - kolik dní do earnings
2. days_since_earnings - kolik dní od posledních earnings
3. iv_percentile_pre_earnings - IV rank před earnings
4. expected_move - očekávaný pohyb z options
5. historical_earnings_move_avg - průměrný historický pohyb
6. historical_earnings_move_std - volatilita pohybu
7. earnings_surprise_avg - průměrné překvapení (beat/miss)
8. is_earnings_week - binární flag
9. earnings_hour - BMO (before market open) / AMC (after market close)

Usage:
    from earnings_features import EarningsFeatureEngineer
    
    engineer = EarningsFeatureEngineer()
    df = engineer.add_features(df, symbol='AAPL')
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# EARNINGS DATA
# =============================================================================

@dataclass
class EarningsEvent:
    """Jeden earnings event."""
    symbol: str
    date: date
    time: str  # 'BMO' (before market open) nebo 'AMC' (after market close)
    
    # Expectations vs Actual
    eps_estimate: Optional[float] = None
    eps_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_actual: Optional[float] = None
    
    # Price movement
    price_before: Optional[float] = None
    price_after: Optional[float] = None
    move_percent: Optional[float] = None
    
    @property
    def surprise_percent(self) -> Optional[float]:
        """EPS surprise jako procento."""
        if self.eps_estimate and self.eps_actual and self.eps_estimate != 0:
            return (self.eps_actual - self.eps_estimate) / abs(self.eps_estimate) * 100
        return None
    
    @property
    def beat_miss(self) -> Optional[str]:
        """Beat, miss, nebo meet."""
        surprise = self.surprise_percent
        if surprise is None:
            return None
        if surprise > 2:
            return 'BEAT'
        elif surprise < -2:
            return 'MISS'
        return 'MEET'


class EarningsCalendar:
    """
    Earnings kalendář s historickými daty.
    
    Zdroje dat:
    - Yahoo Finance (yfinance)
    - Alpha Vantage
    - Polygon.io
    - Lokální cache
    """
    
    def __init__(self, cache_dir: str = "data/earnings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, List[EarningsEvent]] = {}
    
    def get_earnings_history(self, symbol: str, 
                              years: int = 10) -> List[EarningsEvent]:
        """
        Získá historii earnings pro symbol.
        
        Args:
            symbol: Ticker symbol
            years: Kolik let historie
            
        Returns:
            Seznam EarningsEvent seřazený od nejstaršího
        """
        # Check cache
        if symbol in self._cache:
            return self._cache[symbol]
        
        # Try to load from file
        cache_file = self.cache_dir / f"{symbol}_earnings.json"
        if cache_file.exists():
            events = self._load_from_cache(cache_file)
            self._cache[symbol] = events
            return events
        
        # Fetch from source
        events = self._fetch_earnings(symbol, years)
        
        # Save to cache
        self._save_to_cache(cache_file, events)
        self._cache[symbol] = events
        
        return events
    
    def _fetch_earnings(self, symbol: str, years: int) -> List[EarningsEvent]:
        """Fetch earnings z dostupných zdrojů."""
        events = []
        
        # Try yfinance first
        try:
            events = self._fetch_from_yfinance(symbol, years)
            if events:
                logger.info(f"Loaded {len(events)} earnings for {symbol} from yfinance")
                return events
        except Exception as e:
            logger.warning(f"yfinance failed for {symbol}: {e}")
        
        # Fallback: generate approximate dates (quarterly)
        events = self._generate_approximate_earnings(symbol, years)
        logger.warning(f"Using approximate earnings for {symbol}")
        
        return events
    
    def _fetch_from_yfinance(self, symbol: str, years: int) -> List[EarningsEvent]:
        """Fetch z yfinance."""
        try:
            import yfinance as yf
        except ImportError:
            logger.warning("yfinance not installed")
            return []
        
        ticker = yf.Ticker(symbol)
        
        events = []
        
        # Get earnings dates
        try:
            earnings_dates = ticker.earnings_dates
            if earnings_dates is not None and not earnings_dates.empty:
                for idx, row in earnings_dates.iterrows():
                    event_date = idx.date() if hasattr(idx, 'date') else idx
                    
                    events.append(EarningsEvent(
                        symbol=symbol,
                        date=event_date,
                        time='AMC',  # Default, yfinance doesn't always provide
                        eps_estimate=row.get('EPS Estimate'),
                        eps_actual=row.get('Reported EPS'),
                    ))
        except Exception as e:
            logger.warning(f"Could not get earnings_dates: {e}")
        
        # Get earnings history for more detail
        try:
            earnings_hist = ticker.earnings_history
            if earnings_hist is not None and not earnings_hist.empty:
                # Merge with existing events or add new ones
                for idx, row in earnings_hist.iterrows():
                    event_date = idx.date() if hasattr(idx, 'date') else idx
                    
                    # Find existing event or create new
                    existing = next((e for e in events if e.date == event_date), None)
                    
                    if existing:
                        existing.eps_estimate = row.get('epsEstimate', existing.eps_estimate)
                        existing.eps_actual = row.get('epsActual', existing.eps_actual)
                    else:
                        events.append(EarningsEvent(
                            symbol=symbol,
                            date=event_date,
                            time='AMC',
                            eps_estimate=row.get('epsEstimate'),
                            eps_actual=row.get('epsActual'),
                        ))
        except:
            pass
        
        # Sort by date
        events.sort(key=lambda e: e.date)
        
        # Filter to requested years
        cutoff = date.today() - timedelta(days=years * 365)
        events = [e for e in events if e.date >= cutoff]
        
        return events
    
    def _generate_approximate_earnings(self, symbol: str, 
                                        years: int) -> List[EarningsEvent]:
        """Generuje přibližné earnings dates (kvartálně)."""
        events = []
        
        # Většina firem reportuje kvartálně
        # Přibližné měsíce: Leden, Duben, Červenec, Říjen
        today = date.today()
        
        for year_offset in range(years):
            year = today.year - year_offset
            
            for month in [1, 4, 7, 10]:
                # Přibližně 3. týden v měsíci
                event_date = date(year, month, 20)
                
                if event_date <= today:
                    events.append(EarningsEvent(
                        symbol=symbol,
                        date=event_date,
                        time='AMC'
                    ))
        
        events.sort(key=lambda e: e.date)
        return events
    
    def _load_from_cache(self, path: Path) -> List[EarningsEvent]:
        """Load z cache souboru."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        events = []
        for item in data:
            events.append(EarningsEvent(
                symbol=item['symbol'],
                date=datetime.fromisoformat(item['date']).date(),
                time=item.get('time', 'AMC'),
                eps_estimate=item.get('eps_estimate'),
                eps_actual=item.get('eps_actual'),
                revenue_estimate=item.get('revenue_estimate'),
                revenue_actual=item.get('revenue_actual'),
                move_percent=item.get('move_percent'),
            ))
        
        return events
    
    def _save_to_cache(self, path: Path, events: List[EarningsEvent]):
        """Save do cache souboru."""
        data = []
        for event in events:
            data.append({
                'symbol': event.symbol,
                'date': event.date.isoformat(),
                'time': event.time,
                'eps_estimate': event.eps_estimate,
                'eps_actual': event.eps_actual,
                'revenue_estimate': event.revenue_estimate,
                'revenue_actual': event.revenue_actual,
                'move_percent': event.move_percent,
            })
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_next_earnings(self, symbol: str, 
                          from_date: date = None) -> Optional[EarningsEvent]:
        """Získá příští earnings po daném datu."""
        from_date = from_date or date.today()
        events = self.get_earnings_history(symbol)
        
        for event in events:
            if event.date > from_date:
                return event
        
        return None
    
    def get_previous_earnings(self, symbol: str,
                               from_date: date = None) -> Optional[EarningsEvent]:
        """Získá předchozí earnings před daným datem."""
        from_date = from_date or date.today()
        events = self.get_earnings_history(symbol)
        
        previous = None
        for event in events:
            if event.date < from_date:
                previous = event
            else:
                break
        
        return previous


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

class EarningsFeatureEngineer:
    """
    Vytváří earnings features pro ML/LSTM/PPO.
    
    Features:
    - days_to_earnings: Dny do příštího earnings (0 = dnes, -1 = včera bylo)
    - days_since_earnings: Dny od posledního earnings
    - is_earnings_week: 1 pokud earnings tento týden
    - is_earnings_day: 1 pokud earnings dnes
    - earnings_expected_move: Očekávaný pohyb z ATM straddle
    - historical_move_avg: Průměrný historický pohyb po earnings
    - historical_move_std: Std dev historického pohybu
    - earnings_beat_rate: Jak často firma překonává expectations
    - iv_rank_normalized: IV rank normalizovaný (pro srovnání)
    """
    
    def __init__(self):
        self.calendar = EarningsCalendar()
        self._move_cache: Dict[str, Dict] = {}
    
    def add_features(self, df: pd.DataFrame, 
                     symbol: str) -> pd.DataFrame:
        """
        Přidá earnings features do DataFrame.
        
        Args:
            df: DataFrame s 'timestamp' nebo 'date' sloupcem
            symbol: Ticker symbol
            
        Returns:
            DataFrame s novými sloupci
        """
        df = df.copy()
        
        # Ensure we have date column
        if 'date' not in df.columns:
            if 'timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp']).dt.date
            else:
                logger.warning("No date/timestamp column found")
                return df
        
        # Get earnings history
        earnings = self.calendar.get_earnings_history(symbol)
        
        if not earnings:
            logger.warning(f"No earnings data for {symbol}")
            # Add empty columns
            for col in ['days_to_earnings', 'days_since_earnings', 
                       'is_earnings_week', 'is_earnings_day']:
                df[col] = 0
            return df
        
        # Calculate historical stats
        historical_stats = self._calculate_historical_stats(earnings)
        
        # Add features for each row
        logger.info(f"Adding earnings features for {symbol} ({len(earnings)} events)")
        
        # Vectorized operations where possible
        earnings_dates = [e.date for e in earnings]
        
        # Create lookup for faster processing
        df['_date_obj'] = pd.to_datetime(df['date']).dt.date
        
        # Days to/since earnings
        df['days_to_earnings'] = df['_date_obj'].apply(
            lambda d: self._days_to_next_earnings(d, earnings_dates)
        )
        
        df['days_since_earnings'] = df['_date_obj'].apply(
            lambda d: self._days_since_last_earnings(d, earnings_dates)
        )
        
        # Binary flags
        df['is_earnings_day'] = df['days_to_earnings'].apply(lambda x: 1 if x == 0 else 0)
        df['is_earnings_week'] = df['days_to_earnings'].apply(lambda x: 1 if 0 <= x <= 5 else 0)
        
        # Earnings proximity features (continuous, better for ML)
        # Transform to 0-1 scale where 1 = very close to earnings
        df['earnings_proximity'] = df['days_to_earnings'].apply(
            lambda x: max(0, 1 - abs(x) / 30) if x is not None else 0
        )
        
        # Historical stats (same for all rows of this symbol)
        df['historical_move_avg'] = historical_stats.get('move_avg', 0)
        df['historical_move_std'] = historical_stats.get('move_std', 0)
        df['earnings_beat_rate'] = historical_stats.get('beat_rate', 0.5)
        df['avg_surprise_pct'] = historical_stats.get('avg_surprise', 0)
        
        # Expected move (simplified - would need options data for accurate)
        # Approximation: historical_move_avg * IV_factor
        if 'iv_rank' in df.columns:
            df['earnings_expected_move'] = (
                df['historical_move_avg'] * 
                (1 + (df['iv_rank'] - 50) / 100)
            )
        else:
            df['earnings_expected_move'] = df['historical_move_avg']
        
        # Clean up
        df = df.drop(columns=['_date_obj'], errors='ignore')
        
        logger.info(f"Added {8} earnings features for {symbol}")
        
        return df
    
    def _days_to_next_earnings(self, current_date: date, 
                                earnings_dates: List[date]) -> int:
        """Počet dní do příštího earnings."""
        for earn_date in earnings_dates:
            if earn_date >= current_date:
                return (earn_date - current_date).days
        
        # No future earnings found, assume quarterly (~90 days)
        return 90
    
    def _days_since_last_earnings(self, current_date: date,
                                   earnings_dates: List[date]) -> int:
        """Počet dní od posledního earnings."""
        last_date = None
        for earn_date in earnings_dates:
            if earn_date < current_date:
                last_date = earn_date
            else:
                break
        
        if last_date:
            return (current_date - last_date).days
        
        return 90  # Default
    
    def _calculate_historical_stats(self, 
                                     earnings: List[EarningsEvent]) -> Dict:
        """Vypočítá historické statistiky z earnings."""
        moves = []
        surprises = []
        beats = 0
        total_with_data = 0
        
        for event in earnings:
            if event.move_percent is not None:
                moves.append(abs(event.move_percent))
            
            surprise = event.surprise_percent
            if surprise is not None:
                surprises.append(surprise)
                total_with_data += 1
                if surprise > 0:
                    beats += 1
        
        return {
            'move_avg': np.mean(moves) if moves else 5.0,  # Default 5% move
            'move_std': np.std(moves) if len(moves) > 1 else 2.0,
            'beat_rate': beats / total_with_data if total_with_data > 0 else 0.5,
            'avg_surprise': np.mean(surprises) if surprises else 0,
        }
    
    def add_features_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Přidá earnings features pro DataFrame s více symboly.
        
        Expects 'symbol' column in DataFrame.
        """
        if 'symbol' not in df.columns:
            logger.warning("No 'symbol' column, cannot add batch features")
            return df
        
        result_dfs = []
        
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol].copy()
            symbol_df = self.add_features(symbol_df, symbol)
            result_dfs.append(symbol_df)
        
        return pd.concat(result_dfs, ignore_index=True)


# =============================================================================
# MEGA-CAP EARNINGS (pro SPY/QQQ)
# =============================================================================

# Mega-cap stocks that significantly move SPY/QQQ
MEGA_CAP_STOCKS = {
    'SPY': ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'BRK.B'],
    'QQQ': ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'AVGO'],
}


class IndexEarningsFeatures:
    """
    Earnings features pro indexy (SPY, QQQ).
    
    Index se pohybuje když mega-cap stocks reportují earnings.
    """
    
    def __init__(self):
        self.calendar = EarningsCalendar()
    
    def add_features(self, df: pd.DataFrame, 
                     index_symbol: str = 'SPY') -> pd.DataFrame:
        """
        Přidá mega-cap earnings features pro index.
        
        Features:
        - megacap_earnings_this_week: Počet mega-cap earnings tento týden
        - next_megacap_days: Dny do příštího mega-cap earnings
        - megacap_reporting: Seznam firem reportujících (jako string)
        """
        df = df.copy()
        
        mega_caps = MEGA_CAP_STOCKS.get(index_symbol, MEGA_CAP_STOCKS['SPY'])
        
        # Get earnings for all mega-caps
        all_earnings: Dict[str, List[EarningsEvent]] = {}
        for stock in mega_caps:
            all_earnings[stock] = self.calendar.get_earnings_history(stock)
        
        # Ensure date column
        if 'date' not in df.columns:
            if 'timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp']).dt.date
            else:
                return df
        
        df['_date_obj'] = pd.to_datetime(df['date']).dt.date
        
        # For each row, check mega-cap earnings
        def get_megacap_features(current_date):
            this_week = 0
            next_days = 90
            reporting = []
            
            week_start = current_date - timedelta(days=current_date.weekday())
            week_end = week_start + timedelta(days=6)
            
            for stock, events in all_earnings.items():
                for event in events:
                    # This week?
                    if week_start <= event.date <= week_end:
                        this_week += 1
                        if event.date >= current_date:
                            reporting.append(stock)
                    
                    # Next earnings
                    if event.date >= current_date:
                        days = (event.date - current_date).days
                        if days < next_days:
                            next_days = days
                        break
            
            return this_week, next_days, ','.join(reporting[:3])
        
        features = df['_date_obj'].apply(get_megacap_features)
        
        df['megacap_earnings_this_week'] = features.apply(lambda x: x[0])
        df['next_megacap_days'] = features.apply(lambda x: x[1])
        df['megacap_reporting'] = features.apply(lambda x: x[2])
        
        # Binary: any mega-cap this week?
        df['is_megacap_week'] = (df['megacap_earnings_this_week'] > 0).astype(int)
        
        df = df.drop(columns=['_date_obj'], errors='ignore')
        
        return df


# =============================================================================
# INTEGRATION WITH SATURDAY TRAINING
# =============================================================================

def add_earnings_to_training_data(data_dir: str = "data/enriched"):
    """
    Přidá earnings features do všech training parquet souborů.
    
    Volat v saturday_training_full.py před ML tréninkem.
    """
    from pathlib import Path
    
    data_dir = Path(data_dir)
    engineer = EarningsFeatureEngineer()
    index_engineer = IndexEarningsFeatures()
    
    processed = 0
    
    for parquet_file in data_dir.glob("*_1min_vanna.parquet"):
        symbol = parquet_file.stem.split('_')[0]
        
        try:
            df = pd.read_parquet(parquet_file)
            
            # Add earnings features
            if symbol in ['SPY', 'QQQ']:
                df = index_engineer.add_features(df, symbol)
            
            df = engineer.add_features(df, symbol)
            
            # Save back
            df.to_parquet(parquet_file, index=False)
            processed += 1
            
            logger.info(f"Added earnings features to {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to add earnings to {symbol}: {e}")
    
    logger.info(f"Processed {processed} files with earnings features")
    return processed


# =============================================================================
# CLI / TEST
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Earnings Feature Engineering")
    print("=" * 60)
    
    # Test
    engineer = EarningsFeatureEngineer()
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='D')
    df = pd.DataFrame({
        'date': dates,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'iv_rank': np.random.uniform(20, 80, len(dates))
    })
    
    # Add features
    df = engineer.add_features(df, 'AAPL')
    
    print("\nSample output:")
    print(df[['date', 'days_to_earnings', 'days_since_earnings', 
              'is_earnings_week', 'earnings_proximity', 
              'historical_move_avg']].head(20))
    
    # Test index features
    print("\n" + "=" * 60)
    print("Index (SPY) Mega-Cap Earnings")
    print("=" * 60)
    
    index_eng = IndexEarningsFeatures()
    df_spy = index_eng.add_features(df.copy(), 'SPY')
    
    print(df_spy[['date', 'megacap_earnings_this_week', 
                  'next_megacap_days', 'is_megacap_week']].head(20))
