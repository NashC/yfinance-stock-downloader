"""
Validation utilities for ticker symbols and data quality.
Enhanced with yfinance symbol existence validation and historical data detection.
"""

import logging
import re
from typing import List, Set, Dict, Tuple, Optional
from datetime import datetime, timedelta
import warnings

import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Suppress yfinance warnings for cleaner output during large batch processing
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*possibly delisted.*')
warnings.filterwarnings('ignore', message='.*Period.*is invalid.*')
warnings.filterwarnings('ignore', message='.*no timezone found.*')
warnings.filterwarnings('ignore', message='.*no price data found.*')


class TickerValidator:
    """Utility class for validating ticker symbols."""
    
    # Common patterns for valid ticker symbols
    VALID_PATTERNS = [
        r'^[A-Z]{1,5}$',           # Standard 1-5 letter tickers (AAPL, MSFT)
        r'^[A-Z]{1,4}\.[A-Z]{1,2}$',  # International tickers (BRK.A, BRK.B)
        r'^[A-Z]{1,5}-[A-Z]$',     # Preferred shares (BRK-A)
        r'^[A-Z]{1,5}\^[A-Z]$',    # Some preferred share formats
    ]
    
    # Known invalid patterns
    INVALID_PATTERNS = [
        r'^\d+$',                  # Pure numbers
        r'^[^A-Z]',               # Doesn't start with letter
        r'[^A-Z0-9\.\-\^]',       # Contains invalid characters
    ]
    
    @classmethod
    def clean(cls, ticker: str) -> str:
        """Clean and normalize a ticker symbol."""
        if not isinstance(ticker, str):
            return ""
        
        # Basic cleaning
        cleaned = str(ticker).strip().upper()
        
        # Remove common prefixes/suffixes that might be metadata
        cleaned = re.sub(r'^(TICKER:|SYMBOL:)', '', cleaned)
        cleaned = re.sub(r'\s+', '', cleaned)  # Remove all whitespace
        
        return cleaned
    
    @classmethod
    def validate(cls, ticker: str) -> bool:
        """Validate ticker symbol format (basic validation)."""
        if not ticker or len(ticker) == 0:
            return False
        
        # Check against invalid patterns first
        for pattern in cls.INVALID_PATTERNS:
            if re.search(pattern, ticker):
                return False
        
        # Check against valid patterns
        for pattern in cls.VALID_PATTERNS:
            if re.match(pattern, ticker):
                return True
        
        return False
    
    @classmethod
    def get_earliest_available_date(cls, ticker: str, timeout: int = 10) -> Optional[datetime]:
        """
        Determine the earliest available date for a symbol by testing different periods.
        
        Args:
            ticker: The ticker symbol to check
            timeout: Timeout in seconds for each request
            
        Returns:
            Earliest available date as datetime, or None if no data available
        """
        try:
            yf_ticker = yf.Ticker(ticker)
            
            # Test periods in order from longest to shortest
            test_periods = ["max", "20y", "10y", "5y", "2y", "1y", "6mo", "3mo", "1mo"]
            
            for period in test_periods:
                try:
                    data = yf_ticker.history(period=period, timeout=timeout)
                    if not data.empty:
                        earliest_date = data.index.min()
                        return earliest_date.to_pydatetime() if hasattr(earliest_date, 'to_pydatetime') else earliest_date
                except Exception:
                    continue
            
            # If no period worked, try a specific date range going back far
            try:
                # Try going back to 1970 (Unix epoch)
                start_date = "1970-01-01"
                data = yf_ticker.history(start=start_date, timeout=timeout)
                if not data.empty:
                    earliest_date = data.index.min()
                    return earliest_date.to_pydatetime() if hasattr(earliest_date, 'to_pydatetime') else earliest_date
            except Exception:
                pass
            
            return None
            
        except Exception:
            return None
    
    @classmethod
    def validate_symbol_exists(cls, ticker: str, timeout: int = 10) -> bool:
        """
        Check if a symbol exists in yfinance database by attempting to fetch basic info.
        
        Args:
            ticker: The ticker symbol to validate
            timeout: Timeout in seconds for the validation request
            
        Returns:
            True if symbol exists and has data, False otherwise
        """
        try:
            yf_ticker = yf.Ticker(ticker)
            
            # Try to get basic info - this is faster than downloading full history
            info = yf_ticker.info
            
            # Check if we got meaningful data back
            if not info or len(info) <= 1:
                return False
            
            # Check for common indicators that the symbol exists
            has_basic_info = any(key in info for key in [
                'symbol', 'shortName', 'longName', 'marketCap', 
                'regularMarketPrice', 'currency', 'exchange'
            ])
            
            if not has_basic_info:
                return False
            
            # Additional check: try to get a small amount of recent data
            recent_data = yf_ticker.history(period="5d", timeout=timeout)
            
            # If we get any recent data, the symbol likely exists
            return not recent_data.empty
            
        except Exception as e:
            # Any exception likely means the symbol doesn't exist or is invalid
            return False
    
    @classmethod
    def validate_and_get_date_range(cls, ticker: str, timeout: int = 10) -> Tuple[bool, Optional[datetime]]:
        """
        Validate symbol existence and get the earliest available date in one call.
        
        Args:
            ticker: The ticker symbol to validate
            timeout: Timeout in seconds for the validation request
            
        Returns:
            Tuple of (exists, earliest_date)
        """
        try:
            yf_ticker = yf.Ticker(ticker)
            
            # Try to get the maximum historical data available
            data = yf_ticker.history(period="max", timeout=timeout)
            
            if data.empty:
                return False, None
            
            # If we got data, the symbol exists
            earliest_date = data.index.min()
            earliest_datetime = earliest_date.to_pydatetime() if hasattr(earliest_date, 'to_pydatetime') else earliest_date
            
            return True, earliest_datetime
            
        except Exception:
            return False, None
    
    @classmethod
    def batch_validate_symbols(cls, tickers: List[str], max_workers: int = 5, 
                             timeout: int = 10) -> Tuple[List[str], List[str]]:
        """
        Validate multiple symbols in parallel for existence in yfinance.
        
        Args:
            tickers: List of ticker symbols to validate
            max_workers: Maximum number of parallel validation threads
            timeout: Timeout per symbol validation
            
        Returns:
            Tuple of (valid_symbols, invalid_symbols)
        """
        valid_symbols = []
        invalid_symbols = []
        
        # First do basic format validation
        format_valid_tickers = []
        for ticker in tickers:
            cleaned = cls.clean(ticker)
            if cls.validate(cleaned):
                format_valid_tickers.append(cleaned)
            else:
                invalid_symbols.append(ticker)
        
        if not format_valid_tickers:
            return valid_symbols, invalid_symbols
        
        # Then validate existence in yfinance database
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit validation tasks
            future_to_ticker = {
                executor.submit(cls.validate_symbol_exists, ticker, timeout): ticker
                for ticker in format_valid_tickers
            }
            
            # Process results as they complete
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    if future.result():
                        valid_symbols.append(ticker)
                    else:
                        invalid_symbols.append(ticker)
                except Exception:
                    invalid_symbols.append(ticker)
                
                # Small delay to avoid overwhelming the API
                time.sleep(0.1)
        
        return valid_symbols, invalid_symbols
    
    @classmethod
    def batch_validate_with_date_ranges(cls, tickers: List[str], max_workers: int = 3, 
                                       timeout: int = 10) -> Tuple[Dict[str, datetime], List[str]]:
        """
        Validate multiple symbols and get their earliest available dates in parallel.
        
        Args:
            tickers: List of ticker symbols to validate
            max_workers: Maximum number of parallel validation threads
            timeout: Timeout per symbol validation
            
        Returns:
            Tuple of (valid_symbols_with_dates, invalid_symbols)
            where valid_symbols_with_dates is {symbol: earliest_date}
        """
        valid_symbols_with_dates = {}
        invalid_symbols = []
        
        # First do basic format validation
        format_valid_tickers = []
        for ticker in tickers:
            cleaned = cls.clean(ticker)
            if cls.validate(cleaned):
                format_valid_tickers.append(cleaned)
            else:
                invalid_symbols.append(ticker)
        
        if not format_valid_tickers:
            return valid_symbols_with_dates, invalid_symbols
        
        # Then validate existence and get date ranges
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit validation tasks
            future_to_ticker = {
                executor.submit(cls.validate_and_get_date_range, ticker, timeout): ticker
                for ticker in format_valid_tickers
            }
            
            # Process results as they complete
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    exists, earliest_date = future.result()
                    if exists and earliest_date:
                        valid_symbols_with_dates[ticker] = earliest_date
                    else:
                        invalid_symbols.append(ticker)
                except Exception:
                    invalid_symbols.append(ticker)
                
                # Small delay to avoid overwhelming the API
                time.sleep(0.2)
        
        return valid_symbols_with_dates, invalid_symbols


class DataValidator:
    """Utility class for validating downloaded data quality."""
    
    REQUIRED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    @classmethod
    def validate(cls, df: pd.DataFrame, ticker: str, min_data_points: int = 10, 
                logger: logging.Logger = None) -> bool:
        """Validate that downloaded data meets quality requirements."""
        if df is None or df.empty:
            if logger:
                logger.warning(f"Empty data for {ticker}")
            return False
        
        # Check for required columns
        missing_cols = [col for col in cls.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            if logger:
                logger.warning(f"Missing columns for {ticker}: {missing_cols}")
            return False
        
        # Check minimum data points
        if len(df) < min_data_points:
            if logger:
                logger.warning(f"Insufficient data for {ticker}: {len(df)} < {min_data_points}")
            return False
        
        # Check for all-null columns
        null_cols = [col for col in cls.REQUIRED_COLUMNS if df[col].isnull().all()]
        if null_cols:
            if logger:
                logger.warning(f"All-null columns for {ticker}: {null_cols}")
            return False
        
        return True
    
    @classmethod
    def clean_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare dataframe for saving."""
        if df is None or df.empty:
            return df
        
        # Create a copy to avoid modifying original
        clean_df = df.copy()
        
        # Ensure index is datetime
        if not isinstance(clean_df.index, pd.DatetimeIndex):
            clean_df.index = pd.to_datetime(clean_df.index)
        
        # Sort by date (newest first)
        clean_df = clean_df.sort_index(ascending=False)
        
        # Remove any duplicate dates
        clean_df = clean_df[~clean_df.index.duplicated(keep='first')]
        
        # Forward fill small gaps (up to 3 days) for missing data
        clean_df = clean_df.ffill(limit=3)
        
        # Remove rows where all OHLCV data is null
        ohlcv_cols = [col for col in cls.REQUIRED_COLUMNS if col in clean_df.columns]
        clean_df = clean_df.dropna(subset=ohlcv_cols, how='all')
        
        return clean_df 