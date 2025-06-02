"""
Validation classes for ticker symbols and downloaded data.
"""

import logging
import re
from typing import List

import pandas as pd


class TickerValidator:
    """Handles ticker validation and cleaning."""
    
    VALID_TICKER_PATTERN = re.compile(r'^[A-Z]{1,5}(-[A-Z])?$')
    
    @classmethod
    def validate(cls, ticker: str) -> bool:
        """Validate ticker symbol format."""
        if not ticker or len(ticker) > 6:
            return False
        return bool(cls.VALID_TICKER_PATTERN.match(ticker.upper()))
    
    @classmethod
    def clean(cls, ticker: str) -> str:
        """Clean and standardize ticker symbol for yfinance."""
        if not ticker:
            return ""
        
        # Remove quotes and whitespace
        ticker = str(ticker).strip().strip('"\'').upper()
        
        # Replace periods with hyphens for yfinance compatibility
        ticker = ticker.replace(".", "-")
        
        return ticker


class DataValidator:
    """Handles data validation for downloaded stock data."""
    
    REQUIRED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
    PRICE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Adj Close']
    
    @classmethod
    def validate(cls, df: pd.DataFrame, ticker: str, min_points: int, logger: logging.Logger) -> bool:
        """Validate downloaded data quality."""
        if df.empty:
            logger.warning(f"Empty data for {ticker}")
            return False
        
        if len(df) < min_points:
            logger.warning(f"Insufficient data for {ticker}: {len(df)} points")
            return False
        
        # Check for essential columns
        missing_cols = [col for col in cls.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns for {ticker}: {missing_cols}")
            return False
        
        return True
    
    @classmethod
    def clean_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and format dataframe for consistent output."""
        clean_df = df.copy()
        
        # Handle yfinance multi-level columns if present
        if hasattr(clean_df.columns, 'levels'):
            clean_df.columns = clean_df.columns.get_level_values(0)
        
        # Ensure standard column order
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        available_columns = [col for col in expected_columns if col in clean_df.columns]
        if available_columns:
            clean_df = clean_df[available_columns]
        
        # Format data types and precision
        clean_df.index.name = 'Date'
        
        # Round price columns to 4 decimal places
        for col in cls.PRICE_COLUMNS:
            if col in clean_df.columns:
                clean_df[col] = clean_df[col].round(4)
        
        # Ensure Volume is integer
        if 'Volume' in clean_df.columns:
            clean_df['Volume'] = clean_df['Volume'].astype('int64')
        
        # Remove rows with all NaN values
        clean_df = clean_df.dropna(how='all')
        
        return clean_df 