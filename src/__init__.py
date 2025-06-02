"""
YFinance Stock Prices - Enhanced Multi-Source Stock Price Data Downloader

A production-ready tool for downloading historical stock price data from Yahoo Finance
supporting multiple data sources with robust error handling and rate limiting.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .downloader import StockDataDownloader
from .config import DownloadConfig, DataSourceConfig, ProcessingMode, ConfigLoader, create_legacy_config
from .validators import TickerValidator, DataValidator

__all__ = [
    "StockDataDownloader",
    "DownloadConfig", 
    "DataSourceConfig",
    "ProcessingMode",
    "ConfigLoader",
    "create_legacy_config",
    "TickerValidator",
    "DataValidator",
] 