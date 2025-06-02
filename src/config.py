"""
Configuration classes and enums for the stock data downloader.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ProcessingMode(Enum):
    """Available processing modes for data sources."""
    STOCKS = "stocks"
    ETFS = "etfs"
    BOTH = "both"
    AUTO = "auto"


@dataclass
class DataSourceConfig:
    """Configuration for a data source."""
    file: str
    ticker_column: str
    description: str
    output_dir: str
    enabled: bool = True


@dataclass
class DownloadConfig:
    """Main configuration for the downloader."""
    start_date: str = "2000-01-01"
    end_date: Optional[str] = None
    processing_mode: ProcessingMode = ProcessingMode.BOTH
    chunk_size: int = 25
    sleep_between_chunks: int = 5
    individual_sleep: int = 2
    max_retries: int = 3
    initial_backoff: int = 2
    min_data_points: int = 10
    log_file: str = "multi_source_download.log" 