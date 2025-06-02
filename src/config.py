"""
Configuration classes and enums for the generalized stock data downloader.
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union


class ProcessingMode(Enum):
    """Available processing modes for data sources."""
    ALL = "all"  # Process all available sources
    SPECIFIC = "specific"  # Process specific sources by name
    AUTO = "auto"  # Auto-detect available sources


@dataclass
class DataSourceConfig:
    """Configuration for a data source."""
    name: str
    file: str
    ticker_column: str
    description: str
    output_dir: str
    enabled: bool = True
    # Optional filters
    filter_column: Optional[str] = None
    filter_values: Optional[List[str]] = None


@dataclass
class DownloadConfig:
    """Main configuration for the downloader."""
    # Date range
    start_date: str = "2000-01-01"
    end_date: Optional[str] = None
    
    # Processing configuration
    processing_mode: ProcessingMode = ProcessingMode.ALL
    specific_sources: List[str] = field(default_factory=list)  # Used with SPECIFIC mode
    
    # Performance settings
    chunk_size: int = 25
    sleep_between_chunks: int = 5
    individual_sleep: int = 2
    max_retries: int = 3
    initial_backoff: int = 2
    
    # Data validation
    min_data_points: int = 10
    
    # Logging
    log_file: str = "stock_download.log"
    
    # Data sources - can be loaded from config file or set programmatically
    data_sources: Dict[str, DataSourceConfig] = field(default_factory=dict)


class ConfigLoader:
    """Utility class for loading configuration from various sources."""
    
    @staticmethod
    def from_json_file(config_file: str) -> DownloadConfig:
        """Load configuration from JSON file."""
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # Parse data sources
        data_sources = {}
        for name, source_data in config_data.get('data_sources', {}).items():
            data_sources[name] = DataSourceConfig(
                name=name,
                **source_data
            )
        
        # Remove data_sources from config_data to avoid duplicate
        config_data.pop('data_sources', None)
        
        # Handle enum conversion
        if 'processing_mode' in config_data:
            config_data['processing_mode'] = ProcessingMode(config_data['processing_mode'])
        
        return DownloadConfig(
            data_sources=data_sources,
            **config_data
        )
    
    @staticmethod
    def auto_discover_csv_files(data_dir: str = "data", output_base_dir: str = "output") -> DownloadConfig:
        """Auto-discover CSV files in a directory and create configuration."""
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        data_sources = {}
        
        for csv_file in data_path.glob("*.csv"):
            # Try to auto-detect ticker column
            import pandas as pd
            try:
                df = pd.read_csv(csv_file, nrows=5)  # Read just a few rows to check columns
                
                # Common ticker column names
                ticker_columns = ['ticker', 'symbol', 'Ticker', 'Symbol', 'TICKER', 'SYMBOL']
                ticker_column = None
                
                for col in ticker_columns:
                    if col in df.columns:
                        ticker_column = col
                        break
                
                if ticker_column is None:
                    print(f"⚠️  Could not auto-detect ticker column in {csv_file.name}. Skipping.")
                    continue
                
                # Create data source config
                source_name = csv_file.stem.lower().replace(' ', '_')
                data_sources[source_name] = DataSourceConfig(
                    name=source_name,
                    file=str(csv_file),
                    ticker_column=ticker_column,
                    description=f"Auto-discovered: {csv_file.name}",
                    output_dir=f"{output_base_dir}/{source_name}_data",
                    enabled=True
                )
                
                print(f"✓ Auto-discovered: {csv_file.name} (ticker column: {ticker_column})")
                
            except Exception as e:
                print(f"⚠️  Error reading {csv_file.name}: {e}")
                continue
        
        if not data_sources:
            raise ValueError(f"No valid CSV files with ticker columns found in {data_dir}")
        
        return DownloadConfig(
            data_sources=data_sources,
            processing_mode=ProcessingMode.ALL
        )
    
    @staticmethod
    def create_sample_config_file(filename: str = "config.json") -> None:
        """Create a sample configuration file."""
        sample_config = {
            "start_date": "2000-01-01",
            "end_date": None,
            "processing_mode": "all",
            "specific_sources": [],
            "chunk_size": 25,
            "sleep_between_chunks": 5,
            "individual_sleep": 2,
            "max_retries": 3,
            "initial_backoff": 2,
            "min_data_points": 10,
            "log_file": "stock_download.log",
            "data_sources": {
                "sp500": {
                    "file": "data/sp500_tickers.csv",
                    "ticker_column": "Symbol",
                    "description": "S&P 500 Companies",
                    "output_dir": "output/sp500_data",
                    "enabled": True
                },
                "nasdaq100": {
                    "file": "data/nasdaq100.csv",
                    "ticker_column": "Ticker",
                    "description": "NASDAQ 100 Companies",
                    "output_dir": "output/nasdaq100_data",
                    "enabled": True,
                    "filter_column": "Market Cap",
                    "filter_values": ["Large Cap"]
                },
                "etfs": {
                    "file": "data/etf_list.csv",
                    "ticker_column": "Symbol",
                    "description": "ETF Universe",
                    "output_dir": "output/etf_data",
                    "enabled": True
                }
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(sample_config, f, indent=2)
        
        print(f"✓ Sample configuration created: {filename}")
        print("Edit this file to customize your data sources and settings.")


# Legacy compatibility - create default config similar to original
def create_legacy_config() -> DownloadConfig:
    """Create configuration compatible with the original hardcoded sources."""
    data_sources = {
        "iwb_holdings": DataSourceConfig(
            name="iwb_holdings",
            file="data/IWB_holdings_250529.csv",
            ticker_column="Ticker",
            description="IWB Russell 1000 Holdings (Individual Stocks)",
            output_dir="stock_data",
            enabled=True
        ),
        "etf_list": DataSourceConfig(
            name="etf_list",
            file="data/etf_list.csv",
            ticker_column="Symbol",
            description="ETF Universe",
            output_dir="etf_data",
            enabled=True
        )
    }
    
    return DownloadConfig(
        data_sources=data_sources,
        processing_mode=ProcessingMode.ALL
    ) 