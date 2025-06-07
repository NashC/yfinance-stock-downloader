"""
Configuration classes and enums for the generalized stock data downloader.
Optimized for PostgreSQL database uploads and large-scale processing.
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
    """Configuration for the stock data downloader with PostgreSQL optimizations."""
    start_date: str = "2000-01-01"
    end_date: Optional[str] = None
    processing_mode: ProcessingMode = ProcessingMode.ALL
    specific_sources: List[str] = field(default_factory=list)
    
    # Performance settings optimized for large datasets
    chunk_size: int = 10  # Smaller chunks for better memory management
    sleep_between_chunks: int = 3  # Conservative rate limiting
    individual_sleep: int = 1  # Faster individual downloads
    max_retries: int = 3
    initial_backoff: int = 2
    min_data_points: int = 10
    log_file: str = "stock_download.log"
    data_sources: Dict[str, DataSourceConfig] = field(default_factory=dict)
    
    # Memory management settings
    memory_threshold: float = 0.85  # 85% memory usage threshold
    batch_size: int = 1000  # Batch size for ticker processing
    force_gc_interval: int = 100  # Force garbage collection every N tickers
    
    # PostgreSQL-specific settings
    postgres_batch_size: int = 500  # Optimal batch size for PostgreSQL COPY
    postgres_date_format: str = "%Y-%m-%d"  # PostgreSQL DATE format
    postgres_float_precision: int = 6  # Decimal precision for floats
    postgres_null_value: str = "NULL"  # NULL representation
    
    # Large dataset optimizations
    large_dataset_threshold: int = 10000  # Threshold for large dataset optimizations
    adaptive_chunk_sizing: bool = True  # Enable adaptive chunk sizing
    parallel_downloads: bool = True  # Enable parallel downloads for chunks
    max_parallel_workers: int = 3  # Maximum parallel workers
    
    # Resume and progress tracking
    enable_resume: bool = True  # Skip already downloaded files
    progress_checkpoint_interval: int = 100  # Log progress every N tickers
    
    # Early fallback settings for failed chunks
    early_fallback_threshold: int = 3  # Switch to individual after N consecutive chunk failures


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
        """Auto-discover CSV files in a directory and create configuration with optimizations."""
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        data_sources = {}
        
        for csv_file in data_path.glob("**/*.csv"):  # Recursive search
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
        
        # Optimize configuration based on discovered sources
        config = DownloadConfig(
            data_sources=data_sources,
            processing_mode=ProcessingMode.ALL
        )
        
        # Apply optimizations for large datasets
        total_estimated_tickers = 0
        for source_name, source_config in data_sources.items():
            try:
                # Quick estimate of ticker count
                df_sample = pd.read_csv(source_config.file, nrows=100)
                if source_config.ticker_column in df_sample.columns:
                    # Rough estimate based on sample
                    file_size = Path(source_config.file).stat().st_size
                    sample_size = len(df_sample)
                    estimated_total = int((file_size / 1024) * sample_size / 10)  # Rough estimate
                    total_estimated_tickers += estimated_total
            except:
                pass
        
        # Apply large dataset optimizations
        if total_estimated_tickers > config.large_dataset_threshold:
            print(f"🔧 Large dataset detected (~{total_estimated_tickers:,} tickers). Applying optimizations...")
            config.chunk_size = max(5, config.chunk_size // 2)  # Smaller chunks
            config.sleep_between_chunks = max(2, config.sleep_between_chunks)  # Conservative timing
            config.memory_threshold = 0.80  # Lower memory threshold
            config.force_gc_interval = 50  # More frequent garbage collection
            print(f"   - Chunk size: {config.chunk_size}")
            print(f"   - Sleep between chunks: {config.sleep_between_chunks}s")
            print(f"   - Memory threshold: {config.memory_threshold*100:.0f}%")
        
        return config
    
    @staticmethod
    def create_sample_config_file(filename: str = "config.json") -> None:
        """Create a sample configuration file with PostgreSQL optimizations."""
        sample_config = {
            "start_date": "2000-01-01",
            "end_date": None,
            "processing_mode": "all",
            "specific_sources": [],
            
            # Performance settings optimized for large datasets
            "chunk_size": 10,
            "sleep_between_chunks": 3,
            "individual_sleep": 1,
            "max_retries": 3,
            "initial_backoff": 2,
            "min_data_points": 10,
            "log_file": "stock_download.log",
            
            # Memory management
            "memory_threshold": 0.85,
            "batch_size": 1000,
            "force_gc_interval": 100,
            
            # PostgreSQL-specific settings
            "postgres_batch_size": 500,
            "postgres_date_format": "%Y-%m-%d",
            "postgres_float_precision": 6,
            "postgres_null_value": "NULL",
            
            # Large dataset optimizations
            "large_dataset_threshold": 10000,
            "adaptive_chunk_sizing": True,
            "parallel_downloads": True,
            "max_parallel_workers": 3,
            
            # Resume and progress tracking
            "enable_resume": True,
            "progress_checkpoint_interval": 100,
            "early_fallback_threshold": 3,
            
            "data_sources": {
                "unified_symbols": {
                    "file": "data/symbol_lists/unified_symbols.csv",
                    "ticker_column": "Symbol",
                    "description": "Unified Symbol List (Large Dataset)",
                    "output_dir": "output/unified_symbols_data",
                    "enabled": True
                },
                "sp500": {
                    "file": "data/sp500_tickers.csv",
                    "ticker_column": "Symbol",
                    "description": "S&P 500 Companies",
                    "output_dir": "output/sp500_data",
                    "enabled": False
                },
                "etfs": {
                    "file": "data/etf_list.csv",
                    "ticker_column": "Symbol",
                    "description": "ETF Universe",
                    "output_dir": "output/etf_data",
                    "enabled": False
                }
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(sample_config, f, indent=2)
        
        print(f"✓ PostgreSQL-optimized configuration created: {filename}")
        print("Edit this file to customize your data sources and settings.")
        print("Configuration is optimized for large datasets and PostgreSQL uploads.")


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
        processing_mode=ProcessingMode.ALL,
        # Use conservative settings for legacy compatibility
        chunk_size=5,
        sleep_between_chunks=5,
        individual_sleep=3
    ) 