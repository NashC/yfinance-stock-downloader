#!/usr/bin/env python3
"""
Test version of the Enhanced Multi-Source Stock Price Data Downloader

This version processes only a small subset of tickers for testing purposes.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import StockDataDownloader, DownloadConfig, DataSourceConfig, ProcessingMode


class TestStockDataDownloader(StockDataDownloader):
    """Test version that limits the number of tickers processed."""
    
    def __init__(self, config: DownloadConfig, max_tickers_per_source: int = 5):
        super().__init__(config)
        self.max_tickers_per_source = max_tickers_per_source
    
    def _initialize_data_sources(self):
        """Initialize data source configurations with test output directories."""
        return {
            "iwb_holdings": DataSourceConfig(
                file="data/IWB_holdings_250529.csv",
                ticker_column="Ticker",
                description="IWB Russell 1000 Holdings (Individual Stocks)",
                output_dir="tests/test_stock_data",
                enabled=True
            ),
            "etf_list": DataSourceConfig(
                file="data/etf_list.csv",
                ticker_column="Symbol",
                description="ETF Universe",
                output_dir="tests/test_etf_data",
                enabled=True
            )
        }
    
    def load_tickers_from_source(self, source_config):
        """Load tickers but limit to max_tickers_per_source for testing."""
        tickers, description = super().load_tickers_from_source(source_config)
        
        if tickers and len(tickers) > self.max_tickers_per_source:
            limited_tickers = tickers[:self.max_tickers_per_source]
            self.logger.info(f"🧪 TEST MODE: Limited {description} to {len(limited_tickers)} tickers for testing")
            return limited_tickers, description
        
        return tickers, description


def main():
    """Test the refactored downloader with a small subset of tickers."""
    print("🧪 Running test version with limited tickers...")
    
    # Configuration for testing
    config = DownloadConfig(
        start_date="2020-01-01",  # Shorter date range for faster testing
        end_date=None,
        processing_mode=ProcessingMode.BOTH,
        chunk_size=5,  # Smaller chunks for testing
        sleep_between_chunks=2,  # Shorter sleep for testing
        individual_sleep=1,
        max_retries=2,  # Fewer retries for testing
        initial_backoff=1,
        min_data_points=10,
        log_file="tests/test_download.log"
    )
    
    # Create test downloader with limited tickers
    downloader = TestStockDataDownloader(config, max_tickers_per_source=5)
    downloader.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Test error: {e}")
        sys.exit(1) 