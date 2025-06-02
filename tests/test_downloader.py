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
    
    def load_tickers_from_source(self, source_config):
        """Load tickers but limit to max_tickers_per_source for testing."""
        tickers, description = super().load_tickers_from_source(source_config)
        
        if tickers and len(tickers) > self.max_tickers_per_source:
            limited_tickers = tickers[:self.max_tickers_per_source]
            self.logger.info(f"🧪 TEST MODE: Limited {description} to {len(limited_tickers)} tickers for testing")
            return limited_tickers, description
        
        return tickers, description


def create_test_config() -> DownloadConfig:
    """Create test configuration with limited data sources."""
    # Create test data sources with test output directories
    test_data_sources = {
        "iwb_holdings": DataSourceConfig(
            name="iwb_holdings",
            file="data/IWB_holdings_250529.csv",
            ticker_column="Ticker",
            description="IWB Russell 1000 Holdings (Individual Stocks) - TEST",
            output_dir="tests/test_outputs/test_stock_data",
            enabled=True
        ),
        "etf_list": DataSourceConfig(
            name="etf_list",
            file="data/etf_list.csv",
            ticker_column="Symbol",
            description="ETF Universe - TEST",
            output_dir="tests/test_outputs/test_etf_data",
            enabled=True
        )
    }
    
    # Configuration for testing
    return DownloadConfig(
        start_date="2020-01-01",  # Shorter date range for faster testing
        end_date=None,
        processing_mode=ProcessingMode.ALL,
        chunk_size=3,  # Smaller chunks for testing
        sleep_between_chunks=1,  # Shorter sleep for testing
        individual_sleep=0.5,
        max_retries=2,  # Fewer retries for testing
        initial_backoff=1,
        min_data_points=10,
        log_file="tests/logs/test_download.log",
        data_sources=test_data_sources
    )


def test_auto_discovery():
    """Test the auto-discovery functionality."""
    print("\n🔍 Testing auto-discovery functionality...")
    
    from src.config import ConfigLoader
    
    try:
        # Test auto-discovery
        config = ConfigLoader.auto_discover_csv_files("data", "tests/test_outputs/auto_discovery")
        
        print(f"✓ Auto-discovery found {len(config.data_sources)} data sources:")
        for name, source in config.data_sources.items():
            print(f"  • {source.description}")
            print(f"    File: {source.file}")
            print(f"    Ticker column: {source.ticker_column}")
            print(f"    Output: {source.output_dir}")
        
        return config
        
    except Exception as e:
        print(f"❌ Auto-discovery test failed: {e}")
        return None


def test_json_config():
    """Test JSON configuration loading."""
    print("\n📄 Testing JSON configuration...")
    
    from src.config import ConfigLoader
    
    try:
        # Create a test config file
        test_config_path = "tests/test_config.json"
        ConfigLoader.create_sample_config_file(test_config_path)
        
        # Load the config
        config = ConfigLoader.from_json_file(test_config_path)
        
        print(f"✓ JSON config loaded with {len(config.data_sources)} data sources")
        
        # Clean up
        Path(test_config_path).unlink(missing_ok=True)
        
        return True
        
    except Exception as e:
        print(f"❌ JSON config test failed: {e}")
        return False


def test_legacy_config():
    """Test legacy configuration compatibility."""
    print("\n🔄 Testing legacy configuration...")
    
    from src.config import create_legacy_config
    
    try:
        config = create_legacy_config()
        
        print(f"✓ Legacy config created with {len(config.data_sources)} data sources:")
        for name, source in config.data_sources.items():
            print(f"  • {name}: {source.description}")
        
        return config
        
    except Exception as e:
        print(f"❌ Legacy config test failed: {e}")
        return None


def main():
    """Run comprehensive tests of the generalized downloader."""
    print("🧪 Running comprehensive tests of the generalized downloader...")
    print("=" * 80)
    
    # Ensure test directories exist
    Path("tests/logs").mkdir(parents=True, exist_ok=True)
    Path("tests/test_outputs").mkdir(parents=True, exist_ok=True)
    
    # Test 1: Auto-discovery
    auto_config = test_auto_discovery()
    
    # Test 2: JSON configuration
    json_test_passed = test_json_config()
    
    # Test 3: Legacy configuration
    legacy_config = test_legacy_config()
    
    # Test 4: Limited download test
    print("\n💾 Testing limited download with test configuration...")
    
    try:
        test_config = create_test_config()
        
        # Create test downloader with limited tickers
        downloader = TestStockDataDownloader(test_config, max_tickers_per_source=3)
        
        print(f"✓ Test downloader created with {len(test_config.data_sources)} data sources")
        print("🚀 Starting limited download test...")
        
        downloader.run()
        
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Download test failed: {e}")
        raise


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Test error: {e}")
        sys.exit(1) 