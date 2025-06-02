#!/usr/bin/env python3
"""
Enhanced Multi-Source Stock Price Data Downloader

Main entry point for the generalized stock price data downloader.
Supports multiple configuration methods:
1. Auto-discovery of CSV files
2. JSON configuration file
3. Legacy hardcoded configuration
"""

import sys
import argparse
from pathlib import Path

from src import StockDataDownloader
from src.config import DownloadConfig, ConfigLoader, ProcessingMode, create_legacy_config


def create_sample_config():
    """Create a sample configuration file."""
    ConfigLoader.create_sample_config_file("config.json")
    print("\n📝 Sample configuration file created!")
    print("You can now:")
    print("1. Edit config.json to customize your data sources")
    print("2. Run: python main.py --config config.json")


def auto_discover_mode(data_dir: str = "data", output_dir: str = "output"):
    """Auto-discover CSV files and run downloader."""
    try:
        print(f"🔍 Auto-discovering CSV files in '{data_dir}' directory...")
        config = ConfigLoader.auto_discover_csv_files(data_dir, output_dir)
        
        print(f"\n✓ Found {len(config.data_sources)} data source(s):")
        for name, source in config.data_sources.items():
            print(f"  • {source.description}")
            print(f"    File: {source.file}")
            print(f"    Ticker column: {source.ticker_column}")
            print(f"    Output: {source.output_dir}")
        
        return config
        
    except Exception as e:
        print(f"❌ Auto-discovery failed: {e}")
        print("\nTry one of these options:")
        print("1. Create a config file: python main.py --create-config")
        print("2. Use legacy mode: python main.py --legacy")
        sys.exit(1)


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Enhanced Multi-Source Stock Price Data Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Auto-discover CSV files in 'data/' directory
  python main.py --config config.json     # Use JSON configuration file
  python main.py --legacy                 # Use legacy hardcoded configuration
  python main.py --create-config          # Create sample configuration file
  python main.py --data-dir my_data       # Auto-discover in custom directory
  python main.py --sources etfs,stocks    # Process specific sources only
        """
    )
    
    # Configuration options
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument("--config", "-c", help="JSON configuration file path")
    config_group.add_argument("--legacy", "-l", action="store_true", 
                             help="Use legacy hardcoded configuration (IWB + ETF)")
    config_group.add_argument("--create-config", action="store_true",
                             help="Create sample configuration file and exit")
    
    # Auto-discovery options
    parser.add_argument("--data-dir", "-d", default="data",
                       help="Directory to search for CSV files (default: data)")
    parser.add_argument("--output-dir", "-o", default="output",
                       help="Base output directory (default: output)")
    
    # Processing options
    parser.add_argument("--sources", "-s", 
                       help="Comma-separated list of specific sources to process")
    parser.add_argument("--start-date", default="2000-01-01",
                       help="Start date for data download (YYYY-MM-DD)")
    parser.add_argument("--end-date", 
                       help="End date for data download (YYYY-MM-DD, default: today)")
    
    # Performance options
    parser.add_argument("--chunk-size", type=int, default=25,
                       help="Number of tickers per batch (default: 25)")
    parser.add_argument("--sleep", type=int, default=5,
                       help="Seconds between chunks (default: 5)")
    
    parser.add_argument(
        "--stock-optimized",
        action="store_true",
        help="Use optimized settings for individual stock downloads (smaller chunks, longer delays)"
    )
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.create_config:
        create_sample_config()
        return
    
    # Load configuration
    config = None
    
    if args.config:
        # Load from JSON file
        if not Path(args.config).exists():
            print(f"❌ Configuration file not found: {args.config}")
            sys.exit(1)
        
        try:
            config = ConfigLoader.from_json_file(args.config)
            print(f"✓ Loaded configuration from: {args.config}")
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            sys.exit(1)
    
    elif args.legacy:
        # Use legacy configuration
        config = create_legacy_config()
        print("✓ Using legacy configuration (IWB + ETF)")
    
    else:
        # Auto-discover mode
        config = auto_discover_mode(args.data_dir, args.output_dir)
    
    # Apply command-line overrides
    if args.sources:
        source_list = [s.strip() for s in args.sources.split(',')]
        config.processing_mode = ProcessingMode.SPECIFIC
        config.specific_sources = source_list
        print(f"✓ Processing specific sources: {source_list}")
    
    if args.start_date:
        config.start_date = args.start_date
    
    if args.end_date:
        config.end_date = args.end_date
    
    if args.chunk_size:
        config.chunk_size = args.chunk_size
    
    if args.sleep:
        config.sleep_between_chunks = args.sleep
    
    # Apply stock-optimized settings if requested
    if args.stock_optimized:
        config.stock_chunk_size = 3  # Very small chunks for stocks
        config.etf_chunk_size = 15   # Moderate chunks for ETFs  
        config.early_fallback_threshold = 2  # Switch to individual faster
        config.stock_individual_sleep = 4  # Longer delays for stocks
        config.sleep_between_chunks = max(config.sleep_between_chunks, 8)  # Longer chunk delays
        print("🎯 Using stock-optimized settings:")
        print(f"   Stock chunks: {config.stock_chunk_size}, ETF chunks: {config.etf_chunk_size}")
        print(f"   Early fallback after: {config.early_fallback_threshold} failures")
        print(f"   Stock sleep: {config.stock_individual_sleep}s, Chunk sleep: {config.sleep_between_chunks}s")
    
    # Validate configuration
    if not config.data_sources:
        print("❌ No data sources configured!")
        print("Try: python main.py --create-config")
        sys.exit(1)
    
    # Create and run downloader
    try:
        downloader = StockDataDownloader(config)
        downloader.run()
    except KeyboardInterrupt:
        print("\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 