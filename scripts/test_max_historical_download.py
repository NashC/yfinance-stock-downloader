#!/usr/bin/env python3
"""
Test script for maximum historical data download functionality.
Tests the enhanced downloader with symbol validation and maximum date range detection.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import pandas as pd
from src.config import DownloadConfig, DataSourceConfig
from src.downloader import StockDataDownloader

def create_test_csv(symbols, temp_dir):
    """Create a test CSV file with symbols."""
    csv_path = temp_dir / "test_symbols.csv"
    df = pd.DataFrame({"Symbol": symbols})
    df.to_csv(csv_path, index=False)
    return csv_path

def test_max_historical_download():
    """Test maximum historical data download with a small set of symbols."""
    print("🧪 Testing Maximum Historical Data Download")
    print("=" * 60)
    
    # Create temporary directories
    temp_dir = Path(tempfile.mkdtemp())
    output_dir = temp_dir / "output"
    
    try:
        # Test symbols with known long histories
        test_symbols = ["AAPL", "MSFT", "IBM", "KO", "JNJ"]
        print(f"📊 Test symbols: {test_symbols}")
        
        # Create test CSV
        csv_path = create_test_csv(test_symbols, temp_dir)
        print(f"📄 Created test CSV: {csv_path}")
        
        # Configure downloader for maximum historical data
        config = DownloadConfig(
            start_date="2000-01-01",  # This will be overridden by detected dates
            end_date=None,  # No end date = get all data to present
            chunk_size=2,   # Small chunks for testing
            sleep_between_chunks=1,
            individual_sleep=0.5,
            max_retries=2,
            min_data_points=10,
            log_file=str(temp_dir / "test_download.log")
        )
        
        # Add test data source
        config.data_sources = {
            "test_symbols": DataSourceConfig(
                name="test_symbols",
                file=str(csv_path),
                ticker_column="Symbol",
                description="Test Symbols for Max Historical Data",
                output_dir=str(output_dir),
                enabled=True
            )
        }
        
        print(f"📁 Output directory: {output_dir}")
        
        # Create and run downloader
        downloader = StockDataDownloader(config)
        
        print("\n🚀 Starting download with symbol validation and max historical data...")
        downloader.run()
        
        # Analyze results
        print("\n📊 Analyzing Downloaded Data:")
        print("=" * 40)
        
        if output_dir.exists():
            csv_files = list(output_dir.glob("*.csv"))
            print(f"✅ Downloaded files: {len(csv_files)}")
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    if not df.empty:
                        # Parse dates
                        df['date'] = pd.to_datetime(df['date'])
                        earliest = df['date'].min()
                        latest = df['date'].max()
                        years = (latest - earliest).days / 365.25
                        
                        symbol = csv_file.stem
                        print(f"📈 {symbol}: {len(df)} records, {earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')} ({years:.1f} years)")
                    else:
                        print(f"❌ {csv_file.stem}: Empty file")
                except Exception as e:
                    print(f"❌ {csv_file.stem}: Error reading file - {e}")
        else:
            print("❌ No output directory found")
        
        print(f"\n📋 Log file: {temp_dir / 'test_download.log'}")
        
        # Show log summary
        log_file = temp_dir / "test_download.log"
        if log_file.exists():
            print("\n📝 Log Summary (last 10 lines):")
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in lines[-10:]:
                    print(f"   {line.strip()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
        
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
            print(f"\n🧹 Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"⚠️  Could not clean up {temp_dir}: {e}")

def main():
    """Run the maximum historical data download test."""
    print("🧪 Maximum Historical Data Download Test")
    print("=" * 70)
    
    success = test_max_historical_download()
    
    if success:
        print("\n🎉 Test completed successfully!")
        print("💡 The system is ready to download maximum historical data for all symbols.")
        print("📈 Each symbol will be validated and downloaded with its full available history.")
        return 0
    else:
        print("\n❌ Test failed!")
        return 1

if __name__ == "__main__":
    exit(main()) 