#!/usr/bin/env python3
"""
Enhanced Multi-Source Stock Price Data Downloader

Main entry point for the stock price data downloader.
"""

import sys

from src import StockDataDownloader, DownloadConfig, ProcessingMode


def main():
    """Main entry point."""
    # Configuration with optimized rate limiting
    config = DownloadConfig(
        start_date="2000-01-01",
        end_date=None,
        processing_mode=ProcessingMode.BOTH,
        chunk_size=25,
        sleep_between_chunks=5,  # ~12 requests/minute for chunks
        individual_sleep=2,      # ~30 requests/minute for retries
        max_retries=3,
        initial_backoff=2,
        min_data_points=10,
        log_file="multi_source_download.log"
    )
    
    # Create and run downloader
    downloader = StockDataDownloader(config)
    downloader.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        sys.exit(1) 