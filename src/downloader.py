"""
Main downloader class for fetching stock price data from Yahoo Finance.
Optimized for PostgreSQL database uploads and large-scale processing.
"""

import os
import time
import logging
import sys
import signal
import gc
import psutil
from pathlib import Path
from typing import List, Optional, Set, Tuple, Dict, Any
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from .config import DownloadConfig, DataSourceConfig, ProcessingMode
from .validators import TickerValidator, DataValidator


class StockDataDownloader:
    """Main class for downloading stock price data with PostgreSQL optimizations."""
    
    def __init__(self, config: DownloadConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.data_sources = config.data_sources
        self.interrupted = False
        self._memory_threshold = 0.85  # 85% memory usage threshold
        self._batch_size = 1000  # Batch size for memory management
        self._thread_lock = threading.Lock()
        
        # Setup signal handlers for graceful interruption
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Validate configuration
        if not self.data_sources:
            raise ValueError("No data sources configured. Use ConfigLoader to set up data sources.")
    
    def _signal_handler(self, signum, frame):
        """Handle interruption signals gracefully."""
        self.logger.warning("⚠️  Download interrupted by user")
        self.interrupted = True
    
    def _check_memory_usage(self) -> bool:
        """Check if memory usage is above threshold."""
        memory_percent = psutil.virtual_memory().percent / 100
        return memory_percent > self._memory_threshold
    
    def _force_garbage_collection(self):
        """Force garbage collection to free memory."""
        gc.collect()
        if self._check_memory_usage():
            self.logger.warning(f"High memory usage: {psutil.virtual_memory().percent:.1f}%")
    
    def _setup_logging(self) -> logging.Logger:
        """Configure enhanced logging with both file and console output."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_formatter = logging.Formatter("%(levelname)-8s | %(message)s")
        
        # File handler
        file_handler = logging.FileHandler(self.config.log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def load_tickers_from_source(self, source_config: DataSourceConfig) -> Tuple[List[str], str]:
        """Load and validate tickers from a specific data source with yfinance existence validation."""
        try:
            if not os.path.exists(source_config.file):
                self.logger.warning(f"File not found: {source_config.file} - skipping {source_config.description}")
                return [], source_config.description
            
            # Read CSV in chunks for large files
            chunk_size = 10000
            chunks = []
            
            for chunk in pd.read_csv(source_config.file, chunksize=chunk_size):
                chunks.append(chunk)
                if self._check_memory_usage():
                    self.logger.warning("High memory usage detected, processing chunks immediately")
                    break
            
            df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            del chunks  # Free memory immediately
            
            self.logger.info(f"Loaded {source_config.description}: {len(df)} rows, columns: {list(df.columns)}")
            
            if source_config.ticker_column not in df.columns:
                self.logger.error(f"Column '{source_config.ticker_column}' not found in {source_config.file}. Available: {list(df.columns)}")
                return [], source_config.description
            
            # Apply filters if specified
            if source_config.filter_column and source_config.filter_values:
                if source_config.filter_column in df.columns:
                    original_count = len(df)
                    df = df[df[source_config.filter_column].isin(source_config.filter_values)]
                    self.logger.info(f"Applied filter on '{source_config.filter_column}': {original_count} → {len(df)} rows")
                else:
                    self.logger.warning(f"Filter column '{source_config.filter_column}' not found, ignoring filter")
            
            # Extract and clean tickers
            raw_tickers = df[source_config.ticker_column].dropna().astype(str).tolist()
            del df  # Free memory immediately
            
            self.logger.info(f"Found {len(raw_tickers)} raw ticker entries in {source_config.description}")
            
            # Clean and validate tickers in batches
            clean_tickers = []
            invalid_format_tickers = []
            
            for i in range(0, len(raw_tickers), self._batch_size):
                batch = raw_tickers[i:i + self._batch_size]
                
                for ticker in batch:
                    cleaned = TickerValidator.clean(ticker)
                    if cleaned and TickerValidator.validate(cleaned):
                        clean_tickers.append(cleaned)
                    else:
                        invalid_format_tickers.append(ticker)
                
                # Force garbage collection every batch
                if i % (self._batch_size * 5) == 0:
                    self._force_garbage_collection()
            
            # Remove duplicates while preserving order
            unique_tickers = list(dict.fromkeys(clean_tickers))
            
            self.logger.info(f"Format validation: {len(unique_tickers)} valid, {len(invalid_format_tickers)} invalid format")
            
            # Validate symbol existence and get earliest available dates
            self.logger.info(f"🔍 Validating {len(unique_tickers)} symbols and detecting historical data ranges...")
            
            # Use smaller batches for yfinance validation to avoid overwhelming the API
            validation_batch_size = 50
            all_valid_symbols_with_dates = {}
            all_invalid_symbols = []
            
            for i in range(0, len(unique_tickers), validation_batch_size):
                batch = unique_tickers[i:i + validation_batch_size]
                self.logger.info(f"Validating batch {i//validation_batch_size + 1}/{(len(unique_tickers) + validation_batch_size - 1)//validation_batch_size}: {len(batch)} symbols")
                
                valid_batch_with_dates, invalid_batch = TickerValidator.batch_validate_with_date_ranges(
                    batch, 
                    max_workers=3,  # Conservative to avoid API rate limits
                    timeout=10      # Longer timeout for date range detection
                )
                
                all_valid_symbols_with_dates.update(valid_batch_with_dates)
                all_invalid_symbols.extend(invalid_batch)
                
                # Progress update with date range info
                if valid_batch_with_dates:
                    earliest_dates = list(valid_batch_with_dates.values())
                    min_date = min(earliest_dates).strftime('%Y-%m-%d')
                    max_date = max(earliest_dates).strftime('%Y-%m-%d')
                    self.logger.info(f"Batch results: {len(valid_batch_with_dates)} valid, {len(invalid_batch)} invalid")
                    self.logger.info(f"Date range: {min_date} to {max_date}")
                else:
                    self.logger.info(f"Batch results: 0 valid, {len(invalid_batch)} invalid")
                
                # Sleep between validation batches to be respectful to the API
                if i + validation_batch_size < len(unique_tickers):
                    time.sleep(2)
            
            # Store the date ranges for later use in downloads
            self._symbol_date_ranges = getattr(self, '_symbol_date_ranges', {})
            self._symbol_date_ranges.update(all_valid_symbols_with_dates)
            
            # Final summary with historical data statistics
            all_valid_symbols = list(all_valid_symbols_with_dates.keys())
            total_invalid = len(invalid_format_tickers) + len(all_invalid_symbols)
            
            self.logger.info(f"✅ Final validation results for {source_config.description}:")
            self.logger.info(f"   📊 Total symbols processed: {len(raw_tickers)}")
            self.logger.info(f"   ✅ Valid symbols (exist in yfinance): {len(all_valid_symbols)}")
            self.logger.info(f"   ❌ Invalid format: {len(invalid_format_tickers)}")
            self.logger.info(f"   ❌ Not found in yfinance: {len(all_invalid_symbols)}")
            self.logger.info(f"   📈 Success rate: {len(all_valid_symbols)/len(raw_tickers)*100:.1f}%")
            
            if all_valid_symbols_with_dates:
                # Historical data statistics
                earliest_dates = list(all_valid_symbols_with_dates.values())
                overall_min_date = min(earliest_dates)
                overall_max_date = max(earliest_dates)
                # Handle timezone-aware datetime comparison
                now = datetime.now()
                avg_years = sum((now - (date.replace(tzinfo=None) if date.tzinfo else date)).days for date in earliest_dates) / len(earliest_dates) / 365.25
                
                self.logger.info(f"📈 Historical data summary:")
                self.logger.info(f"   📅 Earliest available data: {overall_min_date.strftime('%Y-%m-%d')}")
                self.logger.info(f"   📅 Latest available data: {overall_max_date.strftime('%Y-%m-%d')}")
                self.logger.info(f"   📊 Average history per symbol: {avg_years:.1f} years")
            
            if all_invalid_symbols:
                self.logger.warning(f"Symbols not found in yfinance: {all_invalid_symbols[:10]}{'...' if len(all_invalid_symbols) > 10 else ''}")
            
            return all_valid_symbols, source_config.description
            
        except Exception as e:
            self.logger.error(f"Error loading {source_config.description} from {source_config.file}: {e}")
            return [], source_config.description
    
    def determine_processing_sources(self) -> List[str]:
        """Determine which data sources to process based on mode."""
        available_sources = []
        
        for source_name, config in self.data_sources.items():
            if config.enabled and os.path.exists(config.file):
                available_sources.append(source_name)
        
        mode = self.config.processing_mode
        
        if mode == ProcessingMode.AUTO:
            self.logger.info(f"Auto mode: Processing all available sources: {available_sources}")
            return available_sources
        elif mode == ProcessingMode.ALL:
            self.logger.info(f"All mode: Processing all enabled sources: {available_sources}")
            return available_sources
        elif mode == ProcessingMode.SPECIFIC:
            # Filter to only requested sources that are available
            requested_sources = [s for s in self.config.specific_sources if s in available_sources]
            if not requested_sources:
                self.logger.warning(f"None of the requested sources {self.config.specific_sources} are available")
            else:
                self.logger.info(f"Specific mode: Processing requested sources: {requested_sources}")
            return requested_sources
        else:
            self.logger.error(f"Invalid processing mode: {mode}")
            return available_sources
    
    def get_existing_tickers(self, output_dir: Path) -> Set[str]:
        """Get set of tickers that already have data files."""
        if not output_dir.exists():
            return set()
        
        existing = {f.stem for f in output_dir.glob("*.csv")}
        if existing:
            self.logger.info(f"Found {len(existing)} existing data files in {output_dir}")
        return existing
    
    def save_ticker_data(self, df: pd.DataFrame, ticker: str, output_dir: Path, source_config: DataSourceConfig) -> bool:
        """Save individual ticker data to CSV file with PostgreSQL-optimized unified schema."""
        try:
            if not DataValidator.validate(df, ticker, self.config.min_data_points, self.logger):
                return False
            
            clean_df = DataValidator.clean_dataframe(df)

            # Create PostgreSQL-optimized unified schema
            unified_df = pd.DataFrame()
            
            # Reset index to make 'Date' a column
            clean_df.reset_index(inplace=True)
            
            # PostgreSQL-optimized data types and formatting
            unified_df['date'] = pd.to_datetime(clean_df['Date']).dt.date  # DATE type
            unified_df['open'] = clean_df['Open'].astype('float64')  # NUMERIC type
            unified_df['high'] = clean_df['High'].astype('float64')
            unified_df['low'] = clean_df['Low'].astype('float64')
            unified_df['close'] = clean_df['Close'].astype('float64')
            unified_df['volume'] = clean_df['Volume'].astype('int64')  # BIGINT type
            
            unified_df['symbol'] = ticker.upper()  # VARCHAR, normalized to uppercase
            unified_df['asset_type'] = 'stock'  # VARCHAR
            unified_df['data_source'] = source_config.name  # VARCHAR
            unified_df['download_date'] = pd.to_datetime(datetime.now().date())  # DATE type
            
            # Handle Adj Close with proper NULL handling for PostgreSQL
            if 'Adj Close' in clean_df.columns:
                unified_df['adj_close'] = clean_df['Adj Close'].astype('float64')
            else:
                unified_df['adj_close'] = None
            
            unified_df['source'] = source_config.file  # TEXT type
            unified_df['transform_date'] = pd.to_datetime(datetime.now().date())  # DATE type
            
            if not clean_df.empty:
                unified_df['data_start_date'] = pd.to_datetime(clean_df['Date'].min().date())
                unified_df['data_end_date'] = pd.to_datetime(clean_df['Date'].max().date())
                unified_df['record_count'] = len(clean_df)  # INTEGER type
            else:
                unified_df['data_start_date'] = None
                unified_df['data_end_date'] = None
                unified_df['record_count'] = 0
            
            # Calculate data quality score (0.0 to 1.0)
            total_days = (unified_df['data_end_date'].iloc[0] - unified_df['data_start_date'].iloc[0]).days + 1 if not clean_df.empty else 0
            trading_days_estimate = total_days * 0.7  # Rough estimate of trading days
            quality_score = min(1.0, len(clean_df) / max(1, trading_days_estimate)) if trading_days_estimate > 0 else 0.0
            unified_df['data_quality_score'] = round(quality_score, 4)  # NUMERIC(5,4)
            
            unified_df['yfinance_version'] = yf.__version__  # VARCHAR
            
            # Crypto-specific columns (NULL for stocks)
            unified_df['exchange'] = None  # VARCHAR
            unified_df['trading_pair'] = None  # VARCHAR
            
            # Sort by date (newest first) for better database performance
            unified_df = unified_df.sort_values('date', ascending=False)
            
            # Save with PostgreSQL-friendly settings
            output_file = output_dir / f"{ticker}.csv"
            unified_df.to_csv(
                output_file, 
                index=False,
                date_format='%Y-%m-%d',  # PostgreSQL DATE format
                float_format='%.6f',  # Consistent precision
                na_rep='NULL'  # PostgreSQL NULL representation
            )
            
            self.logger.info(f"✓ Saved {ticker}: {len(unified_df)} data points to {ticker}.csv")
            return True
            
        except Exception as e:
            self.logger.error(f"✗ Error saving {ticker}: {e}")
            return False
    
    def download_individual_ticker(self, ticker: str, output_dir: Path, source_name: str) -> bool:
        """Download maximum historical data for a single ticker with enhanced error handling."""
        try:
            # Check if already downloaded
            output_file = output_dir / f"{ticker}.csv"
            if output_file.exists():
                return True
            
            # Download with timeout and retry logic
            yf_ticker = yf.Ticker(ticker)
            
            # Use the earliest available date for this symbol if we have it
            symbol_date_ranges = getattr(self, '_symbol_date_ranges', {})
            if ticker in symbol_date_ranges:
                earliest_date = symbol_date_ranges[ticker]
                start_date = earliest_date.strftime('%Y-%m-%d')
                # self.logger.debug(f"📅 Using detected earliest date for {ticker}: {start_date}")
            else:
                # Fallback to configured start date
                start_date = self.config.start_date
                # self.logger.debug(f"📅 Using configured start date for {ticker}: {start_date}")
            
            # Download maximum historical data available
            if self.config.end_date:
                df = yf_ticker.history(
                    start=start_date,
                    end=self.config.end_date,
                    auto_adjust=False,
                    prepost=False
                )
            else:
                # Try to get maximum data available
                try:
                    # First try with period="max" which often gets more data
                    df = yf_ticker.history(
                        period="max",
                        auto_adjust=False,
                        prepost=False
                    )
                    
                    # If that doesn't work or gives less data, try with start date
                    if df.empty or (ticker in symbol_date_ranges and len(df) < 100):
                        df_with_start = yf_ticker.history(
                            start=start_date,
                            auto_adjust=False,
                            prepost=False
                        )
                        # Use whichever gives more data
                        if len(df_with_start) > len(df):
                            df = df_with_start
                            
                except Exception:
                    # Fallback to start date method
                    df = yf_ticker.history(
                        start=start_date,
                        auto_adjust=False,
                        prepost=False
                    )
            
            if df.empty:
                self.logger.warning(f"Empty data returned for {ticker}")
                return False
            
            # Log the actual date range we got
            if not df.empty:
                actual_start = df.index.min().strftime('%Y-%m-%d')
                actual_end = df.index.max().strftime('%Y-%m-%d')
                years_of_data = (df.index.max() - df.index.min()).days / 365.25
                # self.logger.debug(f"📊 {ticker}: {len(df)} data points from {actual_start} to {actual_end} ({years_of_data:.1f} years)")
            
            # Save with source config
            source_config = self.data_sources[source_name]
            success = self.save_ticker_data(df, ticker, output_dir, source_config)
            
            # Clean up memory
            del df, yf_ticker
            
            return success
            
        except Exception as e:
            self.logger.error(f"✗ Error downloading {ticker}: {e}")
            return False
    
    def download_chunk_parallel(self, chunk: List[str], output_dir: Path, source_config: DataSourceConfig, max_workers: int = 3) -> Tuple[int, List[str]]:
        """Download a chunk of tickers in parallel with memory management."""
        successful = 0
        failed_tickers = []
        
        # Reduce chunk size if memory usage is high
        if self._check_memory_usage():
            chunk = chunk[:max(1, len(chunk) // 2)]
            self.logger.warning(f"High memory usage, reducing chunk size to {len(chunk)}")
        
        try:
            # Use ThreadPoolExecutor for parallel downloads
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all downloads
                future_to_ticker = {
                    executor.submit(self.download_individual_ticker, ticker, output_dir, source_config.name): ticker
                    for ticker in chunk
                }
                
                # Process completed downloads
                for future in as_completed(future_to_ticker):
                    if self.interrupted:
                        break
                        
                    ticker = future_to_ticker[future]
                    try:
                        if future.result():
                            successful += 1
                            # Progress checkpoint
                            if successful % self.config.progress_checkpoint_interval == 0:
                                self.logger.info(f"📊 Progress checkpoint: {successful} tickers downloaded successfully")
                        else:
                            failed_tickers.append(ticker)
                    except Exception as e:
                        self.logger.error(f"✗ Parallel download failed for {ticker}: {e}")
                        failed_tickers.append(ticker)
                    
                    # Sleep between individual downloads
                    time.sleep(self.config.individual_sleep)
            
            # Force garbage collection after chunk
            self._force_garbage_collection()
            
        except Exception as e:
            self.logger.error(f"✗ Chunk download failed: {e}")
            failed_tickers.extend([t for t in chunk if t not in [future_to_ticker[f] for f in future_to_ticker if f.done()]])
        
        return successful, failed_tickers

    def download_chunk(self, chunk: List[str], output_dir: Path, source_config: DataSourceConfig) -> Tuple[int, List[str]]:
        """Download a chunk of tickers with fallback to individual downloads."""
        successful = 0
        failed_tickers = []
        
        try:
            # Try bulk download first for better performance
            tickers_str = " ".join(chunk)
            
            # Download with memory-conscious settings and maximum historical data
            # For bulk downloads, we'll use period="max" to get the most data
            # This is more efficient than trying to determine individual start dates
            if self.config.end_date:
                data = yf.download(
                    tickers_str,
                    start=self.config.start_date,
                    end=self.config.end_date,
                    group_by='ticker',
                    auto_adjust=False,
                    prepost=False,
                    threads=True,
                    progress=False  # Disable yfinance progress bar
                )
            else:
                # Try to get maximum historical data for all symbols
                try:
                    data = yf.download(
                        tickers_str,
                        period="max",
                        group_by='ticker',
                        auto_adjust=False,
                        prepost=False,
                        threads=True,
                        progress=False
                    )
                except Exception:
                    # Fallback to configured start date
                    data = yf.download(
                        tickers_str,
                        start=self.config.start_date,
                        group_by='ticker',
                        auto_adjust=False,
                        prepost=False,
                        threads=True,
                        progress=False
                    )
            
            if data.empty:
                self.logger.warning(f"Empty data returned for chunk of {len(chunk)} tickers")
                return 0, chunk
            
            # Process each ticker in the chunk
            for ticker in chunk:
                if self.interrupted:
                    break
                
                try:
                    # Check if already exists
                    output_file = output_dir / f"{ticker}.csv"
                    if output_file.exists():
                        successful += 1
                        continue
                    
                    # Extract ticker data
                    if len(chunk) == 1:
                        ticker_data = data
                    else:
                        ticker_data = data[ticker] if ticker in data.columns.get_level_values(0) else pd.DataFrame()
                    
                    if ticker_data.empty:
                        failed_tickers.append(ticker)
                        continue
                    
                    # Save ticker data
                    if self.save_ticker_data(ticker_data, ticker, output_dir, source_config):
                        successful += 1
                    else:
                        failed_tickers.append(ticker)
                        
                except Exception as e:
                    self.logger.error(f"✗ Error processing {ticker} in chunk: {e}")
                    failed_tickers.append(ticker)
            
            # Clean up memory
            del data
            self._force_garbage_collection()
            
        except Exception as e:
            self.logger.error(f"✗ Chunk download failed, falling back to individual downloads: {e}")
            # Fallback to individual downloads
            return self.download_chunk_parallel(chunk, output_dir, source_config, max_workers=2)
        
        return successful, failed_tickers
    
    def get_adaptive_chunk_size(self, source_name: str, total_tickers: int) -> int:
        """Get adaptive chunk size based on source type and system resources."""
        # Base chunk size from config
        base_chunk_size = self.config.chunk_size
        
        # Adjust based on memory usage
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 80:
            base_chunk_size = max(1, base_chunk_size // 4)
        elif memory_percent > 60:
            base_chunk_size = max(1, base_chunk_size // 2)
        
        # Adjust based on total number of tickers
        if total_tickers > 10000:
            base_chunk_size = max(1, base_chunk_size // 2)
        elif total_tickers > 25000:
            base_chunk_size = max(1, base_chunk_size // 4)
        
        return base_chunk_size
    
    def get_adaptive_sleep_time(self, source_name: str) -> int:
        """Get adaptive sleep time based on source type and recent performance."""
        base_sleep = self.config.sleep_between_chunks
        
        # Increase sleep time for large datasets to be more conservative
        if source_name and 'unified' in source_name.lower():
            return base_sleep * 2
        
        return base_sleep
    
    def process_data_source(self, source_name: str, source_config: DataSourceConfig) -> Dict[str, Any]:
        """Process a single data source and return summary statistics."""
        self.logger.info("=" * 80)
        self.logger.info(f"PROCESSING: {source_config.description}")
        self.logger.info("=" * 80)
        
        # Load tickers
        tickers, description = self.load_tickers_from_source(source_config)
        if not tickers:
            self.logger.warning(f"No valid tickers found for {description}")
            return {
                "source": source_name, 
                "description": description,
                "total": 0, 
                "processed": 0, 
                "successful": 0, 
                "failed": 0,
                "output_dir": str(Path(source_config.output_dir).absolute())
            }
        
        # Setup output directory
        output_dir = Path(source_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory: {output_dir.absolute()}")
        
        # Filter out already downloaded tickers
        existing_tickers = self.get_existing_tickers(output_dir)
        to_download = [t for t in tickers if t not in existing_tickers]
        
        self.logger.info(f"Total tickers: {len(tickers)}")
        self.logger.info(f"Already downloaded: {len(existing_tickers)}")
        self.logger.info(f"Remaining to download: {len(to_download)}")
        
        if not to_download:
            self.logger.info("🎉 All tickers already downloaded!")
            return {
                "source": source_name,
                "description": description,
                "total": len(tickers),
                "processed": 0,
                "successful": len(existing_tickers),
                "failed": 0,
                "output_dir": str(output_dir.absolute())
            }
        
        # Process in chunks
        adaptive_chunk_size = self.get_adaptive_chunk_size(source_name, len(to_download))
        chunks = [to_download[i:i + adaptive_chunk_size] for i in range(0, len(to_download), adaptive_chunk_size)]
        total_chunks = len(chunks)
        total_success = 0
        all_failed_tickers = []
        consecutive_chunk_failures = 0
        
        self.logger.info(f"Processing {total_chunks} chunks with adaptive size {adaptive_chunk_size}...")
        
        # Progress bar for chunks
        with tqdm(total=len(to_download), desc=f"Downloading {description}", unit="ticker") as pbar:
            for idx, chunk in enumerate(chunks, 1):
                # Check for interruption
                if self.interrupted:
                    self.logger.info("🛑 Download interrupted - saving progress and exiting gracefully")
                    break
                    
                self.logger.info(f"=== Chunk {idx}/{total_chunks}: {len(chunk)} tickers ===")
                
                chunk_success, chunk_failed = self.download_chunk(chunk, output_dir, source_config)
                
                total_success += chunk_success
                all_failed_tickers.extend(chunk_failed)
                pbar.update(chunk_success)
                
                # Track consecutive failures for early individual fallback
                if chunk_success == 0:
                    consecutive_chunk_failures += 1
                else:
                    consecutive_chunk_failures = 0
                
                # Early fallback to individual downloads if chunks consistently fail
                if consecutive_chunk_failures >= self.config.early_fallback_threshold and idx < total_chunks:
                    remaining_chunks = chunks[idx:]
                    remaining_tickers = [ticker for chunk in remaining_chunks for ticker in chunk]
                    self.logger.warning(f"⚠️  {consecutive_chunk_failures} consecutive chunk failures detected")
                    self.logger.info(f"🔄 Switching to individual downloads for remaining {len(remaining_tickers)} tickers")
                    
                    # Process remaining tickers individually
                    individual_success = 0
                    with tqdm(total=len(remaining_tickers), desc="Individual fallback", unit="ticker") as individual_pbar:
                        for ticker in remaining_tickers:
                            # Check for interruption in individual downloads too
                            if self.interrupted:
                                self.logger.info("🛑 Individual download interrupted - saving progress")
                                break
                            if self.download_individual_ticker(ticker, output_dir, source_name):
                                individual_success += 1
                                total_success += 1
                            individual_pbar.update(1)
                            pbar.update(1)
                    
                    self.logger.info(f"Individual fallback results: {individual_success}/{len(remaining_tickers)} successful")
                    break
                
                # Sleep between chunks (except for the last one)
                if idx < total_chunks and not self.interrupted:
                    self.logger.info(f"Sleeping {self.config.sleep_between_chunks}s before next chunk...")
                    time.sleep(self.config.sleep_between_chunks)
        
        # Retry failed tickers individually (only if not interrupted and we haven't already done individual fallback)
        individual_success = 0
        if all_failed_tickers and not self.interrupted and consecutive_chunk_failures < self.config.early_fallback_threshold:
            self.logger.info(f"\n=== Retrying {len(all_failed_tickers)} failed tickers individually ===")
            
            with tqdm(total=len(all_failed_tickers), desc="Individual retry", unit="ticker") as pbar:
                for ticker in all_failed_tickers:
                    # Check for interruption
                    if self.interrupted:
                        self.logger.info("🛑 Individual retry interrupted - saving progress")
                        break
                    if self.download_individual_ticker(ticker, output_dir, source_name):
                        individual_success += 1
                    pbar.update(1)
            
            total_success += individual_success
            final_failed = len(all_failed_tickers) - individual_success
            
            self.logger.info(f"Individual retry results: {individual_success} successful, {final_failed} failed")
        
        # Source summary
        self.logger.info("=" * 60)
        self.logger.info(f"SUMMARY: {description}")
        self.logger.info(f"Total tickers processed: {len(to_download)}")
        self.logger.info(f"Successfully downloaded: {total_success}")
        self.logger.info(f"Failed downloads: {len(to_download) - total_success}")
        self.logger.info(f"Success rate: {total_success/len(to_download)*100:.1f}%")
        self.logger.info(f"Data saved to: {output_dir.absolute()}")
        self.logger.info("=" * 60)
        
        return {
            "source": source_name,
            "description": description,
            "total": len(tickers),
            "processed": len(to_download),
            "successful": total_success + len(existing_tickers),
            "failed": len(to_download) - total_success,
            "output_dir": str(output_dir.absolute())
        }
    
    def run(self) -> None:
        """Main execution method with PostgreSQL optimization and memory management."""
        start_time = time.time()
        
        # Log system information for optimization tracking
        memory_info = psutil.virtual_memory()
        self.logger.info("🚀 Starting PostgreSQL-optimized multi-source stock data downloader")
        self.logger.info(f"System Memory: {memory_info.total / (1024**3):.1f}GB total, {memory_info.available / (1024**3):.1f}GB available")
        self.logger.info(f"Configuration: {self.config.start_date} to {self.config.end_date or 'present'}")
        self.logger.info(f"Processing mode: {self.config.processing_mode.value}")
        
        # Determine which sources to process
        sources_to_process = self.determine_processing_sources()
        
        if not sources_to_process:
            self.logger.error("No data sources available for processing")
            return
        
        self.logger.info(f"✓ Found {len(sources_to_process)} data source(s):")
        for source_name in sources_to_process:
            config = self.data_sources[source_name]
            self.logger.info(f"  • {config.description}")
            self.logger.info(f"    File: {config.file}")
            self.logger.info(f"    Ticker column: {config.ticker_column}")
            self.logger.info(f"    Output: {config.output_dir}")
        
        # Process each data source with memory monitoring
        results = []
        for i, source_name in enumerate(sources_to_process, 1):
            if self.interrupted:
                self.logger.info("🛑 Processing interrupted by user")
                break
            
            # Log memory usage before processing each source
            current_memory = psutil.virtual_memory().percent
            self.logger.info(f"Memory usage before source {i}/{len(sources_to_process)}: {current_memory:.1f}%")
            
            source_config = self.data_sources[source_name]
            result = self.process_data_source(source_name, source_config)
            results.append(result)
            
            # Force garbage collection between sources
            self._force_garbage_collection()
            
            # Log memory usage after processing
            post_memory = psutil.virtual_memory().percent
            self.logger.info(f"Memory usage after source {i}/{len(sources_to_process)}: {post_memory:.1f}%")
        
        # Final summary
        self._log_final_summary(results)
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"⏱️  Total execution time: {elapsed_time:.1f} seconds")
        self.logger.info("🎉 PostgreSQL-optimized download process completed!")
    
    def _log_final_summary(self, results: List[Dict[str, Any]]) -> None:
        """Log comprehensive final summary with PostgreSQL upload guidance."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("COMPREHENSIVE DOWNLOAD SUMMARY")
        self.logger.info("=" * 80)
        
        total_tickers = sum(r["total"] for r in results)
        total_processed = sum(r["processed"] for r in results)
        total_successful = sum(r["successful"] for r in results)
        total_failed = sum(r["failed"] for r in results)
        
        self.logger.info(f"Total tickers across all sources: {total_tickers}")
        self.logger.info(f"Total processed (new downloads): {total_processed}")
        self.logger.info(f"Total successful downloads: {total_successful}")
        self.logger.info(f"Total failed downloads: {total_failed}")
        
        if total_tickers > 0:
            overall_success_rate = (total_successful / total_tickers) * 100
            self.logger.info(f"Overall success rate: {overall_success_rate:.1f}%")
        
        self.logger.info("\nPer-source breakdown:")
        total_files = 0
        for result in results:
            self.logger.info(f"  📊 {result['description']}")
            self.logger.info(f"     Total: {result['total']}, Successful: {result['successful']}, Failed: {result['failed']}")
            if result['total'] > 0:
                success_rate = (result['successful'] / result['total']) * 100
                self.logger.info(f"     Success rate: {success_rate:.1f}%")
            self.logger.info(f"     Output: {result['output_dir']}")
            
            # Count CSV files for PostgreSQL upload estimation
            output_path = Path(result['output_dir'])
            if output_path.exists():
                csv_count = len(list(output_path.glob("*.csv")))
                total_files += csv_count
                self.logger.info(f"     CSV files ready for PostgreSQL: {csv_count}")
        
        # PostgreSQL upload guidance
        self.logger.info("\n" + "=" * 80)
        self.logger.info("POSTGRESQL UPLOAD GUIDANCE")
        self.logger.info("=" * 80)
        self.logger.info(f"Total CSV files ready for database upload: {total_files}")
        self.logger.info("Data format: PostgreSQL-optimized with proper data types")
        self.logger.info("Column naming: snake_case for PostgreSQL compatibility")
        self.logger.info("Date format: YYYY-MM-DD (PostgreSQL DATE type)")
        self.logger.info("NULL handling: 'NULL' strings for proper PostgreSQL import")
        self.logger.info("Numeric precision: 6 decimal places for consistent formatting")
        
        if total_files > 0:
            estimated_rows = total_files * 1000  # Rough estimate
            self.logger.info(f"Estimated total rows for database: ~{estimated_rows:,}")
            
            # Suggest batch upload strategy
            if total_files > 1000:
                self.logger.info("💡 Recommendation: Use batch upload strategy for large dataset")
                self.logger.info("   - Process files in batches of 100-500 CSV files")
                self.logger.info("   - Use COPY command for optimal PostgreSQL performance")
                self.logger.info("   - Consider parallel uploads for multiple sources")
            else:
                self.logger.info("💡 Recommendation: Standard upload approach suitable")
        
        self.logger.info("=" * 80) 