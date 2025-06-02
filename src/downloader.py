"""
Main downloader class for fetching stock price data from Yahoo Finance.
"""

import os
import time
import logging
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple, Dict, Any

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from .config import DownloadConfig, DataSourceConfig, ProcessingMode
from .validators import TickerValidator, DataValidator


class StockDataDownloader:
    """Main class for downloading stock price data."""
    
    def __init__(self, config: DownloadConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.data_sources = config.data_sources
        
        # Validate configuration
        if not self.data_sources:
            raise ValueError("No data sources configured. Use ConfigLoader to set up data sources.")
    
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
        """Load and validate tickers from a specific data source."""
        try:
            if not os.path.exists(source_config.file):
                self.logger.warning(f"File not found: {source_config.file} - skipping {source_config.description}")
                return [], source_config.description
            
            df = pd.read_csv(source_config.file)
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
            self.logger.info(f"Found {len(raw_tickers)} raw ticker entries in {source_config.description}")
            
            # Clean and validate tickers
            clean_tickers = []
            invalid_tickers = []
            
            for ticker in raw_tickers:
                cleaned = TickerValidator.clean(ticker)
                if cleaned and TickerValidator.validate(cleaned):
                    clean_tickers.append(cleaned)
                else:
                    invalid_tickers.append(ticker)
            
            # Remove duplicates while preserving order
            unique_tickers = list(dict.fromkeys(clean_tickers))
            
            self.logger.info(f"{source_config.description}: {len(unique_tickers)} valid, {len(invalid_tickers)} invalid tickers")
            if invalid_tickers:
                self.logger.warning(f"Invalid tickers in {source_config.description}: {invalid_tickers[:5]}{'...' if len(invalid_tickers) > 5 else ''}")
            
            return unique_tickers, source_config.description
            
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
    
    def save_ticker_data(self, df: pd.DataFrame, ticker: str, output_dir: Path) -> bool:
        """Save individual ticker data to CSV file with clean formatting."""
        try:
            if not DataValidator.validate(df, ticker, self.config.min_data_points, self.logger):
                return False
            
            clean_df = DataValidator.clean_dataframe(df)
            
            # Save to CSV with clean formatting
            output_path = output_dir / f"{ticker}.csv"
            clean_df.to_csv(output_path, index=True, float_format='%.4f')
            
            self.logger.info(f"✓ Saved {ticker}: {len(clean_df)} data points to {output_path.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving {ticker}: {e}")
            return False
    
    def download_individual_ticker(self, ticker: str, output_dir: Path) -> bool:
        """Download data for a single ticker with retry logic."""
        attempt = 0
        backoff = self.config.initial_backoff
        
        while attempt < self.config.max_retries:
            try:
                self.logger.info(f"→ Downloading {ticker} (attempt {attempt + 1})")
                
                data = yf.download(
                    ticker,
                    start=self.config.start_date,
                    end=self.config.end_date,
                    progress=False,
                    auto_adjust=False,
                    threads=False
                )
                
                if not data.empty:
                    success = self.save_ticker_data(data, ticker, output_dir)
                    if success:
                        time.sleep(self.config.individual_sleep)
                        return True
                
                self.logger.warning(f"Empty data returned for {ticker}")
                return False
                
            except Exception as e:
                attempt += 1
                if attempt >= self.config.max_retries:
                    self.logger.error(f"✗ Failed to download {ticker} after {self.config.max_retries} attempts: {e}")
                    return False
                else:
                    self.logger.warning(f"✗ Error downloading {ticker}: {e} — retrying in {backoff}s")
                    time.sleep(backoff)
                    backoff *= 2
        
        return False
    
    def download_chunk(self, chunk: List[str], output_dir: Path) -> Tuple[int, List[str]]:
        """Download data for a chunk of tickers."""
        attempt = 0
        backoff = self.config.initial_backoff
        
        while attempt < self.config.max_retries:
            try:
                self.logger.info(f"→ Downloading chunk of {len(chunk)} tickers (attempt {attempt + 1})")
                
                data = yf.download(
                    tickers=chunk,
                    start=self.config.start_date,
                    end=self.config.end_date,
                    group_by="ticker",
                    threads=True,
                    progress=False,
                    auto_adjust=False,
                )
                
                if data.empty:
                    self.logger.warning("Empty data returned for entire chunk")
                    break
                
                success_count = 0
                failed_tickers = []
                
                # Handle single ticker case
                if len(chunk) == 1:
                    ticker = chunk[0]
                    if self.save_ticker_data(data, ticker, output_dir):
                        success_count = 1
                    else:
                        failed_tickers.append(ticker)
                else:
                    # Multiple tickers - data has MultiIndex columns
                    for ticker in chunk:
                        try:
                            if ticker in data.columns.levels[1]:
                                ticker_data = data.xs(ticker, axis=1, level=1)
                                if self.save_ticker_data(ticker_data, ticker, output_dir):
                                    success_count += 1
                                else:
                                    failed_tickers.append(ticker)
                            else:
                                self.logger.warning(f"No data found for {ticker} in chunk response")
                                failed_tickers.append(ticker)
                        except Exception as e:
                            self.logger.warning(f"Error processing {ticker} from chunk: {e}")
                            failed_tickers.append(ticker)
                
                self.logger.info(f"✓ Chunk completed: {success_count} successful, {len(failed_tickers)} failed")
                return success_count, failed_tickers
                
            except Exception as e:
                attempt += 1
                if attempt >= self.config.max_retries:
                    self.logger.error(f"✗ Chunk failed after {self.config.max_retries} attempts: {e}")
                    return 0, chunk
                else:
                    self.logger.warning(f"✗ Chunk error: {e} — retrying in {backoff}s")
                    time.sleep(backoff)
                    backoff *= 2
        
        return 0, chunk
    
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
        chunks = [to_download[i:i + self.config.chunk_size] for i in range(0, len(to_download), self.config.chunk_size)]
        total_chunks = len(chunks)
        total_success = 0
        all_failed_tickers = []
        
        self.logger.info(f"Processing {total_chunks} chunks...")
        
        # Progress bar for chunks
        with tqdm(total=len(to_download), desc=f"Downloading {description}", unit="ticker") as pbar:
            for idx, chunk in enumerate(chunks, 1):
                self.logger.info(f"=== Chunk {idx}/{total_chunks}: {len(chunk)} tickers ===")
                
                success_count, failed_tickers = self.download_chunk(chunk, output_dir)
                
                total_success += success_count
                all_failed_tickers.extend(failed_tickers)
                pbar.update(success_count)
                
                # Sleep between chunks (except for the last one)
                if idx < total_chunks:
                    self.logger.info(f"Sleeping {self.config.sleep_between_chunks}s before next chunk...")
                    time.sleep(self.config.sleep_between_chunks)
        
        # Retry failed tickers individually
        individual_success = 0
        if all_failed_tickers:
            self.logger.info(f"\n=== Retrying {len(all_failed_tickers)} failed tickers individually ===")
            
            with tqdm(total=len(all_failed_tickers), desc="Individual retry", unit="ticker") as pbar:
                for ticker in all_failed_tickers:
                    if self.download_individual_ticker(ticker, output_dir):
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
        """Main execution method."""
        # Log startup information
        self.logger.info("=" * 80)
        self.logger.info("Enhanced Multi-Source Stock Price Data Downloader Started")
        self.logger.info(f"Processing Mode: {self.config.processing_mode.value}")
        self.logger.info(f"Configuration: START_DATE={self.config.start_date}, END_DATE={self.config.end_date}")
        self.logger.info(f"Chunk size: {self.config.chunk_size}, Sleep: {self.config.sleep_between_chunks}s")
        self.logger.info("=" * 80)
        
        # Determine which sources to process
        sources_to_process = self.determine_processing_sources()
        
        if not sources_to_process:
            self.logger.error("No valid data sources found to process. Check file paths and configuration.")
            sys.exit(1)
        
        self.logger.info(f"Will process {len(sources_to_process)} data source(s): {sources_to_process}")
        
        # Process each data source
        all_results = []
        
        for source_name in sources_to_process:
            if source_name not in self.data_sources:
                self.logger.warning(f"Unknown data source: {source_name}")
                continue
            
            source_config = self.data_sources[source_name]
            
            try:
                result = self.process_data_source(source_name, source_config)
                all_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error processing {source_name}: {e}")
                continue
        
        # Final comprehensive summary
        self._log_final_summary(all_results)
    
    def _log_final_summary(self, results: List[Dict[str, Any]]) -> None:
        """Log comprehensive final summary."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("COMPREHENSIVE DOWNLOAD SUMMARY")
        self.logger.info("=" * 80)
        
        total_tickers = sum(r["total"] for r in results)
        total_successful = sum(r["successful"] for r in results)
        total_failed = sum(r["failed"] for r in results)
        
        for result in results:
            self.logger.info(f"{result['description']}:")
            self.logger.info(f"  • Total tickers: {result['total']}")
            self.logger.info(f"  • Successfully downloaded: {result['successful']}")
            self.logger.info(f"  • Failed: {result['failed']}")
            self.logger.info(f"  • Success rate: {result['successful']/result['total']*100:.1f}%")
            self.logger.info(f"  • Output directory: {result['output_dir']}")
            self.logger.info("")
        
        self.logger.info(f"OVERALL TOTALS:")
        self.logger.info(f"  • Total tickers across all sources: {total_tickers}")
        self.logger.info(f"  • Total successful downloads: {total_successful}")
        self.logger.info(f"  • Total failed downloads: {total_failed}")
        self.logger.info(f"  • Overall success rate: {total_successful/total_tickers*100:.1f}%")
        self.logger.info("=" * 80)
        
        if total_failed == 0:
            self.logger.info("🎉 All downloads completed successfully across all sources!")
        else:
            self.logger.warning(f"⚠️  {total_failed} tickers failed to download across all sources")
        
        self.logger.info(f"Complete log saved to: {self.config.log_file}") 