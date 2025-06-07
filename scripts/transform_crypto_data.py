#!/usr/bin/env python3
"""
Crypto Data Transformation Script

Transforms crypto price data from various exchanges into a standardized format
with additional metadata columns for better organization and analysis.

Features:
- Handles crypto exchange data format (unix timestamp, different volume columns)
- Extracts exchange and trading pair information from filenames
- Adds metadata columns: symbol, exchange, trading_pair, data_source, download_date
- Standardizes date format and column structure
- Comprehensive logging and progress tracking
"""

import pandas as pd
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import re
from tqdm import tqdm

def setup_logging(log_file: str = "crypto_data_transformation.log") -> logging.Logger:
    """Setup comprehensive logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_crypto_filename(filename: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Parse crypto data filename to extract exchange and trading pair information.
    
    Expected format: historical_price_data_daily_{exchange}_{trading_pair}.csv
    Example: historical_price_data_daily_gemini_BTCUSD.csv
    
    Returns:
        Tuple of (exchange, trading_pair, symbol)
    """
    # Remove .csv extension
    name = filename.replace('.csv', '')
    
    # Pattern to match: historical_price_data_daily_{exchange}_{trading_pair}
    pattern = r'historical_price_data_daily_([^_]+)_([A-Z0-9]+)$'
    match = re.match(pattern, name)
    
    if match:
        exchange = match.group(1)
        trading_pair = match.group(2)
        
        # Extract base symbol (e.g., BTC from BTCUSD)
        # Common patterns: BTCUSD, ETHUSD, ATOMUSDT, etc.
        symbol_patterns = [
            r'^([A-Z0-9]+)USD[T]?$',  # BTCUSD, ETHUSD, ATOMUSDT
            r'^([A-Z0-9]+)EUR[T]?$',  # BTCEUR, ETHEUR
            r'^([A-Z0-9]+)GBP[T]?$',  # BTCGBP
        ]
        
        symbol = None
        for pattern in symbol_patterns:
            symbol_match = re.match(pattern, trading_pair)
            if symbol_match:
                symbol = symbol_match.group(1)
                break
        
        if not symbol:
            # Fallback: use trading pair as symbol
            symbol = trading_pair
            
        return exchange, trading_pair, symbol
    
    return None, None, None

def standardize_crypto_data(df: pd.DataFrame, exchange: str, trading_pair: str, symbol: str, download_date: str) -> pd.DataFrame:
    """
    Standardize crypto data format and add metadata columns.
    
    Input format: unix,date,symbol,open,high,low,close,Volume BTC,Volume USD
    Output format: Date,Open,High,Low,Close,Volume,Symbol,Exchange,Trading_Pair,Data_Source,Download_Date
    """
    # Create a copy to avoid modifying original
    standardized_df = df.copy()
    
    # Rename columns to match standard format
    column_mapping = {
        'date': 'Date',
        'open': 'Open',
        'high': 'High', 
        'low': 'Low',
        'close': 'Close'
    }
    
    # Handle volume - prefer USD volume if available, otherwise use the first volume column
    volume_columns = [col for col in df.columns if 'volume' in col.lower() or 'Volume' in col]
    if volume_columns:
        # Prefer USD volume
        usd_volume_cols = [col for col in volume_columns if 'USD' in col]
        if usd_volume_cols:
            column_mapping[usd_volume_cols[0]] = 'Volume'
        else:
            column_mapping[volume_columns[0]] = 'Volume'
    
    # Rename columns
    standardized_df = standardized_df.rename(columns=column_mapping)
    
    # Ensure Date column is properly formatted
    if 'Date' in standardized_df.columns:
        standardized_df['Date'] = pd.to_datetime(standardized_df['Date']).dt.strftime('%Y-%m-%d')
    
    # Add metadata columns
    standardized_df['Symbol'] = symbol
    standardized_df['Exchange'] = exchange.title()  # Capitalize exchange name
    standardized_df['Trading_Pair'] = trading_pair
    standardized_df['Data_Source'] = f"{exchange.title()} Exchange"
    standardized_df['Download_Date'] = download_date
    
    # Select and reorder columns
    output_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 
                     'Symbol', 'Exchange', 'Trading_Pair', 'Data_Source', 'Download_Date']
    
    # Only include columns that exist
    available_columns = [col for col in output_columns if col in standardized_df.columns]
    standardized_df = standardized_df[available_columns]
    
    # Sort by date (newest first)
    if 'Date' in standardized_df.columns:
        standardized_df = standardized_df.sort_values('Date', ascending=False)
    
    return standardized_df

def transform_crypto_directory(input_dir: str, output_dir: str, download_date: str, sample_mode: bool = False) -> Dict[str, int]:
    """
    Transform all crypto CSV files in the input directory.
    
    Returns:
        Dictionary with transformation statistics
    """
    logger = logging.getLogger(__name__)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files
    csv_files = list(input_path.glob("*.csv"))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {input_dir}")
        return {"total_files": 0, "successful": 0, "failed": 0}
    
    logger.info(f"Found {len(csv_files)} CSV files to transform")
    
    if sample_mode:
        csv_files = csv_files[:5]  # Limit to 5 files for sampling
        logger.info(f"Sample mode: Processing only {len(csv_files)} files")
    
    stats = {"total_files": len(csv_files), "successful": 0, "failed": 0}
    failed_files = []
    
    # Process files with progress bar
    for csv_file in tqdm(csv_files, desc="Transforming crypto data"):
        try:
            # Parse filename to extract metadata
            exchange, trading_pair, symbol = parse_crypto_filename(csv_file.name)
            
            if not all([exchange, trading_pair, symbol]):
                logger.warning(f"Could not parse filename: {csv_file.name} - skipping")
                stats["failed"] += 1
                failed_files.append(csv_file.name)
                continue
            
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            if df.empty:
                logger.warning(f"Empty file: {csv_file.name} - skipping")
                stats["failed"] += 1
                failed_files.append(csv_file.name)
                continue
            
            # Transform data
            transformed_df = standardize_crypto_data(df, exchange, trading_pair, symbol, download_date)
            
            # Generate output filename
            output_filename = f"{symbol}_{exchange}.csv"
            output_file_path = output_path / output_filename
            
            # Save transformed data
            transformed_df.to_csv(output_file_path, index=False)
            
            logger.info(f"✓ Transformed {csv_file.name} -> {output_filename} ({len(transformed_df)} rows)")
            stats["successful"] += 1
            
        except Exception as e:
            logger.error(f"✗ Failed to transform {csv_file.name}: {str(e)}")
            stats["failed"] += 1
            failed_files.append(csv_file.name)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"CRYPTO DATA TRANSFORMATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total files processed: {stats['total_files']}")
    logger.info(f"Successful transformations: {stats['successful']}")
    logger.info(f"Failed transformations: {stats['failed']}")
    logger.info(f"Success rate: {(stats['successful']/stats['total_files']*100):.1f}%")
    logger.info(f"Output directory: {output_dir}")
    
    if failed_files:
        logger.warning(f"Failed files: {', '.join(failed_files)}")
    
    return stats

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Transform crypto exchange data into standardized format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transform all crypto data
  python transform_crypto_data.py --input-dir output/20250405_crypto_data --output-dir output/transformed_crypto_data --download-date 2025-04-05
  
  # Sample mode (process only 5 files)
  python transform_crypto_data.py --input-dir output/20250405_crypto_data --output-dir output/sample_crypto_data --download-date 2025-04-05 --sample
        """
    )
    
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing crypto CSV files to transform"
    )
    
    parser.add_argument(
        "--output-dir", 
        required=True,
        help="Directory to save transformed CSV files"
    )
    
    parser.add_argument(
        "--download-date",
        required=True,
        help="Date when the data was downloaded (YYYY-MM-DD format)"
    )
    
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample mode: process only first 5 files for testing"
    )
    
    parser.add_argument(
        "--log-file",
        default="crypto_data_transformation.log",
        help="Log file path (default: crypto_data_transformation.log)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    
    try:
        # Validate download date format
        datetime.strptime(args.download_date, '%Y-%m-%d')
        
        logger.info(f"Starting crypto data transformation...")
        logger.info(f"Input directory: {args.input_dir}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Download date: {args.download_date}")
        logger.info(f"Sample mode: {args.sample}")
        
        # Transform data
        stats = transform_crypto_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            download_date=args.download_date,
            sample_mode=args.sample
        )
        
        logger.info("Crypto data transformation completed successfully!")
        
    except ValueError as e:
        logger.error(f"Invalid date format: {args.download_date}. Use YYYY-MM-DD format.")
        return 1
    except Exception as e:
        logger.error(f"Transformation failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 